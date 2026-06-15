import numpy as np


class HMM:

    def __init__(self, n_states, n_symbols, max_iter=100, tol=1e-4, random_state=None):

        self.n_states  = n_states
        self.n_symbols = n_symbols
        self.max_iter  = max_iter
        self.tol       = tol
        self.random_state = random_state

        self.pi = None          
        self.A  = None            
        self.B  = None          
        self.log_likelihood_history = []

    def init_params(self):

        rng = np.random.default_rng(self.random_state)
        n, k = self.n_states, self.n_symbols

        self.pi = np.ones(n) / n
        self.A = np.zeros((n, n))
        for i in range(n):
            if i < n - 1:
                self.A[i, i]     = 0.5
                self.A[i, i + 1] = 0.5
            else:
                self.A[i, i] = 1.0

        raw = rng.random((n, k)) + 0.1
        self.B = raw / raw.sum(axis=1, keepdims=True) 

    def log_Emission(self, obs):
  
        log_B = np.log(self.B + 1e-300)           
        log_b = log_B[:, obs].T                   
        return log_b

    def logsumexp(self, a, axis=None):

        a_max = np.max(a, axis=axis, keepdims=True)
        a_max = np.where(np.isfinite(a_max), a_max, 0.0)
        out = np.log(np.sum(np.exp(a - a_max), axis=axis) + 1e-300)
        squeezed = np.squeeze(a_max, axis=axis) if axis is not None else a_max.item()

        return out + squeezed


    def log_forward(self, log_b):

        t, n = log_b.shape
        log_alpha = np.full((t, n), -np.inf)
        log_alpha[0] = np.log(self.pi + 1e-300) + log_b[0]

        log_A = np.log(self.A + 1e-300)        

        for i in range(1, t):
           
            acc = log_alpha[i - 1][:, None] + log_A   
            log_alpha[i] = log_b[i] + self.logsumexp(acc, axis=0)

        log_likelihood = self.logsumexp(log_alpha[-1])

        return log_alpha, log_likelihood


    def log_backward(self, log_b):

        t, n = log_b.shape
        log_beta = np.full((t, n), -np.inf)
        log_beta[-1] = 0.0                          

        log_A = np.log(self.A + 1e-300)             

        for i in reversed(range(t - 1)):
    
            acc = log_A + log_b[i + 1][None, :] + log_beta[i + 1][None, :]  
            log_beta[i] = self.logsumexp(acc, axis=1)

        return log_beta


    def Estep(self, obs):

        log_b = self.log_Emission(obs)
        log_alpha, log_likelihood = self.log_forward(log_b)
        log_beta  = self.log_backward(log_b)

        t, n = log_b.shape

        log_gamma = log_alpha + log_beta - log_likelihood     
        log_gamma -= self.logsumexp(log_gamma, axis=1)[:, None]
        gamma = np.exp(np.clip(log_gamma, -500, 0))
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300      

        log_A = np.log(self.A + 1e-300)
        log_xi_raw = (  log_alpha[:-1, :, None] + log_A[None, :, :] + log_b[1:, None, :] + log_beta[1:, None, :] - log_likelihood)                                                       

        flat_max = log_xi_raw.max(axis=(1, 2), keepdims=True)
        xi = np.exp(np.clip(log_xi_raw - flat_max, -500, 0))
        xi /= xi.sum(axis=(1, 2), keepdims=True) + 1e-300      

        return gamma, xi, log_likelihood

    def fit(self, obs_list):

        if isinstance(obs_list, np.ndarray) and obs_list.ndim == 1:
            obs_list = [obs_list]

        obs_list = [np.asarray(o, dtype=np.int32) for o in obs_list]
        self.init_params()
        n, k = self.n_states, self.n_symbols

        for _ in range(self.max_iter):

            pi_acc  = np.zeros(n)
            A_num ,A_den   = np.zeros((n, n)) , np.zeros(n)
            B_num ,B_den  = np.zeros((n, k)) , np.zeros(n)
            total_ll = 0.0
            for obs in obs_list:
                if len(obs) < 2:
                    continue
                gamma, xi, ll = self.Estep(obs)
                total_ll += ll
                pi_acc += gamma[0]
                A_num  += xi.sum(axis=0)
                A_den  += gamma[:-1].sum(axis=0)

                for i in range(k):
                    mask = (obs == i)              
                    B_num[:, i] += gamma[mask].sum(axis=0)

                B_den += gamma.sum(axis=0)

            self.pi = pi_acc / (pi_acc.sum() + 1e-300)

            for i in range(n):
                row = A_num[i] / (A_den[i] + 1e-300)
                row = np.maximum(row, 1e-10)
                self.A[i] = row / row.sum()

            for i in range(n):
                row = B_num[i] / (B_den[i] + 1e-300)
                row = np.maximum(row, 1e-10)      
                self.B[i] = row / row.sum()

            self.log_likelihood_history.append(total_ll)

            if len(self.log_likelihood_history) > 1:
                if abs(self.log_likelihood_history[-1] - self.log_likelihood_history[-2]) < self.tol:
                    break

        return self


    def score(self, obs):

        obs = np.asarray(obs, dtype=np.int32)
        log_b = self.log_Emission(obs)
        _, log_likelihood = self.log_forward(log_b)
        return log_likelihood


class VectorQuantiser:

    def __init__(self, n_symbols=64, max_iter=100, random_state=None):

        self.n_symbols   = n_symbols
        self.max_iter    = max_iter
        self.random_state = random_state
        self.codebook    = None   

    def fit(self, X_list):

        rng = np.random.default_rng(self.random_state)

        X_all = np.vstack(X_list).astype(np.float64)
        n_samples, n_features = X_all.shape
        K = self.n_symbols

        idx = rng.choice(n_samples, K, replace=False)
        self.codebook = X_all[idx].copy()        

        for iteration in range(self.max_iter):

            labels = self.assign_index(X_all)            
            new_codebook = np.zeros_like(self.codebook)
            for k in range(K):
                members = X_all[labels == k]
                if len(members) > 0:
                    new_codebook[k] = members.mean(axis=0)
                else:
                    new_codebook[k] = X_all[rng.integers(n_samples)]

            shift = np.linalg.norm(new_codebook - self.codebook)
            self.codebook = new_codebook
            if shift < 1e-6:
                break
        return self

    def assign_index(self, X):

        X_sq   = (X ** 2).sum(axis=1, keepdims=True)           
        C_sq   = (self.codebook ** 2).sum(axis=1, keepdims=True).T  
        cross  = X @ self.codebook.T                           
        distances  = X_sq - 2 * cross + C_sq                      
        return np.argmin(distances, axis=1).astype(np.int32)       

    def transform(self, X):

        X = np.asarray(X, dtype=np.float64)
        return self.assign_index(X)

    def fit_transform(self, X_list):
        
        self.fit(X_list)

        return [self.transform(X) for X in X_list]