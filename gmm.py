import numpy as np

class GaussianMixtureModel:

    def __init__(self, n_components=2, max_iter=100, tol=1e-4, random_state=None):

        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.pi = None
        self.mu = None
        self.sigma = None
        self.log_likelihood_history = []

    def ensure_2D(self, X):

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def init_params(self, X):

        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape

        self.pi = np.ones(self.n_components) / self.n_components

        random_idx = rng.choice(n_samples, self.n_components, replace=False)
        self.mu = X[random_idx].copy()

        data_var = np.var(X, axis=0).mean()
        self.sigma = np.array([ data_var * np.eye(n_features) for _ in range(self.n_components)])

    def gaussian_logpdf(self, X, mu, sigma):

        n_features = X.shape[1]
        sigma_reg = sigma + 1e-2 * np.eye(n_features)

        try:
            inv_sigma = np.linalg.pinv(sigma_reg)
            sign, log_det = np.linalg.slogdet(sigma_reg)
            if sign <= 0:
                log_det = np.log(1e-6)
        except np.linalg.LinAlgError:
            return np.full(X.shape[0], -1e10)

        diff = X - mu
        exponent = np.clip(-0.5 * np.sum(diff @ inv_sigma * diff, axis=1), -500, 0)
        log_norm = -0.5 * (n_features * np.log(2 * np.pi) + log_det)
        return log_norm + exponent

    def E_step(self, X):

        n_samples = X.shape[0]
        log_resp = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            log_resp[:, k] = np.log(self.pi[k] + 1e-300) + self.gaussian_logpdf(X, self.mu[k], self.sigma[k])

        log_resp_max = log_resp.max(axis=1, keepdims=True)
        log_resp_shifted = log_resp - log_resp_max
        responsibilities = np.exp(log_resp_shifted)
        row_sums = responsibilities.sum(axis=1, keepdims=True)

        zero_rows = (row_sums.squeeze() < 1e-300)
        responsibilities[zero_rows] = 1.0 / self.n_components
        row_sums[zero_rows] = 1.0

        responsibilities /= row_sums

        return responsibilities

    def M_step(self, X, responsibilities):

        n_samples, n_features = X.shape

        Nk = responsibilities.sum(axis=0)
        Nk = np.maximum(Nk, 1e-6)

        self.pi = Nk / n_samples
        self.mu = (responsibilities.T @ X) / Nk[:, None]

        self.sigma = []
        for k in range(self.n_components):
            diff = X - self.mu[k]
            weighted = responsibilities[:, k][:, None] * diff
            cov = weighted.T @ diff / Nk[k]

            self.sigma.append(cov + 1e-2 * np.eye(n_features))

        self.sigma = np.array(self.sigma)

    def compute_loglikelihood(self, X):
        n_samples = X.shape[0]
        log_liks = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            log_liks[:, k] = np.log(self.pi[k] + 1e-300) + self.gaussian_logpdf(X, self.mu[k], self.sigma[k])

        log_liks_max = log_liks.max(axis=1)
        log_sum = log_liks_max + np.log( np.exp(log_liks - log_liks_max[:, None]).sum(axis=1) + 1e-300)
        return np.sum(log_sum)

    def fit(self, X):

        X = self.ensure_2D(X)
        self.init_params(X)

        for _ in range(self.max_iter):
            responsibilities = self.E_step(X)
            self.M_step(X, responsibilities)

            log_likelihood = self.compute_loglikelihood(X)
            self.log_likelihood_history.append(log_likelihood)

            if len(self.log_likelihood_history) > 1:
                delta = abs(self.log_likelihood_history[-1] - self.log_likelihood_history[-2])
                if delta < self.tol:
                    break

        return self

    def score(self, X):
        X = self.ensure_2D(X)
        return self.compute_loglikelihood(X)