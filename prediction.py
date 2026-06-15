import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from hmm import HMM, VectorQuantiser
from gmm import GaussianMixtureModel
from feature_extraction import extract_features, train_test_split


train_data = train_test_split()

all_X = np.vstack([feats for _, feats in train_data])

global_mean = all_X.mean(axis=0)
global_std  = all_X.std(axis=0) + 1e-6

train_norm = [(label, (feats - global_mean) / global_std) for label, feats in train_data]

vq = VectorQuantiser(n_symbols=64, max_iter=100, random_state=0)
vq.fit([feats for _, feats in train_norm])

train_discrete = [(label, vq.transform(feats)) for label, feats in train_norm]

hmm_models = {}

for digit in range(10):

    sequences = [obs for label, obs in train_discrete if label == digit]
    model = HMM(n_states=3, n_symbols=64, max_iter=100, random_state=digit)
    model.fit(sequences)
    hmm_models[digit] = model

gmm_models = {}

for digit in range(10):

    frames = [feats for label, feats in train_norm if label == digit]
    X_digit = np.vstack(frames)
    model = GaussianMixtureModel(n_components=4, max_iter=100, random_state=digit)
    model.fit(X_digit)
    gmm_models[digit] = model


digits = list(range(10))

def plot_cm(cm, title, ax):

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.where(row_sums > 0, cm / row_sums, 0.0)

    image = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Predicted digit", fontsize=9)
    ax.set_ylabel("True digit",      fontsize=9)
    ax.set_xticks(digits)
    ax.set_yticks(digits)
    ax.set_xticklabels(digits)
    ax.set_yticklabels(digits)

    thresh = cm_norm.max() / 2.0
    for i in range(len(digits)):
        for j in range(len(digits)):
            ax.text(j, i, str(cm[i, j]),ha="center", va="center", fontsize=8,color="white" if cm_norm[i, j] > thresh else "black")

    return image


def per_digit_comp(title, y_true, y_pred, split_name):
    print("=" * 46)
    print(f"  {title}  —  {split_name}")
    print("=" * 46)
    overall = accuracy_score(y_true, y_pred)
    print(f"  Overall Accuracy : {overall:.4f} {int(overall * len(y_true))}/{len(y_true)} correct)\n")

    cm = confusion_matrix(y_true, y_pred, labels=digits)
    print(f"  {'Digit':<8} {'Total':>6} {'Correct':>8} {'Accuracy':>10}")
    print(f"  {'-'*35}")
    for d in digits:
        total   = cm[d].sum()
        correct = cm[d, d]
        acc     = correct / total if total > 0 else 0.0
        print(f"  {d:<8} {total:>6} {correct:>8} {acc:>9.2%}")
    print()
    return cm


def prediction(folder): #inference function

    files = [f for f in os.listdir(folder) if f.endswith(".wav")]
    hmm_true, hmm_pred = [], []
    gmm_true, gmm_pred = [], []

    for file in files:
        file_path = os.path.join(folder, file)

        feats = extract_features(file_path)
        feats_norm = (feats - global_mean) / global_std
        obs = vq.transform(feats_norm)
        label = int(file.split("_")[0])

        if hmm_models:
            scores = {d: m.score(obs) / len(obs) for d, m in hmm_models.items()}
            hmm_true.append(label)
            hmm_pred.append(max(scores, key=scores.get))

        if gmm_models:
            scores = {d: m.score(feats_norm) / len(feats_norm) for d, m in gmm_models.items()}
            gmm_true.append(label)
            gmm_pred.append(max(scores, key=scores.get))

    return hmm_true, hmm_pred, gmm_true, gmm_pred

hmm_vt, hmm_vp, gmm_vt, gmm_vp = prediction("val_recordings")

cm_hmm_val = per_digit_comp("HMM", hmm_vt, hmm_vp, "Validation (speaker 4)")
cm_gmm_val = per_digit_comp("GMM", gmm_vt, gmm_vp, "Validation (speaker 4)")


hmm_tt, hmm_tp, gmm_tt, gmm_tp = prediction("test_recordings")

cm_hmm_test = per_digit_comp("HMM", hmm_tt, hmm_tp, "Test (speaker 6)")
cm_gmm_test = per_digit_comp("GMM", gmm_tt, gmm_tp, "Test (speaker 6)")


def save_cm(cm_val, cm_test, model_name, filename):

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{model_name} — Confusion Matrices (rows = true digit, cols = predicted digit)",
                 fontsize=13, fontweight="bold", y=1.02)

    splits = [(cm_val,  "Validation — speaker 4", axes[0]),(cm_test, "Test — speaker 6",axes[1]) ]

    for cm, title, ax in splits:
        if cm is not None:
            im = plot_cm(cm, title, ax)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Row-normalised proportion", fontsize=8)
        else:
            ax.axis("off")
            ax.set_title(title + "\n(no data)", fontsize=10)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()



save_cm(cm_hmm_val, cm_hmm_test,"HMM", "confusionmatrix_hmm.png")
save_cm(cm_gmm_val, cm_gmm_test,"GMM","confusionmatrix_gmm.png")