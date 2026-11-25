import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
)
from step5_null_model import fit_ridge_mvar


def run_step6_evaluation(
    y_hat_csv="Y_hat_eloreta.csv",
    g_true_csv="G_true.csv",
    thresh_csv="null_threshold.csv",
    alpha=0.5,
):
    print("=== Step 6: Metric Evaluation ===")
    Y_hat = np.loadtxt(y_hat_csv, delimiter=",")
    G_true = np.loadtxt(g_true_csv, delimiter=",", dtype=int)
    thresh = float(np.loadtxt(thresh_csv, delimiter=","))

    Y_hat_z = (Y_hat - np.mean(Y_hat, axis=1, keepdims=True)) / np.std(Y_hat, axis=1, keepdims=True)
    W_est = fit_ridge_mvar(Y_hat_z, alpha=alpha)

    G_est = (W_est > thresh).astype(int)
    np.fill_diagonal(G_est, 0)

    np.savetxt("W_est.csv", W_est, delimiter=",", fmt="%.6f")
    np.savetxt("G_est.csv", G_est, delimiter=",", fmt="%d")

    mask = ~np.eye(G_true.shape[0], dtype=bool)
    y_true = G_true[mask]
    y_pred = G_est[mask]
    y_score = W_est[mask]

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    print(f"Locked Threshold : {thresh:.6f}")
    print(f"TP / FP / FN     : {tp} / {fp} / {fn}")
    print(f"Accuracy         : {acc * 100:.2f}%")
    print(f"Precision        : {prec * 100:.2f}%")
    print(f"Recall           : {rec * 100:.2f}%")
    print(f"F1 Score         : {f1:.4f}")
    print(f"MCC              : {mcc:.4f}")
    print(f"AUROC            : {auroc:.4f}")
    print(f"Average Prec (AP): {ap:.4f}")


if __name__ == "__main__":
    run_step6_evaluation()