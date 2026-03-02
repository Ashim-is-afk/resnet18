import torch
import torch.nn.functional as F
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

def evaluate(model, test_loader, device, status_map, figo_map, ckpt_path="best_model.pth"):
    # --- CORRECTED LOADING LOGIC ---
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Check if the saved file is a dictionary (new way) or raw weights (old way)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()

    y_s_true, y_s_pred, y_s_probs = [], [], []
    y_f_true, y_f_pred, y_f_probs = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            s_out, f_out = model(images)

            # Probabilities for AUC-ROC
            s_p = F.softmax(s_out, dim=1).cpu().numpy()
            f_p = F.softmax(f_out, dim=1).cpu().numpy()
            y_s_probs.append(s_p)
            y_f_probs.append(f_p)

            # Labels
            y_s_true.extend(batch["status"].cpu().numpy().tolist())
            y_s_pred.extend(torch.argmax(s_out, dim=1).cpu().numpy().tolist())
            y_f_true.extend(batch["figo"].cpu().numpy().tolist())
            y_f_pred.extend(torch.argmax(f_out, dim=1).cpu().numpy().tolist())

    # Stack results
    y_s_probs = np.concatenate(y_s_probs, axis=0)
    y_f_probs = np.concatenate(y_f_probs, axis=0)
    y_s_true, y_s_pred = np.array(y_s_true), np.array(y_s_pred)
    y_f_true, y_f_pred = np.array(y_f_true), np.array(y_f_pred)

    # --- 1) Save Predictions TSV ---
    inv_status = {v: k for k, v in status_map.items()}
    inv_figo = {v: k for k, v in figo_map.items()}
    df_preds = pd.DataFrame({
        "Status_True": [inv_status[i] for i in y_s_true],
        "Status_Pred": [inv_status[i] for i in y_s_pred],
        "FIGO_True":   [inv_figo[i] for i in y_f_true],
        "FIGO_Pred":   [inv_figo[i] for i in y_f_pred],
    })
    df_preds.to_csv("test_predictions.tsv", sep="\t", index=False)

    # --- 2) ROC Plotting Function ---
    def plot_roc_task(name, true_labels, probs, label_map):
        labels = list(label_map.keys())
        n_classes = len(labels)
        true_bin = label_binarize(true_labels, classes=list(range(n_classes)))

        plt.figure(figsize=(8, 6))
        for i in range(n_classes):
            if true_bin[:, i].sum() == 0: continue
            fpr, tpr, _ = roc_curve(true_bin[:, i], probs[:, i])
            plt.plot(fpr, tpr, label=f"{labels[i]} (AUC = {auc(fpr, tpr):.2f})")
        
        plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"ROC Curve: {name}")
        plt.legend()
        plt.savefig(f"roc_{name}.png", dpi=200)
        plt.close()
        return roc_auc_score(true_labels, probs, multi_class="ovr")

    status_auc = plot_roc_task("Status", y_s_true, y_s_probs, status_map)
    figo_auc = plot_roc_task("FIGO", y_f_true, y_f_probs, figo_map)

    # --- 3) Confusion Matrices ---
    tasks = [("Status", y_s_true, y_s_pred, list(status_map.keys())), 
             ("FIGO", y_f_true, y_f_pred, list(figo_map.keys()))]

    for name, true, pred, labels in tasks:
        # Normalized by row (Recall)
        cm = confusion_matrix(true, pred, normalize="true")
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=".2f", xticklabels=labels, yticklabels=labels, cmap="Greens")
        plt.title(f"{name} Confusion Matrix (Normalized)")
        plt.savefig(f"confusion_{name}.png", dpi=200)
        plt.close()
        
        # Save Text Report
        report = classification_report(true, pred, target_names=labels, zero_division=0)
        with open(f"report_{name}.txt", "w") as f:
            f.write(report)

    print(f"\nEvaluation Finished. Status AUC: {status_auc:.4f}, FIGO AUC: {figo_auc:.4f}")