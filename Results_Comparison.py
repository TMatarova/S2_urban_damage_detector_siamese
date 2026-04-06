import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, roc_auc_score, precision_recall_curve, 
                             average_precision_score)
import os
import json
import pandas as pd

# Set directory to where all the models results are saved
OUT_DIR = r"C:\Users\Taula\Downloads\Ukraine cities\patches_siamese\outputs"

# Collect results of all models first
results = []

for i in range(8):  # 0 through 7
    # Load .npz file for ROC/PR curve data
    npz_path = os.path.join(OUT_DIR, f"model_{i}.npz")
    data = np.load(npz_path)
    labels, probs = data["labels"], data["probs"]

    # ROC and AUC
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    
    # Precision-Recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)
    
    # Load pre-calculated metrics from JSON
    metrics_path = os.path.join(OUT_DIR, f"Model_{i}", "metrics.json")
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    results.append({
        "model": i,
        "labels": labels,
        "probs": probs,
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc,
        "precision_curve": precision_curve,
        "recall_curve": recall_curve,
        "ap": ap,
        "accuracy": metrics.get('accuracy'),
        "no_damage_precision": metrics.get('No-Damage', {}).get('precision'),
        "no_damage_recall": metrics.get('No-Damage', {}).get('recall'),
        "no_damage_f1": metrics.get('No-Damage', {}).get('f1-score'),
        "no_damage_support": metrics.get('No-Damage', {}).get('support'),
        "damage_precision": metrics.get('Damage', {}).get('precision'),
        "damage_recall": metrics.get('Damage', {}).get('recall'),
        "damage_f1": metrics.get('Damage', {}).get('f1-score'),
        "damage_support": metrics.get('Damage', {}).get('support'),
        "macro_precision": metrics.get('macro avg', {}).get('precision'),
        "macro_recall": metrics.get('macro avg', {}).get('recall'),
        "macro_f1": metrics.get('macro avg', {}).get('f1-score'),
        "weighted_precision": metrics.get('weighted avg', {}).get('precision'),
        "weighted_recall": metrics.get('weighted avg', {}).get('recall'),
        "weighted_f1": metrics.get('weighted avg', {}).get('f1-score')
    })

# Sort by Average Precision (AP)
results_sorted = sorted(results, key=lambda x: x["ap"], reverse=True)

# Save rankings to CSV with all metrics
rankings = []
for rank, r in enumerate(results_sorted, start=1):
    rankings.append({
        "rank": rank,
        "model": int(r['model']),
        "AP": float(r['ap']),
        "AUC": float(r['auc']),
        "accuracy": float(r['accuracy']),
        "no_damage_precision": float(r['no_damage_precision']),
        "no_damage_recall": float(r['no_damage_recall']),
        "no_damage_f1_score": float(r['no_damage_f1']),
        "damage_precision": float(r['damage_precision']),
        "damage_recall": float(r['damage_recall']),
        "damage_f1_score": float(r['damage_f1'])
    })

# Convert to DataFrame and save as CSV
df = pd.DataFrame(rankings)
rankings_path = os.path.join(OUT_DIR, "model_rankings.csv")
df.to_csv(rankings_path, index=False)

to_plot = results_sorted

# ROC Plot
plt.figure(figsize=(8,6))
for res in to_plot:
    plt.plot(res["fpr"], res["tpr"], label=f"model {res['model']} (AUC={res['auc']:.3f})")

plt.plot([0,1], [0,1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right")
plt.grid(True)
roc_path = os.path.join(OUT_DIR, "roc_comparison.png")
plt.savefig(roc_path)
plt.show()

# Precision-Recall Plot
plt.figure(figsize=(8,6))
for res in to_plot:
    plt.plot(res["recall_curve"], res["precision_curve"], 
             label=f"model {res['model']} (AP={res['ap']:.3f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve Comparison")
plt.legend(loc="lower left")
plt.grid(True)
pr_path = os.path.join(OUT_DIR, "pr_comparison.png")
plt.savefig(pr_path)
plt.show()