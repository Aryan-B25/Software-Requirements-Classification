# hybrid_roberta_deberta.py

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

LABELS = ['F', 'PE', 'LF', 'US', 'A', 'SE', 'FT', 'SC', 'PO', 'O', 'MN', 'L']
OUTPUT_DIR = "./outputs/hybrid_roberta_deberta"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load predictions
roberta_probs = pd.read_csv("./outputs/roberta/roberta_soft_outputs.csv")
deberta_probs = pd.read_csv("./outputs/deberta/deberta_soft_outputs.csv")

# Ensure sample order consistency
assert all(roberta_probs["text"] == deberta_probs["text"])

# Extract softmax values
roberta_soft = roberta_probs[[col for col in roberta_probs.columns if col.startswith("prob_")]].values
deberta_soft = deberta_probs[[col for col in deberta_probs.columns if col.startswith("prob_")]].values

# Average the probabilities
hybrid_soft = (roberta_soft + deberta_soft) / 2.0
hybrid_preds = np.argmax(hybrid_soft, axis=1)

# Ground truth
true_labels = roberta_probs["true_label"].apply(lambda x: LABELS.index(x)).values
predicted_labels = [LABELS[i] for i in hybrid_preds]
true_labels_named = [LABELS[i] for i in true_labels]

# Metrics
acc = accuracy_score(true_labels, hybrid_preds)
report = classification_report(true_labels, hybrid_preds, target_names=LABELS, zero_division=0)
conf_matrix = confusion_matrix(true_labels, hybrid_preds)

print("Hybrid RoBERTa + DeBERTa Accuracy:", acc)
print(report)

# Save results
pd.DataFrame({
    "text": roberta_probs["text"],
    "true": true_labels_named,
    "pred": predicted_labels
}).to_csv(os.path.join(OUTPUT_DIR, "hybrid_roberta_deberta_predictions.csv"), index=False)

with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n\n")
    f.write(report)

# Plot confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
plt.title("Hybrid RoBERTa + DeBERTa Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()
