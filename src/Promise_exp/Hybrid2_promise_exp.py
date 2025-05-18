# hybrid_bert_xlnet.py

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

LABELS = ['F', 'PE', 'LF', 'US', 'A', 'SE', 'FT', 'SC', 'PO', 'O', 'MN', 'L']
OUTPUT_DIR = "./outputs/hybrid_bert_xlnet"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load predictions
bert_probs = pd.read_csv("./outputs/bert/bert_soft_outputs.csv")
xlnet_probs = pd.read_csv("./outputs/xlnet/xlnet_soft_outputs.csv")

# Ensure sample order consistency
assert all(bert_probs["text"] == xlnet_probs["text"])

# Extract softmax values
bert_soft = bert_probs[[col for col in bert_probs.columns if col.startswith("prob_")]].values
xlnet_soft = xlnet_probs[[col for col in xlnet_probs.columns if col.startswith("prob_")]].values

# Average the probabilities
hybrid_soft = (bert_soft + xlnet_soft) / 2.0
hybrid_preds = np.argmax(hybrid_soft, axis=1)

# Ground truth
true_labels = bert_probs["true_label"].apply(lambda x: LABELS.index(x)).values
predicted_labels = [LABELS[i] for i in hybrid_preds]
true_labels_named = [LABELS[i] for i in true_labels]

# Metrics
acc = accuracy_score(true_labels, hybrid_preds)
report = classification_report(true_labels, hybrid_preds, target_names=LABELS, zero_division=0)
conf_matrix = confusion_matrix(true_labels, hybrid_preds)

print("Hybrid BERT + XLNet Accuracy:", acc)
print(report)

# Save results
pd.DataFrame({
    "text": bert_probs["text"],
    "true": true_labels_named,
    "pred": predicted_labels
}).to_csv(os.path.join(OUTPUT_DIR, "hybrid_bert_xlnet_predictions.csv"), index=False)

with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n\n")
    f.write(report)

# Plot confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Greens", xticklabels=LABELS, yticklabels=LABELS)
plt.title("Hybrid BERT + XLNet Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()
