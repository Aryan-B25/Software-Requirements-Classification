
import os
os.environ["USE_TF"] = "NO"

import json
import string
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download("punkt")
nltk.download("stopwords")

# === Config ===
MODEL_NAME = "paraphrase-mpnet-base-v2"
DATA_PATH = "./datasets/PROMISE_exp.csv"
OUTPUT_DIR = "./outputs/sbert"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABELS = ['F', 'PE', 'LF', 'US', 'A', 'SE', 'FT', 'SC', 'PO', 'O', 'MN', 'L']
label_descriptions = {
    "F": "Functional Requirement",
    "PE": "Performance Efficiency",
    "LF": "Look and Feel",
    "US": "Usability",
    "A": "Availability",
    "SE": "Security",
    "FT": "Fault Tolerance",
    "SC": "Scalability",
    "PO": "Portability",
    "O": "Operational",
    "MN": "Maintainability",
    "L": "Legal Requirement"
}

# === Preprocessing ===
def preprocess(text):
    text = text.lower()
    text = "".join(c for c in text if c not in string.punctuation)
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text)
    return " ".join(t for t in tokens if t not in stop_words)

# === Load dataset ===
df = pd.read_csv(DATA_PATH)
df = df[df["_class_"].isin(LABELS)].copy()
df["text_clean"] = df["RequirementText"].apply(preprocess)

# === Load SBERT ===
sbert = SentenceTransformer(MODEL_NAME)

# === Encode texts and label descriptions ===
req_embeddings = sbert.encode(df["text_clean"].tolist(), convert_to_tensor=True)
label_embeddings = sbert.encode([label_descriptions[l] for l in LABELS], convert_to_tensor=True)

# === Predict by semantic similarity ===
predictions = []
for emb in req_embeddings:
    sims = util.pytorch_cos_sim(emb, label_embeddings)[0]
    predictions.append(LABELS[torch.argmax(sims).item()])

df["PREDICTED_LABEL"] = predictions

# === Evaluation ===
true_labels = df["_class_"].tolist()
pred_labels = df["PREDICTED_LABEL"].tolist()
accuracy = accuracy_score(true_labels, pred_labels)
report = classification_report(true_labels, pred_labels, labels=LABELS, zero_division=0)
report_dict = classification_report(true_labels, pred_labels, labels=LABELS, output_dict=True, zero_division=0)
conf_matrix = confusion_matrix(true_labels, pred_labels, labels=LABELS)

# === Save Outputs ===
df[["RequirementText", "_class_", "PREDICTED_LABEL"]].rename(columns={"_class_": "TRUE_LABEL"}).to_csv(
    os.path.join(OUTPUT_DIR, "sbert_predictions.csv"), index=False
)

with open(os.path.join(OUTPUT_DIR, "sbert_classification_report.txt"), "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write(report)

with open(os.path.join(OUTPUT_DIR, "sbert_metrics_summary.json"), "w") as f:
    json.dump(report_dict, f, indent=4)

plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
plt.title("SBERT Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(os.path.join(OUTPUT_DIR, "sbert_confusion_matrix.png"))
plt.close()

print(f"SBERT zero-shot model complete. Accuracy: {accuracy:.4f}")
