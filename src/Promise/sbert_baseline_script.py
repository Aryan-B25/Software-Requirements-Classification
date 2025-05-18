import os
os.environ["USE_TF"] = "NO"

import json
import string
import pandas as pd
import nltk
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: avoid TF/Keras issues


# --- Setup ---
nltk.download("stopwords")
nltk.download("punkt")

MODEL_NAME = "paraphrase-mpnet-base-v2"
DATASET_PATH = "./datasets/promise.csv"
OUTPUT_DIR = "./sbert_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Label Descriptions ---
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
label_list = list(label_descriptions.keys())

# --- Load Model ---
print(f"Loading SBERT model: {MODEL_NAME}")
sbert = SentenceTransformer(MODEL_NAME)

# --- Preprocess ---
def preprocess(text):
    text = text.lower()
    text = "".join(c for c in text if c not in string.punctuation)
    stopwords = set(nltk.corpus.stopwords.words("english"))
    tokens = nltk.word_tokenize(text)
    return " ".join(t for t in tokens if t not in stopwords)

# --- Load Dataset ---
df = pd.read_csv(DATASET_PATH)
df = df.dropna(subset=["INPUT", "TYPE"])
df = df[df["TYPE"].isin(label_list)].copy()
df["PROCESSED_INPUT"] = df["INPUT"].apply(preprocess)

# --- Encode Inputs and Labels ---
print("Encoding requirements...")
req_embeddings = sbert.encode(df["PROCESSED_INPUT"].tolist(), convert_to_tensor=True)

print("Encoding label descriptions...")
label_embeddings = sbert.encode([label_descriptions[l] for l in label_list], convert_to_tensor=True)

# --- Predict via Cosine Similarity ---
predictions = []
for emb in req_embeddings:
    sims = util.pytorch_cos_sim(emb, label_embeddings)[0]
    predictions.append(label_list[torch.argmax(sims).item()])

df["PREDICTED_TYPE_SBERT"] = predictions

# --- Evaluation ---
true_labels = df["TYPE"].tolist()
pred_labels = df["PREDICTED_TYPE_SBERT"].tolist()
accuracy = accuracy_score(true_labels, pred_labels)
report = classification_report(true_labels, pred_labels, labels=label_list, zero_division=0)
report_dict = classification_report(true_labels, pred_labels, labels=label_list, output_dict=True, zero_division=0)
precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(true_labels, pred_labels, average="micro", zero_division=0)
precision_macro = report_dict["macro avg"]["precision"]
recall_macro = report_dict["macro avg"]["recall"]
f1_macro = report_dict["macro avg"]["f1-score"]

# --- Save Outputs ---
df[["PROCESSED_INPUT", "TYPE", "PREDICTED_TYPE_SBERT"]].rename(columns={"TYPE": "TRUE_TYPE"}).to_csv(
    os.path.join(OUTPUT_DIR, "sbert_predictions.csv"), index=False
)

with open(os.path.join(OUTPUT_DIR, "sbert_classification_report.txt"), "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write(report)

with open(os.path.join(OUTPUT_DIR, "sbert_metrics_summary.json"), "w") as f:
    json.dump({
        "accuracy": accuracy,
        "micro_precision": precision_micro,
        "micro_recall": recall_micro,
        "micro_f1": f1_micro,
        "macro_precision": precision_macro,
        "macro_recall": recall_macro,
        "macro_f1": f1_macro,
        "report": report_dict
    }, f, indent=4)

# --- Confusion Matrix ---
conf_matrix = confusion_matrix(true_labels, pred_labels, labels=label_list)
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_list, yticklabels=label_list)
plt.title("SBERT Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(os.path.join(OUTPUT_DIR, "sbert_confusion_matrix.png"))
plt.close()

print(f"Done. Accuracy: {accuracy:.4f}. Results saved to `{OUTPUT_DIR}`.")
