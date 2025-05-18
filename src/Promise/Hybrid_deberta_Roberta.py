import os
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import nltk

nltk.download("punkt")
nltk.download("stopwords")

# === Configuration ===
DATA_PATH = "./datasets/promise.csv"
OUTPUT_PATH = "./hybrid_roberta_deberta_outputs"
os.makedirs(OUTPUT_PATH, exist_ok=True)

LABELS = ['F', 'PE', 'LF', 'US', 'A', 'SE', 'FT', 'SC', 'PO', 'O', 'MN', 'L']
label2id = {label: idx for idx, label in enumerate(LABELS)}
id2label = {idx: label for label, idx in label2id.items()}
NUM_LABELS = len(LABELS)
MAX_LEN = 256

# === Preprocessing ===
def preprocess(text):
    text = text.lower()
    text = "".join(c for c in text if c not in string.punctuation)
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text)
    return " ".join(t for t in tokens if t not in stop_words)

# === Load and preprocess dataset ===
df = pd.read_csv(DATA_PATH).dropna(subset=["INPUT", "TYPE"])
df = df[df["TYPE"].isin(LABELS)].copy()
df["PROCESSED_INPUT"] = df["INPUT"].apply(preprocess)

# === Load fine-tuned models ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

roberta_model_path = "./roberta_outputs"
deberta_model_path = "./deberta_outputs"

roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_model_path)
deberta_tokenizer = AutoTokenizer.from_pretrained(deberta_model_path)

roberta_model = AutoModelForSequenceClassification.from_pretrained(roberta_model_path).to(device).eval()
deberta_model = AutoModelForSequenceClassification.from_pretrained(deberta_model_path).to(device).eval()

# === Prediction Function (Soft Voting) ===
def predict(text):
    with torch.no_grad():
        # RoBERTa
        r_inputs = roberta_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(device)
        r_logits = roberta_model(**r_inputs).logits
        r_probs = F.softmax(r_logits, dim=1)

        # DeBERTa
        d_inputs = deberta_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(device)
        d_logits = deberta_model(**d_inputs).logits
        d_probs = F.softmax(d_logits, dim=1)

        # Soft-voting (equal weights)
        avg_probs = (r_probs + d_probs) / 2
        pred_label = torch.argmax(avg_probs, dim=1).item()
        return id2label[pred_label]

# === Run predictions ===
df["PREDICTED_HYBRID_LABEL"] = df["PROCESSED_INPUT"].apply(predict)

# === Evaluation ===
true = df["TYPE"].tolist()
pred = df["PREDICTED_HYBRID_LABEL"].tolist()

acc = accuracy_score(true, pred)
report = classification_report(true, pred, labels=LABELS, zero_division=0)
report_dict = classification_report(true, pred, labels=LABELS, output_dict=True, zero_division=0)
conf_matrix = confusion_matrix(true, pred, labels=LABELS)

# Save predictions
df.to_csv(os.path.join(OUTPUT_PATH, "hybrid_predictions.csv"), index=False)

# Save metrics
with open(os.path.join(OUTPUT_PATH, "hybrid_classification_report.txt"), "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n\n")
    f.write(report)

with open(os.path.join(OUTPUT_PATH, "hybrid_metrics_summary.json"), "w") as f:
    json.dump(report_dict, f, indent=4)

# Save confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
plt.title("Hybrid (RoBERTa + DeBERTa) Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(os.path.join(OUTPUT_PATH, "hybrid_confusion_matrix.png"))
plt.close()

print(f"Hybrid model complete. Accuracy: {acc:.4f}")
