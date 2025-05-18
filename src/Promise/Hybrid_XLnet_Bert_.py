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

# === Config ===
DATA_PATH = "./datasets/promise.csv"
OUTPUT_PATH = "./hybrid_bert_xlnet_outputs"
os.makedirs(OUTPUT_PATH, exist_ok=True)

LABELS = ['F', 'PE', 'LF', 'US', 'A', 'SE', 'FT', 'SC', 'PO', 'O', 'MN', 'L']
label2id = {label: idx for idx, label in enumerate(LABELS)}
id2label = {idx: label for label, idx in label2id.items()}
MAX_LEN = 256

# === Preprocessing ===
def preprocess(text):
    text = text.lower()
    text = "".join(c for c in text if c not in string.punctuation)
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text)
    return " ".join(t for t in tokens if t not in stop_words)

# === Load dataset ===
df = pd.read_csv(DATA_PATH).dropna(subset=["INPUT", "TYPE"])
df = df[df["TYPE"].isin(LABELS)].copy()
df["PROCESSED_INPUT"] = df["INPUT"].apply(preprocess)

# === Load fine-tuned models ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bert_path = "./bert_outputs"
xlnet_path = "./xlnet_outputs"

bert_tokenizer = AutoTokenizer.from_pretrained(bert_path)
xlnet_tokenizer = AutoTokenizer.from_pretrained(xlnet_path)

bert_model = AutoModelForSequenceClassification.from_pretrained(bert_path).to(device).eval()
xlnet_model = AutoModelForSequenceClassification.from_pretrained(xlnet_path).to(device).eval()

# === Prediction Function (Soft Voting) ===
def predict(text):
    with torch.no_grad():
        b_inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(device)
        b_logits = bert_model(**b_inputs).logits
        b_probs = F.softmax(b_logits, dim=1)

        x_inputs = xlnet_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(device)
        x_logits = xlnet_model(**x_inputs).logits
        x_probs = F.softmax(x_logits, dim=1)

        avg_probs = (b_probs + x_probs) / 2
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

df.to_csv(os.path.join(OUTPUT_PATH, "hybrid_predictions.csv"), index=False)

with open(os.path.join(OUTPUT_PATH, "hybrid_classification_report.txt"), "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n\n")
    f.write(report)

with open(os.path.join(OUTPUT_PATH, "hybrid_metrics_summary.json"), "w") as f:
    json.dump(report_dict, f, indent=4)

plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
plt.title("Hybrid (BERT + XLNet) Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(os.path.join(OUTPUT_PATH, "hybrid_confusion_matrix.png"))
plt.close()

print(f"Hybrid model complete. Accuracy: {acc:.4f}")
