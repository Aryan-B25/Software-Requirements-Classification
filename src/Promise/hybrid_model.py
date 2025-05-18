import os
os.environ["USE_TF"] = "NO"

import json
import string
import pandas as pd
import nltk
import torch
import numpy as np
from torch.nn.functional import softmax
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertForSequenceClassification
from sentence_transformers import SentenceTransformer, util

# --- Setup ---
nltk.download("stopwords")
nltk.download("punkt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
BERT_MODEL_PATH = "./bert_outputs/best_bert_model.bin"
SBERT_MODEL_NAME = "paraphrase-mpnet-base-v2"
BERT_MODEL_NAME = "bert-base-uncased"
DATASET_PATH = "./datasets/promise.csv"
OUTPUT_PATH = "./hybrid_outputs"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Labels
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
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}

# --- Text Preprocessing ---
def preprocess(text):
    text = text.lower()
    text = "".join(c for c in text if c not in string.punctuation)
    stopwords = set(nltk.corpus.stopwords.words("english"))
    tokens = nltk.word_tokenize(text)
    return " ".join(t for t in tokens if t not in stopwords)

# --- Load Models ---
print("Loading SBERT...")
sbert_model = SentenceTransformer(SBERT_MODEL_NAME)

print("Loading BERT...")
bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
bert_model = BertForSequenceClassification.from_pretrained(BERT_MODEL_NAME, num_labels=len(label_list))
bert_model.load_state_dict(torch.load(BERT_MODEL_PATH, map_location=DEVICE))
bert_model.to(DEVICE)
bert_model.eval()

# --- Precompute SBERT Label Embeddings ---
label_embeddings = sbert_model.encode([label_descriptions[l] for l in label_list], convert_to_tensor=True)

# --- Inference Function ---
def hybrid_predict(text, bert_weight=0.7, sbert_weight=0.3):
    clean_text = preprocess(text)

    # BERT
    enc = bert_tokenizer.encode_plus(clean_text, add_special_tokens=True, max_length=256,
                                     return_token_type_ids=False, padding="max_length",
                                     truncation=True, return_attention_mask=True, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        logits = bert_model(**enc).logits.squeeze()
    bert_probs = softmax(logits, dim=0).cpu()

    # SBERT
    sbert_emb = sbert_model.encode(clean_text, convert_to_tensor=True)
    sbert_sims = util.pytorch_cos_sim(sbert_emb, label_embeddings)[0]
    sbert_probs = softmax(sbert_sims, dim=0).cpu()

    # Fusion
    combined = bert_weight * bert_probs + sbert_weight * sbert_probs
    pred_idx = torch.argmax(combined).item()
    return id2label[pred_idx], combined.tolist()

# --- Load Dataset ---
print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)
df = df.dropna(subset=["INPUT", "TYPE"])
df = df[df["TYPE"].isin(label_list)].copy()
df["PROCESSED_INPUT"] = df["INPUT"].apply(preprocess)

# --- Predict All ---
print("Running hybrid predictions...")
predicted_labels = []
score_vectors = []

for text in df["PROCESSED_INPUT"]:
    label, scores = hybrid_predict(text)
    predicted_labels.append(label)
    score_vectors.append(scores)

df["PREDICTED_TYPE_HYBRID"] = predicted_labels
df["HYBRID_SCORES"] = score_vectors

# --- Save Predictions ---
df[["PROCESSED_INPUT", "TYPE", "PREDICTED_TYPE_HYBRID", "HYBRID_SCORES"]].rename(
    columns={"TYPE": "TRUE_TYPE"}
).to_csv(os.path.join(OUTPUT_PATH, "hybrid_predictions.csv"), index=False)

# --- Optional: Evaluation if Ground Truth is Available ---
true_labels = df["TYPE"].tolist()
pred_labels = df["PREDICTED_TYPE_HYBRID"].tolist()

print("Evaluating hybrid performance...")
report = classification_report(true_labels, pred_labels, labels=label_list, zero_division=0)
report_dict = classification_report(true_labels, pred_labels, labels=label_list, output_dict=True, zero_division=0)

print(report)
with open(os.path.join(OUTPUT_PATH, "hybrid_classification_report.txt"), "w") as f:
    f.write(report)

with open(os.path.join(OUTPUT_PATH, "hybrid_metrics_summary.json"), "w") as f:
    json.dump(report_dict, f, indent=4)

# --- Confusion Matrix ---
conf_matrix = confusion_matrix(true_labels, pred_labels, labels=label_list)
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_list, yticklabels=label_list)
plt.title("Hybrid Model Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(os.path.join(OUTPUT_PATH, "hybrid_confusion_matrix.png"))
plt.close()

print("Hybrid classification complete. All results saved.")
