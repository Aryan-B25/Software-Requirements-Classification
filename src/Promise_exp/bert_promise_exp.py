
import os
import json
import string
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification, AdamW, get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("stopwords")

# === Config ===
MODEL_NAME = "bert-base-uncased"
DATA_PATH = "./datasets/PROMISE_exp.csv"
OUTPUT_DIR = "./outputs/bert"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABELS = ['F', 'PE', 'LF', 'US', 'A', 'SE', 'FT', 'SC', 'PO', 'O', 'MN', 'L']
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}
NUM_LABELS = len(LABELS)
MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 5
LR = 2e-5

# === Dataset Class ===
class NFRDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer.encode_plus(
            self.texts[idx],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# === Preprocessing ===
def preprocess(text):
    text = text.lower()
    text = "".join(c for c in text if c not in string.punctuation)
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text)
    return " ".join(t for t in tokens if t not in stop_words)

# === Load Dataset ===
df = pd.read_csv(DATA_PATH)
df = df[df["_class_"].isin(LABELS)].copy()
df["text_clean"] = df["RequirementText"].apply(preprocess)
df["label_id"] = df["_class_"].map(label2id)

# === Tokenizer and DataLoader ===
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
X_train, X_test, y_train, y_test = train_test_split(df["text_clean"], df["label_id"], test_size=0.2, random_state=42)

train_ds = NFRDataset(X_train.tolist(), y_train.tolist(), tokenizer, MAX_LEN)
test_ds = NFRDataset(X_test.tolist(), y_test.tolist(), tokenizer, MAX_LEN)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

# === Model, Optimizer, Scheduler ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS).to(device)
optimizer = AdamW(model.parameters(), lr=LR)
lr_scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=len(train_dl)*EPOCHS)

# === Training ===
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in train_dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_dl):.4f}")

# === Evaluation ===
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch["labels"].cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
report = classification_report(all_labels, all_preds, target_names=LABELS, zero_division=0)
report_dict = classification_report(all_labels, all_preds, target_names=LABELS, output_dict=True, zero_division=0)
conf_matrix = confusion_matrix(all_labels, all_preds)

# === Save Outputs ===
pd.DataFrame({
    "text": X_test.tolist(),
    "true": [id2label[i] for i in all_labels],
    "pred": [id2label[i] for i in all_preds]
}).to_csv(os.path.join(OUTPUT_DIR, "bert_predictions.csv"), index=False)

with open(os.path.join(OUTPUT_DIR, "bert_classification_report.txt"), "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n\n")
    f.write(report)

with open(os.path.join(OUTPUT_DIR, "bert_metrics_summary.json"), "w") as f:
    json.dump(report_dict, f, indent=4)

plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
plt.title("BERT Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(os.path.join(OUTPUT_DIR, "bert_confusion_matrix.png"))
plt.close()

print(f"BERT model complete. Accuracy: {acc:.4f}")

import torch.nn.functional as F

# === Save softmax scores ===
softmax_probs = []
for batch in test_dl:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        logits = model(**batch).logits
        probs = F.softmax(logits, dim=1)
        softmax_probs.extend(probs.cpu().tolist())

soft_output_df = pd.DataFrame(softmax_probs, columns=[f"prob_{lbl}" for lbl in LABELS])
soft_output_df.insert(0, "text", X_test.tolist())
soft_output_df.insert(1, "true_label", [id2label[i] for i in all_labels])
soft_output_df.insert(2, "predicted_label", [id2label[i] for i in all_preds])

soft_output_df.to_csv(os.path.join(OUTPUT_DIR, "bert_soft_outputs.csv"), index=False)
