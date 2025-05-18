import os
import json
import string
import pandas as pd
import nltk
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW, get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download("stopwords")
nltk.download("punkt")

# === Config ===
MODEL_NAME = "roberta-base"
DATA_PATH = "./datasets/promise.csv"
OUTPUT_DIR = "./roberta_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABELS = ['F', 'PE', 'LF', 'US', 'A', 'SE', 'FT', 'SC', 'PO', 'O', 'MN', 'L']
label2id = {label: idx for idx, label in enumerate(LABELS)}
id2label = {idx: label for label, idx in label2id.items()}

MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 5
LR = 2e-5

# === Preprocessing ===
def preprocess(text):
    text = text.lower()
    text = "".join(c for c in text if c not in string.punctuation)
    stopwords_set = set(nltk.corpus.stopwords.words("english"))
    tokens = nltk.word_tokenize(text)
    return " ".join(t for t in tokens if t not in stopwords_set)

# === Dataset Class ===
class NFRDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.texts[idx],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# === Load and preprocess dataset ===
df = pd.read_csv(DATA_PATH).dropna(subset=["INPUT", "TYPE"])
df = df[df["TYPE"].isin(LABELS)].copy()
df["PROCESSED_INPUT"] = df["INPUT"].apply(preprocess)
df["label_id"] = df["TYPE"].map(label2id)

# === Tokenizer & Splits ===
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
X_train, X_test, y_train, y_test = train_test_split(df["PROCESSED_INPUT"], df["label_id"], test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

train_ds = NFRDataset(X_train.tolist(), y_train.tolist(), tokenizer, MAX_LEN)
val_ds = NFRDataset(X_val.tolist(), y_val.tolist(), tokenizer, MAX_LEN)
test_ds = NFRDataset(X_test.tolist(), y_test.tolist(), tokenizer, MAX_LEN)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

# === Model Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(LABELS)).to(device)
optimizer = AdamW(model.parameters(), lr=LR)
num_training_steps = len(train_dl) * EPOCHS
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# === Training ===
def train_model():
    model.train()
    best_val_loss = float("inf")
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(train_dl):.4f}")
        # (Optional) Save model every epoch or best epoch if val loss tracked

# === Evaluation ===
def evaluate(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
    return all_labels, all_preds

# === Run training and evaluation ===
train_model()
true, pred = evaluate(model, test_dl)

# === Metrics & Reports ===
acc = accuracy_score(true, pred)
report = classification_report(true, pred, labels=list(range(len(LABELS))), target_names=LABELS, zero_division=0)
report_dict = classification_report(true, pred, labels=list(range(len(LABELS))), target_names=LABELS, output_dict=True, zero_division=0)
conf_matrix = confusion_matrix(true, pred)

# Save metrics
with open(os.path.join(OUTPUT_DIR, "roberta_classification_report.txt"), "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n\n")
    f.write(report)

with open(os.path.join(OUTPUT_DIR, "roberta_metrics_summary.json"), "w") as f:
    json.dump(report_dict, f, indent=4)

plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
plt.title("RoBERTa Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(os.path.join(OUTPUT_DIR, "roberta_confusion_matrix.png"))
plt.close()

# Save fine-tuned model and tokenizer
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training complete. Model and tokenizer saved to:", OUTPUT_DIR)
