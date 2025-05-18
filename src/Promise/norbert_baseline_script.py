import os
import json
import string
import numpy as np
import pandas as pd
import nltk
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

# --- Setup ---
nltk.download("stopwords")
nltk.download("punkt")

MODEL_NAME = "bert-base-uncased"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "datasets", "promise.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "norbert_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_EPOCHS = 5
BATCH_SIZE = 8
MAX_LEN = 256
LEARNING_RATE = 2e-5
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- Load and Preprocess Labels ---
df = pd.read_csv(DATASET_PATH)
df = df.dropna(subset=["INPUT", "TYPE"])
unique_labels = sorted(df["TYPE"].dropna().unique())
label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}
df = df[df["TYPE"].isin(label2id.keys())]
print(f"Labels found: {list(label2id.keys())}")

# --- Preprocessing ---
def preprocess(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    words = nltk.word_tokenize(text)
    stopwords_set = set(nltk.corpus.stopwords.words("english"))
    return " ".join([w for w in words if w not in stopwords_set])

df["PROCESSED_INPUT"] = df["INPUT"].apply(preprocess)
df["label_id"] = df["TYPE"].map(label2id)

# --- Dataset Class ---
class NFRDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts, self.labels, self.tokenizer, self.max_len = texts, labels, tokenizer, max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.max_len,
            return_token_type_ids=False, padding="max_length", truncation=True,
            return_attention_mask=True, return_tensors="pt"
        )
        return {
            "text": text,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# --- Split Data ---
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["PROCESSED_INPUT"].tolist(), df["label_id"].tolist(),
    test_size=TEST_SIZE, random_state=RANDOM_STATE
)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts, train_labels, test_size=0.1, random_state=RANDOM_STATE
)

# --- Tokenizer & Loaders ---
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
train_loader = DataLoader(NFRDataset(train_texts, train_labels, tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(NFRDataset(val_texts, val_labels, tokenizer, MAX_LEN), batch_size=BATCH_SIZE)
test_loader = DataLoader(NFRDataset(test_texts, test_labels, tokenizer, MAX_LEN), batch_size=BATCH_SIZE)

# --- Model Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(label2id)).to(device)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_linear_schedule_with_warmup(optimizer, 0, len(train_loader) * NUM_EPOCHS)

# --- Train/Eval Helpers ---
def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss, correct = 0, 0
    for batch in loader:
        input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss, logits = outputs.loss, outputs.logits
        _, preds = torch.max(logits, dim=1)
        correct += torch.sum(preds == labels)
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return correct.double() / len(loader.dataset), total_loss / len(loader)

def eval_model(model, loader, device):
    model.eval()
    total_loss, correct = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss, logits = outputs.loss, outputs.logits
            _, preds = torch.max(logits, dim=1)
            correct += torch.sum(preds == labels)
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return correct.double() / len(loader.dataset), total_loss / len(loader), all_preds, all_labels

# --- Training ---
best_val_acc = 0
for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    train_acc, train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
    val_acc, val_loss, _, _ = eval_model(model, val_loader, device)
    print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
    if val_acc > best_val_acc:
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.bin"))
        best_val_acc = val_acc

# --- Final Evaluation ---
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.bin")))
test_acc, test_loss, test_preds, test_labels = eval_model(model, test_loader, device)
pred_labels = [id2label[p] for p in test_preds]
true_labels = [id2label[t] for t in test_labels]

print(f"\nTest Accuracy: {test_acc:.4f} | Loss: {test_loss:.4f}")
report = classification_report(true_labels, pred_labels, labels=unique_labels, zero_division=0)
print(report)

# Save Outputs
report_dict = classification_report(true_labels, pred_labels, labels=unique_labels, output_dict=True, zero_division=0)
with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write(report)
with open(os.path.join(OUTPUT_DIR, "metrics_summary.json"), "w") as f:
    json.dump(report_dict, f, indent=4)

conf_mat = confusion_matrix(true_labels, pred_labels, labels=unique_labels)
plt.figure(figsize=(12, 10))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=unique_labels, yticklabels=unique_labels)
plt.title("NoRBERT Confusion Matrix")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()

decoded_inputs = [tokenizer.decode(test_loader.dataset[i]["input_ids"], skip_special_tokens=True) for i in range(len(test_loader.dataset))]
results_df = pd.DataFrame({"PROCESSED_INPUT": decoded_inputs, "TRUE_TYPE": true_labels, "PREDICTED_TYPE": pred_labels})
results_df.to_csv(os.path.join(OUTPUT_DIR, "norbert_predictions.csv"), index=False)
print("Evaluation complete and outputs saved.")
