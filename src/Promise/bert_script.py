import os
import json
import string
import pandas as pd
import nltk
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

# --- Config ---
MODEL_NAME = "bert-base-uncased"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "datasets", "promise.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "bert_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_EPOCHS = 5
BATCH_SIZE = 8
MAX_LEN = 256
LEARNING_RATE = 2e-5
RANDOM_STATE = 42
TEST_SIZE = 0.2

# --- NLTK Setup ---
nltk.download("stopwords")
nltk.download("punkt")

# --- Label Setup ---
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

# --- Load Dataset ---
df = pd.read_csv(DATASET_PATH)
df = df.dropna(subset=["INPUT", "TYPE"])
df = df[df["TYPE"].isin(label_list)].copy()

def preprocess(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    stopwords = set(nltk.corpus.stopwords.words("english"))
    tokens = nltk.word_tokenize(text)
    return " ".join([t for t in tokens if t not in stopwords])

df["PROCESSED_INPUT"] = df["INPUT"].apply(preprocess)
df["label_id"] = df["TYPE"].map(label2id)

# --- Dataset Class ---
class NFRDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts, self.labels = texts, labels
        self.tokenizer, self.max_len = tokenizer, max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        enc = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.max_len,
            return_token_type_ids=False, padding="max_length", truncation=True,
            return_attention_mask=True, return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# --- Splits ---
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["PROCESSED_INPUT"], df["label_id"], test_size=TEST_SIZE, random_state=RANDOM_STATE
)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts, train_labels, test_size=0.1, random_state=RANDOM_STATE
)

# --- Tokenizer & Loaders ---
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
train_loader = DataLoader(NFRDataset(train_texts.tolist(), train_labels.tolist(), tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(NFRDataset(val_texts.tolist(), val_labels.tolist(), tokenizer, MAX_LEN), batch_size=BATCH_SIZE)
test_loader = DataLoader(NFRDataset(test_texts.tolist(), test_labels.tolist(), tokenizer, MAX_LEN), batch_size=BATCH_SIZE)

# --- Model Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(label_list)).to(device)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_linear_schedule_with_warmup(optimizer, 0, len(train_loader) * NUM_EPOCHS)

# --- Training ---
def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss, correct = 0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
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
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss, logits = outputs.loss, outputs.logits
            _, preds = torch.max(logits, dim=1)
            correct += torch.sum(preds == labels)
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return correct.double() / len(loader.dataset), total_loss / len(loader), all_preds, all_labels

best_val_acc = 0
for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    train_acc, train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
    val_acc, val_loss, _, _ = eval_model(model, val_loader, device)
    print(f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    if val_acc > best_val_acc:
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_bert_model.bin"))
        best_val_acc = val_acc

# --- Final Evaluation ---
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_bert_model.bin")))
test_acc, test_loss, test_preds, test_labels = eval_model(model, test_loader, device)
pred_labels = [id2label[p] for p in test_preds]
true_labels = [id2label[t] for t in test_labels]

print(f"\nTest Accuracy: {test_acc:.4f}")
report = classification_report(true_labels, pred_labels, labels=label_list, zero_division=0)
print(report)

# Save Outputs
report_dict = classification_report(true_labels, pred_labels, labels=label_list, output_dict=True, zero_division=0)
with open(os.path.join(OUTPUT_DIR, "bert_classification_report.txt"), "w") as f:
    f.write(report)
with open(os.path.join(OUTPUT_DIR, "bert_metrics_summary.json"), "w") as f:
    json.dump({
        "accuracy": test_acc.item(),
        "micro_precision": precision_recall_fscore_support(true_labels, pred_labels, average="micro", zero_division=0)[0],
        "micro_recall": precision_recall_fscore_support(true_labels, pred_labels, average="micro", zero_division=0)[1],
        "micro_f1": precision_recall_fscore_support(true_labels, pred_labels, average="micro", zero_division=0)[2],
        "macro_precision": report_dict["macro avg"]["precision"],
        "macro_recall": report_dict["macro avg"]["recall"],
        "macro_f1": report_dict["macro avg"]["f1-score"],
        "report": report_dict
    }, f, indent=4)

conf_matrix = confusion_matrix(true_labels, pred_labels, labels=label_list)
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_list, yticklabels=label_list)
plt.title("BERT Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(os.path.join(OUTPUT_DIR, "bert_confusion_matrix.png"))
plt.close()

decoded_inputs = [tokenizer.decode(test_loader.dataset[i]["input_ids"], skip_special_tokens=True) for i in range(len(test_loader.dataset))]
pd.DataFrame({
    "PROCESSED_INPUT": decoded_inputs,
    "TRUE_TYPE": true_labels,
    "PREDICTED_TYPE_BERT": pred_labels
}).to_csv(os.path.join(OUTPUT_DIR, "bert_predictions.csv"), index=False)

print("BERT training and evaluation complete.")

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("BERT model and tokenizer saved to:", OUTPUT_DIR)
