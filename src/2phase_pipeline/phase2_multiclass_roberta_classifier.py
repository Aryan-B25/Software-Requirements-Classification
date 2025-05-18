import os
import json
import string
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from torch.optim import Adam

# === Setup ===
nltk.download("punkt")
nltk.download("stopwords")

MODEL_NAME = "roberta-base"
DATA_PATH = "./datasets/promise_multilabel_cleaned_625_requirementlabels.csv"
OUTPUT_DIR = "./outputs/phase2_multiclass_roberta"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 5
LR = 1e-5

# === Load dataset and target ===
df = pd.read_csv(DATA_PATH).dropna(subset=["SoftwareRequirement", "Type"])
df = df[df["Type"].isin(["FR", "NFR"])]
df["label_id"] = df["Type"].map({"FR": 0, "NFR": 1})

# === Preprocessing ===
def preprocess(text):
    text = text.lower()
    text = "".join(c for c in text if c not in string.punctuation)
    tokens = word_tokenize(text)
    return " ".join(t for t in tokens if t not in stopwords.words("english"))

df["text_clean"] = df["SoftwareRequirement"].apply(preprocess)

# === Tokenization ===
tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)

class BinaryReqDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer.encode_plus(
            self.texts[idx], max_length=self.max_len, truncation=True,
            padding="max_length", return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# === Dataset Prep ===
X_train, X_test, y_train, y_test = train_test_split(df["text_clean"], df["label_id"], test_size=0.2, random_state=42)
train_ds = BinaryReqDataset(X_train.tolist(), y_train.tolist(), tokenizer, MAX_LEN)
test_ds = BinaryReqDataset(X_test.tolist(), y_test.tolist(), tokenizer, MAX_LEN)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

# === Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RobertaForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
).to(device)

optimizer = Adam(model.parameters(), lr=LR)
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=EPOCHS * len(train_dl))

# === Train ===
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in train_dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss / len(train_dl):.4f}")

# === Evaluate ===
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch["labels"].cpu().numpy())

# === Report ===
target_names = ["FR", "NFR"]
report = classification_report(all_labels, all_preds, target_names=target_names, zero_division=0)
print(report)

with open(os.path.join(OUTPUT_DIR, "multiclass_classification_report.txt"), "w") as f:
    f.write(report)

conf_matrix = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
plt.title("Confusion Matrix (Phase 2)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(ticks=[0, 1], labels=target_names)
plt.yticks(ticks=[0, 1], labels=target_names)
plt.imshow(conf_matrix, cmap="Blues", interpolation="nearest")
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color='black')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_phase2.png"))
plt.close()

print("Phase 2 training and evaluation complete.")
