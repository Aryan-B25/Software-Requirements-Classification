import os
import json
import string
import pandas as pd
import torch
import re
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
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
OUTPUT_DIR = "./outputs/phase1_multilabel_roberta"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 5
LR = 1e-5

# === Load and clean dataset ===
df = pd.read_csv(DATA_PATH).dropna(subset=["SoftwareRequirement", "FunctionalityLabels"])

# === Label Parsing (custom for your format like: [Security, Usability]) ===
def parse_label_list(s):
    if not isinstance(s, str): return []
    return re.findall(r'\b[A-Za-z]+\b', s)

df["labels"] = df["FunctionalityLabels"].apply(parse_label_list)
df = df[df["labels"].apply(lambda x: len(x) > 0)]  # Filter out empty label rows
print("Remaining samples after filtering:", len(df))

# === Encode multi-labels ===
mlb = MultiLabelBinarizer()
binary_labels = mlb.fit_transform(df["labels"])
label_names = mlb.classes_

# === Tokenizer ===
tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)

# === Dataset Class ===
class MultiLabelDataset(Dataset):
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
            "labels": torch.tensor(self.labels[idx], dtype=torch.float)
        }

# === Text Preprocessing ===
def preprocess(text):
    text = text.lower()
    text = "".join(c for c in text if c not in string.punctuation)
    tokens = word_tokenize(text)
    return " ".join(t for t in tokens if t not in stopwords.words("english"))

df["text_clean"] = df["SoftwareRequirement"].apply(preprocess)

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(df["text_clean"], binary_labels, test_size=0.2, random_state=42)
train_ds = MultiLabelDataset(X_train.tolist(), y_train.tolist(), tokenizer, MAX_LEN)
test_ds = MultiLabelDataset(X_test.tolist(), y_test.tolist(), tokenizer, MAX_LEN)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

# === Model Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RobertaForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_names),
    problem_type="multi_label_classification"
).to(device)

optimizer = Adam(model.parameters(), lr=LR)
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=EPOCHS * len(train_dl))

# === Training ===
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in train_dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        if torch.isnan(loss):
            print("⚠️ NaN loss detected — skipping this batch")
            continue
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss / len(train_dl):.4f}")

# === Evaluation ===
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        probs = torch.sigmoid(outputs.logits)
        preds = (probs > 0.5).int()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch["labels"].cpu().numpy())

# === Save Metrics ===
report = classification_report(all_labels, all_preds, target_names=label_names, zero_division=0)
with open(os.path.join(OUTPUT_DIR, "multilabel_classification_report.txt"), "w") as f:
    f.write(report)

with open(os.path.join(OUTPUT_DIR, "multilabel_labels.json"), "w") as f:
    json.dump(label_names.tolist(), f, indent=2)

# === Save Per-Instance Predictions
def decode_labels(binary_vector):
    return [label_names[i] for i, val in enumerate(binary_vector) if val == 1]

prediction_rows = []
for i, text in enumerate(X_test.tolist()):
    true_lbls = decode_labels(all_labels[i])
    pred_lbls = decode_labels(all_preds[i])
    prediction_rows.append({
        "text": text,
        "true_labels": ", ".join(true_lbls),
        "predicted_labels": ", ".join(pred_lbls)
    })

pd.DataFrame(prediction_rows).to_csv(
    os.path.join(OUTPUT_DIR, "multilabel_predictions.csv"), index=False
)

# === F1 Score Bar Chart
report_dict = classification_report(all_labels, all_preds, target_names=label_names, output_dict=True, zero_division=0)
f1_scores = {label: report_dict[label]["f1-score"] for label in label_names}

plt.figure(figsize=(10, 6))
plt.bar(f1_scores.keys(), f1_scores.values(), color="skyblue")
plt.xticks(rotation=45, ha="right")
plt.ylabel("F1-score")
plt.title("Label-wise F1 Scores (Multi-label RoBERTa)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "labelwise_f1_scores.png"))
plt.close()

print("Training and evaluation complete.")
