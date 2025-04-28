import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import duckdb
import pickle
import time

# === CONFIG ===
model_name = "facebook/nllb-200-distilled-600M"
source_lang = "tha"
target_lang = "vie"
batch_size = 1
max_length = 256
num_epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LOAD TOKENIZER AND MODEL ===
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

# === LOAD DATA FROM DUCKDB ===
con = duckdb.connect("/home/leloc/Document/USTH/Thesis/Data/translation.db")
df = con.execute("SELECT thai_input_ids, vi_input_ids FROM translations LIMIT 10").fetchdf()

# === CUSTOM DATASET ===
class TranslationDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        thai_ids = pickle.loads(self.df.iloc[idx]['thai_input_ids'])
        vie_ids = pickle.loads(self.df.iloc[idx]['vi_input_ids'])
        
        # decode to text
        thai_text = tokenizer.decode(thai_ids, skip_special_tokens=True)
        vie_text = tokenizer.decode(vie_ids, skip_special_tokens=True)

        return {
            "src_text": thai_text,
            "tgt_text": vie_text
        }

# === COLLATOR ===
def collate_fn(batch):
    src_texts = [item['src_text'] for item in batch]
    tgt_texts = [item['tgt_text'] for item in batch]

    model_inputs = tokenizer(
        src_texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    labels = tokenizer(
        tgt_texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).input_ids

    model_inputs["labels"] = labels
    return model_inputs

# === DATALOADER ===
dataset = TranslationDataset(df)
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

# === TRAINING PREP ===
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
scaler = torch.cuda.amp.GradScaler()  # Mixed precision

model.train()

# === TRAINING LOOP ===
for epoch in range(num_epochs):
    total_loss = 0
    start = time.time()

    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.cuda.amp.autocast():  # mixed precision
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"\n✅ Epoch {epoch+1} completed | Loss: {avg_loss:.4f} | Time: {time.time() - start:.2f}s")

# === SAVE MODEL ===
save_path = "./saved_nllb_model"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print("🎉 Training complete and model saved at:", save_path)
