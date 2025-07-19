import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pickle
import logging
import sys
import signal  # Ensure this line is present
from contextlib import contextmanager

# === CONFIG ===
BATCH_SIZE = 8
MAX_LENGTH = 64
DATA_PATH = "/home/leloc/Document/USTH/Thesis/Machine_translation-Thai-Viet-/tokenized_output"
OUTPUT_MODEL_PATH = "/home/leloc/Document/USTH/Thesis/Machine_translation-Thai-Viet-/nllb-finetuned"
NUM_EPOCHS = 1
LEARNING_RATE = 1e-4
GRADIENT_ACCUMULATION_STEPS = 2
MODEL_PATH = "/home/leloc/Document/USTH/Thesis/Machine_translation-Thai-Viet-/nllb-200-thai-expanded_600M"
TOKENIZER_PATH = "/home/leloc/Document/USTH/Thesis/Machine_translation-Thai-Viet-/nllb-200-thai-expanded_600M"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dataset class
class TranslationDataset(Dataset):
    def __init__(self, thai_ids, viet_ids):
        assert len(thai_ids) == len(viet_ids), "Mismatched data sizes"
        self.thai_ids, self.viet_ids = thai_ids, viet_ids
        
    def __len__(self):
        return len(self.thai_ids)
    
    def __getitem__(self, idx):
        return {"thai_input_ids": self.thai_ids[idx], "vi_input_ids": self.viet_ids[idx]}

# Collate function to pad sequences
class CollateWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, batch):
        thai_ids = [item["thai_input_ids"][:MAX_LENGTH] for item in batch]
        viet_ids = [item["vi_input_ids"][:MAX_LENGTH] for item in batch]
        input_ids = pad_sequence(thai_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(viet_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# Load data function
def load_data(data_path, tokenizer):
    all_thai, all_viet = [], []
    for file in sorted([f for f in os.listdir(data_path) if f.endswith('.pkl')]):
        with open(os.path.join(data_path, file), "rb") as f:
            data = pickle.load(f)
            assert "thai_ids" in data and "viet_ids" in data
            all_thai.append(data["thai_ids"])
            all_viet.append(data["viet_ids"])
    return torch.cat(all_thai), torch.cat(all_viet)

# Signal handler for graceful shutdown
def handle_signal(signum, frame):
    logger.warning(f"Received signal {signum}, shutting down...")
    sys.exit(1)

# Setup device for each rank
def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

# Load model with device_map='auto'
def load_model():
    # Check if model path exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model path {MODEL_PATH} does not exist")
    
    # Load model with device_map='auto'
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=torch.float16)

    # Enable gradient checkpointing if needed for large models
    model.gradient_checkpointing_enable()

    return model

# Main training function
def train():
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Setup device
    device = setup_device()

    # Load the model with device_map
    model = load_model()
    model.to(device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    # Load dataset and preprocess
    thai_tensor, viet_tensor = load_data(DATA_PATH, tokenizer)

    # Split data into train and validation sets (e.g., 90% training, 10% validation)
    split_size = int(0.9 * len(thai_tensor))
    train_thai, val_thai = thai_tensor[:split_size], thai_tensor[split_size:]
    train_viet, val_viet = viet_tensor[:split_size], viet_tensor[split_size:]

    # Create dataset objects
    train_dataset = TranslationDataset(train_thai, train_viet)
    val_dataset = TranslationDataset(val_thai, val_viet)

    # Create DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=CollateWrapper(tokenizer))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=CollateWrapper(tokenizer))

    # Setup optimizer and scaler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device), labels=batch["labels"].to(device))
                loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS

            scaler.scale(loss).backward()
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item()

        logger.info(f"Epoch {epoch+1} | Avg Loss: {total_loss/len(train_loader):.4f}")

    # Save model and tokenizer
    os.makedirs(OUTPUT_MODEL_PATH, exist_ok=True)
    model.save_pretrained(OUTPUT_MODEL_PATH)
    tokenizer.save_pretrained(OUTPUT_MODEL_PATH)
    logger.info(f"Model saved to: {OUTPUT_MODEL_PATH}")

if __name__ == "__main__":
    train()
