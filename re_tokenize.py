import duckdb
from transformers import AutoTokenizer
import torch
import pickle
from tqdm import tqdm

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("/sdd/lv01/leloc/translation_machine/nllb-1.3B-thai-extended-tokenizer")

# Connect to DuckDB
db_path = "/sdd/lv01/leloc/translation_machine/translation.db"
con = duckdb.connect(db_path)

# Fetch data
df = con.execute("""
    SELECT rowid AS id, thai, viet
    FROM translations
""").fetchdf()

# Clean text
df["thai"] = df["thai"].fillna("").astype(str)
df["viet"] = df["viet"].fillna("").astype(str)

# Tokenize Thai with progress
thai_inputs = ["tha_Thai " + text for text in df["thai"].tolist()]
thai_input_ids_blobs = []
thai_attention_mask_blobs = []

print("🔠 Tokenizing Thai text...")
for text in tqdm(thai_inputs, desc="🇹🇭 Thai", unit="sample"):
    tokens = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=256, return_attention_mask=True)
    thai_input_ids_blobs.append(pickle.dumps(tokens["input_ids"][0].numpy()))
    thai_attention_mask_blobs.append(pickle.dumps(tokens["attention_mask"][0].numpy()))

# Tokenize Vietnamese with progress
vi_input_ids_blobs = []
vi_attention_mask_blobs = []

print("🔠 Tokenizing Vietnamese text...")
for text in tqdm(df["viet"].tolist(), desc="🇻🇳 Vietnamese", unit="sample"):
    tokens = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=256, return_attention_mask=True)
    vi_input_ids_blobs.append(pickle.dumps(tokens["input_ids"][0].numpy()))
    vi_attention_mask_blobs.append(pickle.dumps(tokens["attention_mask"][0].numpy()))

# Prepare all data
records = []
for i, row in enumerate(df.itertuples()):
    records.append((
        thai_input_ids_blobs[i],
        thai_attention_mask_blobs[i],
        vi_input_ids_blobs[i],
        vi_attention_mask_blobs[i],
        row.id
    ))

# Batch update
batch_size = 10000
for i in tqdm(range(0, len(records), batch_size), desc="🔄 Updating DuckDB", unit="batch"):
    batch = records[i:i+batch_size]
    con.executemany("""
        UPDATE translations
        SET thai_input_ids = ?,
            thai_attention_mask = ?,
            vi_input_ids = ?,
            vi_attention_mask = ?
        WHERE rowid = ?
    """, batch)

con.close()
print("✅ Done.")
