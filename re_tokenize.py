import duckdb
from transformers import AutoTokenizer
import torch
import pickle
from tqdm import tqdm

# Load the extended tokenizer
tokenizer = AutoTokenizer.from_pretrained("~/nllb-1.3B-thai-extended-tokenizer")

# Connect to DuckDB
db_path = "/home/leloc/Document/USTH/Thesis/translation.db"
con = duckdb.connect(db_path)

# Fetch all rows from translations table
df = con.execute("""
    SELECT rowid AS id, thai, viet
    FROM translations
""").fetchdf()

# Handle potential NULL or non-string values
df["thai"] = df["thai"].fillna("").astype(str)
df["viet"] = df["viet"].fillna("").astype(str)

# Prepare Thai inputs with prepended language code
thai_inputs = ["tha_Thai " + text for text in df["thai"].tolist()]

# Re-tokenize Thai text with progress
print("🔁 Tokenizing Thai...")
thai_tokens = tokenizer(
    thai_inputs,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=256,
    return_attention_mask=True,
    add_special_tokens=True
)

# Re-tokenize Vietnamese text with progress
print("🔁 Tokenizing Vietnamese...")
viet_tokens = tokenizer(
    df["viet"].tolist(),
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=256,
    return_attention_mask=True,
    add_special_tokens=True
)

# Serialize token tensors to BLOB-compatible format
print("📦 Serializing tokens...")
thai_input_ids_blobs = [pickle.dumps(ids.numpy()) for ids in tqdm(thai_tokens["input_ids"], desc="Thai input_ids")]
thai_attention_mask_blobs = [pickle.dumps(mask.numpy()) for mask in tqdm(thai_tokens["attention_mask"], desc="Thai attention_mask")]
vi_input_ids_blobs = [pickle.dumps(ids.numpy()) for ids in tqdm(viet_tokens["input_ids"], desc="Vi input_ids")]
vi_attention_mask_blobs = [pickle.dumps(mask.numpy()) for mask in tqdm(viet_tokens["attention_mask"], desc="Vi attention_mask")]

# Update the database with progress bar
print("📝 Updating database...")
for i, row in tqdm(df.iterrows(), total=len(df), desc="🔄 Updating DB"):
    con.execute("""
        UPDATE translations
        SET thai_input_ids = ?,
            thai_attention_mask = ?,
            vi_input_ids = ?,
            vi_attention_mask = ?
        WHERE rowid = ?
    """, (
        thai_input_ids_blobs[i],
        thai_attention_mask_blobs[i],
        vi_input_ids_blobs[i],
        vi_attention_mask_blobs[i],
        row["id"]
    ))

# Verify updates by fetching and decoding a sample
sample = con.execute("""
    SELECT thai, viet, thai_input_ids, thai_attention_mask, vi_input_ids, vi_attention_mask
    FROM translations
    LIMIT 5
""").fetchdf()

print("\n🔍 Verifying updated tokenizations:")
for i in range(len(sample)):
    print(f"\n🟢 Sample {i + 1}")
    print("🇹🇭 THAI:")
    print("Original:", sample.iloc[i]["thai"])
    thai_ids = pickle.loads(sample.iloc[i]["thai_input_ids"]).tolist()
    print("Token IDs:", thai_ids)
    print("Decoded:", tokenizer.decode(thai_ids, skip_special_tokens=True))

    print("🇻🇳 VIET:")
    print("Original:", sample.iloc[i]["viet"])
    vi_ids = pickle.loads(sample.iloc[i]["vi_input_ids"]).tolist()
    print("Token IDs:", vi_ids)
    print("Decoded:", tokenizer.decode(vi_ids, skip_special_tokens=True))

# Verify new Thai characters
missing_thai_chars = ['ฌ', 'ฦ', 'ำ', '฻', '฼', '฽', '฾', '฿', 'ๅ', '๎', '๏', '๐', '๑', '๒', '๓', '๔', '๕', '๖', '๗', '๘', '๙', '๚', '๛']
print("\n🔍 Verifying new Thai characters:")
for char in missing_thai_chars:
    token_id = tokenizer.convert_tokens_to_ids(char)
    if token_id == tokenizer.unk_token_id:
        print(f"Warning: Character {char} is still <unk>")
    else:
        print(f"Character {char} has token ID {token_id}")

# Close the connection
con.close()
print("✅ Re-tokenization and database update completed.")
