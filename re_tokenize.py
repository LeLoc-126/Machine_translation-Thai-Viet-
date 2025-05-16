import duckdb
from transformers import AutoTokenizer
import torch
import pickle
from tqdm import tqdm

# Cấu hình
db_path = "/sdd/lv01/leloc/translation_machine/translation.db"
tokenizer_path = "/sdd/lv01/leloc/translation_machine/nllb-1.3B-thai-extended-tokenizer"
batch_size = 10000
max_length = 128

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Kết nối DuckDB
con = duckdb.connect(db_path)

# Tải dữ liệu
df = con.execute("""
    SELECT rowid AS id, thai, viet
    FROM translations
""").fetchdf()

# Làm sạch dữ liệu
df["thai"] = df["thai"].fillna("").astype(str)
df["viet"] = df["viet"].fillna("").astype(str)

# Chuẩn bị danh sách kết quả để cập nhật
records = []

# 🔠 Tokenize Thai theo batch
print("🔠 Tokenizing Thai text...")
for i in tqdm(range(0, len(df), batch_size), desc="🇹🇭 Thai", unit="batch"):
    thai_batch = ["tha_Thai " + text for text in df["thai"].iloc[i:i+batch_size]]
    thai_tokens = tokenizer(
        thai_batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True
    )
    thai_input_ids_blobs = [pickle.dumps(x.numpy()) for x in thai_tokens["input_ids"]]
    thai_attention_mask_blobs = [pickle.dumps(x.numpy()) for x in thai_tokens["attention_mask"]]

    # Lưu tạm
    for j, idx in enumerate(range(i, min(i + batch_size, len(df)))):
        records.append({
            "id": df.iloc[idx]["id"],
            "thai_input_ids": thai_input_ids_blobs[j],
            "thai_attention_mask": thai_attention_mask_blobs[j]
        })

# 🔠 Tokenize Việt theo batch và thêm vào records
print("🔠 Tokenizing Vietnamese text...")
for i in tqdm(range(0, len(df), batch_size), desc="🇻🇳 Viet", unit="batch"):
    viet_batch = df["viet"].iloc[i:i+batch_size].tolist()
    viet_tokens = tokenizer(
        viet_batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True
    )
    vi_input_ids_blobs = [pickle.dumps(x.numpy()) for x in viet_tokens["input_ids"]]
    vi_attention_mask_blobs = [pickle.dumps(x.numpy()) for x in viet_tokens["attention_mask"]]

    # Cập nhật thêm vào records
    for j, idx in enumerate(range(i, min(i + batch_size, len(df)))):
        records[idx]["vi_input_ids"] = vi_input_ids_blobs[j]
        records[idx]["vi_attention_mask"] = vi_attention_mask_blobs[j]

# 🔄 Cập nhật DuckDB theo batch
print("🔄 Updating DuckDB...")
update_batches = [
    (
        r["thai_input_ids"],
        r["thai_attention_mask"],
        r["vi_input_ids"],
        r["vi_attention_mask"],
        r["id"]
    )
    for r in records
]

for i in tqdm(range(0, len(update_batches), batch_size), desc="💾 DB Update", unit="batch"):
    con.executemany("""
        UPDATE translations
        SET thai_input_ids = ?,
            thai_attention_mask = ?,
            vi_input_ids = ?,
            vi_attention_mask = ?
        WHERE rowid = ?
    """, update_batches[i:i+batch_size])

# Đóng kết nối
con.close()
print("✅ Done.")
