from transformers import AutoTokenizer
import duckdb
import pandas as pd
from tqdm import tqdm
import pickle

# === Config ===
tokenizer_path = "/sdd/lv01/leloc/translation_machine/model/tokenizer-nllb-extended"
db_path = "/sdd/lv01/leloc/translation_machine/translation.db"
source_table = "translations"
target_table = "translations_retokenized"
batch_size = 1024

# === Load tokenizer ===
print("🔄 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# === Connect DB and get total ===
print("🔌 Connecting to database...")
con = duckdb.connect(db_path)
total = con.execute(f"SELECT COUNT(*) FROM {source_table}").fetchone()[0]
print(f"📊 Total rows to process: {total:,}")

# === Create new table if not exists ===
con.execute(f"""
CREATE TABLE IF NOT EXISTS {target_table} (
    Thai TEXT,
    Viet TEXT,
    thai_input_ids BLOB,
    vi_input_ids BLOB
)
""")

# === Tokenize helper ===
def tokenize_column(texts, lang_token):
    texts_with_lang = [f">>{lang_token}<< {t}" for t in texts]
    return tokenizer(
        texts_with_lang,
        padding=False,
        truncation=True,
        return_attention_mask=False
    )["input_ids"]

# === Process in batches with progress ===
progress = tqdm(range(0, total, batch_size), desc="🚀 Retokenizing", unit="rows")

for offset in progress:
    df = con.execute(f"""
        SELECT Thai, Viet FROM {source_table}
        LIMIT {batch_size} OFFSET {offset}
    """).fetch_df()

    df["thai_input_ids"] = df["thai_input_ids"].apply(lambda x: pickle.dumps(x))
    df["vi_input_ids"] = df["vi_input_ids"].apply(lambda x: pickle.dumps(x))

    con.register("batch_df", df)
    con.execute(f"""
        INSERT INTO {target_table}
        SELECT Thai, Viet, thai_input_ids, vi_input_ids FROM batch_df
    """)
    con.unregister("batch_df")

    processed = min(offset + batch_size, total)
    progress.set_postfix_str(f"{processed:,}/{total:,} done")

# === Kiểm tra lại tổng số dòng đã insert ===
retok_total = con.execute(f"SELECT COUNT(*) FROM {target_table}").fetchone()[0]
print(f"\n✅ Retokenization complete: {retok_total:,} dòng đã được xử lý")
