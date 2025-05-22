import pickle
from transformers import AutoTokenizer
import duckdb
import pandas as pd
from tqdm import tqdm

# === Config ===
tokenizer_path = "/sdd/lv01/leloc/translation_machine/model/tokenizer-nllb-extended"
db_path = "/sdd/lv01/leloc/translation_machine/translation.db"
source_table = "translations"
target_table = "translations_retokenized"
batch_size = 1024

# === Load tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# === Connect DB and get total count ===
con = duckdb.connect(db_path)
total = con.execute(f"SELECT COUNT(*) FROM {source_table}").fetchone()[0]

# === Create target table if not exists ===
con.execute(f"""
CREATE TABLE IF NOT EXISTS {target_table} (
    thai TEXT,
    viet TEXT,
    thai_input_ids BLOB,
    vi_input_ids BLOB
)
""")

def tokenize_column(texts, lang_token):
    texts_with_lang = [f">>{lang_token}<< {t}" for t in texts]
    return tokenizer(
        texts_with_lang,
        padding=False,
        truncation=True,
        return_attention_mask=False
    )["input_ids"]

# === Process in batches with tqdm progress bar ===
for offset in tqdm(range(0, total, batch_size), desc="Retokenizing"):
    # Load batch from source table
    df = con.execute(f"""
        SELECT thai, viet FROM {source_table}
        LIMIT {batch_size} OFFSET {offset}
    """).fetch_df()

    # Tokenize columns
    df["thai_input_ids"] = tokenize_column(df["thai"].tolist(), "tha_Thai")
    df["vi_input_ids"] = tokenize_column(df["viet"].tolist(), "vie_Latn")

    # Serialize lists to bytes for BLOB storage
    df["thai_input_ids"] = df["thai_input_ids"].apply(lambda x: pickle.dumps(x))
    df["vi_input_ids"] = df["vi_input_ids"].apply(lambda x: pickle.dumps(x))

    # Register dataframe as a temporary table in DuckDB
    con.register("batch_df", df)

    # Insert into target table
    con.execute(f"""
        INSERT INTO {target_table}
        SELECT thai, viet, thai_input_ids, vi_input_ids FROM batch_df
    """)
    con.unregister("batch_df")

print("✅ Retokenization complete.")
