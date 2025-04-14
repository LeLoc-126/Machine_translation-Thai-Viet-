import pickle
import duckdb
from transformers import AutoTokenizer
from tqdm import tqdm

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B")

# Set up DB connection
db_path = '/home/leloc/Document/USTH/Thesis/translation.db'
con = duckdb.connect(db_path)

# Verify schema
schema_check = con.execute("PRAGMA table_info(translations)").fetchall()
required_columns = {"thai", "viet", "thai_input_ids", "thai_attention_mask", "vi_input_ids", "vi_attention_mask"}
actual_columns = {col[1] for col in schema_check}
if not required_columns.issubset(actual_columns):
    raise ValueError(f"Database schema missing required columns: {required_columns - actual_columns}")

# Function to tokenize a batch and update the database
def tokenize_batch(batch_start, batch_size, db_path, tokenizer, con):
    try:
        # Load batch from database
        df = con.execute(f"""
            SELECT rowid AS id, thai, viet
            FROM translations
            LIMIT {batch_size} OFFSET {batch_start}
        """).fetchdf()

        if df.empty:
            print(f"No data found for batch starting at {batch_start}")
            return

        # Tokenize Thai
        thai_tokens = tokenizer(
            df["thai"].tolist(),
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            max_length=256
        )

        # Tokenize Vietnamese
        vi_tokens = tokenizer(
            df["viet"].tolist(),
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            max_length=256
        )

        # Prepare data for insertion (match query order)
        data = [
            (
                pickle.dumps(thai_tokens["input_ids"][i].numpy()),
                pickle.dumps(thai_tokens["attention_mask"][i].numpy()),
                pickle.dumps(vi_tokens["input_ids"][i].numpy()),
                pickle.dumps(vi_tokens["attention_mask"][i].numpy()),
                int(df.iloc[i]["id"])  # rowid last
            )
            for i in range(len(df))
        ]

        # Update the database
        con.executemany("""
            UPDATE translations
            SET
                thai_input_ids = ?, 
                thai_attention_mask = ?, 
                vi_input_ids = ?, 
                vi_attention_mask = ?
            WHERE rowid = ?
        """, data)

    except Exception as e:
        print(f"Error processing batch {batch_start}: {str(e)}")

# Call the function for tokenizing and updating
batch_size = 200  # Reduced for faster iteration
total_rows = con.execute("SELECT COUNT(*) FROM translations").fetchone()[0]
for batch_start in tqdm(range(0, total_rows, batch_size), desc="Tokenizing batches"):
    tokenize_batch(batch_start, batch_size, db_path, tokenizer, con)

con.close()
print("✅ Tokenization and database update complete.")