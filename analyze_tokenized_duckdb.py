import duckdb
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from collections import Counter
import pickle  # 👈 Used to load pickle from byte

# 📂 Connect to DuckDB
db_path = "/home/leloc/Document/USTH/Thesis/Data/translation.db"  # Change the path to your DB
table_name = "translations"  # Change the table name if necessary
con = duckdb.connect(db_path)
df = con.execute(f"SELECT thai_input_ids, vi_input_ids FROM {table_name} LIMIT 10").fetchdf()

# 🧹 Unpickle the data
def unpickle(col):
    return [pickle.loads(b) if isinstance(b, (bytes, bytearray)) else b for b in col]

thai_ids = unpickle(df["thai_input_ids"])
vi_ids = unpickle(df["vi_input_ids"])

# 👋 Define padding token ID, typically 1
PAD_TOKEN_ID = 1

# 🧹 Remove padding tokens from sentences
def remove_padding(tokens):
    return [token for token in tokens if token != PAD_TOKEN_ID]

# 📏 Calculate the length of each sentence after removing padding
thai_lens = [len(remove_padding(x)) for x in thai_ids]
vi_lens = [len(remove_padding(x)) for x in vi_ids]
len_ratios = [t / v if v != 0 else 0 for t, v in zip(thai_lens, vi_lens)]

# 📊 Statistics
stats = {
    "thai_input_ids": {
        "max": int(np.max(thai_lens)),
        "min": int(np.min(thai_lens)),
        "mean": float(np.mean(thai_lens)),
        "median": float(np.median(thai_lens)),
    },
    "vi_input_ids": {
        "max": int(np.max(vi_lens)),
        "min": int(np.min(vi_lens)),
        "mean": float(np.mean(vi_lens)),
        "median": float(np.median(vi_lens)),
    },
    "length_ratio (thai/vi)": {
        "max": float(np.max(len_ratios)),
        "min": float(np.min(len_ratios)),
        "mean": float(np.mean(len_ratios)),
        "median": float(np.median(len_ratios)),
    },
    "truncate_rate_128": float(np.sum(np.array(thai_lens) > 128) / len(thai_lens)),
}

# 🔢 Token statistics, remove padding tokens
all_tokens = [token for row in thai_ids for token in remove_padding(row)]
token_counts = Counter(all_tokens)
stats["total_tokens"] = len(all_tokens)
stats["unique_tokens"] = len(token_counts)
stats["top_tokens"] = token_counts.most_common(10)

# 📁 Output directory for results
output_dir = Path("duckdb_tokenize_stats")
output_dir.mkdir(exist_ok=True)

# 📈 Length distribution histogram
plt.figure(figsize=(10, 4))
plt.hist(thai_lens, bins=50, alpha=0.6, label="thai")
plt.hist(vi_lens, bins=50, alpha=0.6, label="vietnamese")
plt.xlabel("Number of tokens")
plt.ylabel("Number of sentences")
plt.title("Sentence Length Distribution")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "length_histogram.png")

# 📉 Length ratio distribution histogram
plt.figure(figsize=(8, 4))
plt.hist(len_ratios, bins=50, color="purple", alpha=0.8)
plt.xlabel("Length ratio: len(thai) / len(vi)")
plt.ylabel("Number of sentences")
plt.title("Length Ratio Distribution Thai/Vietnamese")
plt.tight_layout()
plt.savefig(output_dir / "length_ratio_histogram.png")

# 💾 Save statistics
stats_output_file = output_dir / "statistics.json"
with open(stats_output_file, "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2, default=lambda o: int(o) if isinstance(o, (np.integer,)) else float(o) if isinstance(o, (np.floating,)) else str(o))

print("✅ Process completed and results saved at:", output_dir)
print(f"Statistics have been saved to the file {stats_output_file}")
