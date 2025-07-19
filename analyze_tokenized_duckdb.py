import duckdb
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from collections import Counter
import pickle
import seaborn as sns
from transformers import AutoTokenizer  # Nếu muốn giải mã token (nếu cần)

# 📂 Kết nối đến DuckDB
db_path = "/home/leloc/Document/USTH/Thesis/Data/translation.db"  # Đường dẫn DB
table_name = "translations"  # Tên bảng trong DB
con = duckdb.connect(db_path)
df = con.execute(f"SELECT thai_input_ids, vi_input_ids FROM {table_name} LIMIT 1000").fetchdf()

# 🧹 Unpickle dữ liệu
def unpickle(col):
    return [pickle.loads(b) if isinstance(b, (bytes, bytearray)) else b for b in col]

thai_ids = unpickle(df["thai_input_ids"])
vi_ids = unpickle(df["vi_input_ids"])

# 👋 Định nghĩa ID cho padding token (thường là 1)
PAD_TOKEN_ID = 1

# 🧹 Loại bỏ padding tokens trong câu
def remove_padding(tokens):
    return [token for token in tokens if token != PAD_TOKEN_ID]

# 📏 Tính độ dài câu sau khi loại bỏ padding
thai_lens = [len(remove_padding(x)) for x in thai_ids]
vi_lens = [len(remove_padding(x)) for x in vi_ids]
len_ratios = [t / v if v != 0 else 0 for t, v in zip(thai_lens, vi_lens)]

# 📊 Thống kê độ dài câu
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
}

# 🔢 Thống kê token
all_tokens = [token for row in thai_ids for token in remove_padding(row)]
token_counts = Counter(all_tokens)

# 📁 Thư mục lưu kết quả
output_dir = Path("EDA_tokenize")
output_dir.mkdir(exist_ok=True)

# 📈 Vẽ biểu đồ độ dài câu (Histogram)
plt.figure(figsize=(10, 6))
plt.hist(thai_lens, bins=50, alpha=0.6, color='skyblue', label="Thai", edgecolor='black')
plt.hist(vi_lens, bins=50, alpha=0.6, color='orange', label="Vietnamese", edgecolor='black')
plt.xlabel("Number of tokens (Sentence Length)")
plt.ylabel("Number of Sentences")
plt.title("Distribution of Sentence Lengths (Tokens)")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "sentence_length_histogram.png")

# 📉 Vẽ boxplot cho độ dài câu
plt.figure(figsize=(10, 6))
plt.boxplot([thai_lens, vi_lens], labels=["Thai", "Vietnamese"], vert=False)
plt.xlabel("Number of tokens (Sentence Length)")
plt.title("Boxplot of Sentence Lengths (Tokens)")
plt.tight_layout()
plt.savefig(output_dir / "sentence_length_boxplot.png")

# 💾 Lưu thống kê vào file JSON
stats_output_file = output_dir / "statistics.json"
with open(stats_output_file, "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2, default=lambda o: int(o) if isinstance(o, (np.integer,)) else float(o))

print("✅ Process completed and results saved at:", output_dir)
print(f"Statistics have been saved to the file {stats_output_file}")
