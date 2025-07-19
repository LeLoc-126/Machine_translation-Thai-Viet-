import duckdb
import os
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pythainlp.tokenize import word_tokenize

db_path = "/sdd/lv01/leloc/translation_machine/translation.db"
con = duckdb.connect(db_path)

def build_vocab(con, table, column, is_thai=False, min_freq=2):
    counter = Counter()
    query = f"SELECT {column} FROM {table}"
    for (text,) in con.execute(query).fetchall():
        if not text:
            continue
        words = word_tokenize(text, engine="newmm") if is_thai else text.split()
        words = [w for w in words if len(w) >= 2 and w.isalpha()]
        counter.update(words)
    return counter
thai_counter = build_vocab(con, "translations", "thai", is_thai=True, min_freq=2)
viet_counter = build_vocab(con, "translations", "viet", is_thai=False, min_freq=2)

combined_counter = thai_counter + viet_counter
filtered_vocab = {word for word, count in combined_counter.items() if count >= 2}

model_name = "facebook/nllb-200-distilled-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

token_split_score = {}
for word in filtered_vocab:
    if len(word) < 2 or not word.isalpha():
        continue
    split_len = len(tokenizer.tokenize(word))
    if split_len > 1:
        token_split_score[word] = split_len

sorted_tokens = sorted(token_split_score.items(), key=lambda x: (-x[1], -combined_counter[x[0]]))
new_tokens = [w for w, _ in sorted_tokens[:5000]]

print(f"🆕 Số lượng token được thêm: {len(new_tokens)}")

if new_tokens:
    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))
    print("✅ Tokenizer đã được mở rộng và model đã resize.")

    # 8. Lưu
    tokenizer_dir = "tokenizer-nllb-extended"
    model_dir = "nllb-extended"
    os.makedirs(tokenizer_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_dir)
    model.save_pretrained(model_dir)

    # 9. Nén lại
    os.system(f"zip -r tokenizer-nllb-extended.zip {tokenizer_dir}")
    os.system(f"zip -r nllb-extended.zip {model_dir}")
    print("📦 Đã lưu và nén tokenizer + model mở rộng.")
else:
    print("⚠️ Không có token mới cần thêm.")

