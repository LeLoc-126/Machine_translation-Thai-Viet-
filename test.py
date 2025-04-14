import duckdb
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from tqdm import tqdm
import os
import gc

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B").to(device)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Sử dụng FP16 để tiết kiệm VRAM
model = model.half().to("cuda")

# Kết nối DuckDB
con = duckdb.connect("translation.db")

# Lấy dữ liệu tiếng Thái
df = con.execute("SELECT thai FROM translations").fetchdf()

# Thiết lập mã ngôn ngữ
tokenizer.src_lang = "tha_Thai"
tgt_lang_code = "vie_Latn"
forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang_code)

# Hàm chia batch
def chunks(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

batch_size = 16
all_results = []
all_thai_tokens = []
all_viet_tokens = []

def tokenize_and_show(text, lang):
    tokenizer.src_lang = lang
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    decoded = tokenizer.decode(token_ids)
    return tokens, token_ids, decoded

# Bắt đầu dịch theo batch
for batch in tqdm(list(chunks(df["thai"].tolist(), batch_size)), desc="Translating", unit="batch"):
    inputs = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=64
        )

    translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    thai_tokens = []
    viet_tokens = []

    print("\n--- Tokenizing batch ---\n")
    for i in tqdm(range(len(batch)), desc="Tokenizing", unit="sent", leave=False):
        thai_text = batch[i]
        viet_text = translations[i]

        # Thai tokenization
        thai_subwords, thai_ids, thai_decoded = tokenize_and_show(thai_text, "tha_Thai")
        # Vietnamese tokenization
        viet_subwords, viet_ids, viet_decoded = tokenize_and_show(viet_text, "vie_Latn")

        thai_tokens.append(thai_ids)
        viet_tokens.append(viet_ids)

        print(f"[THAI] {thai_text}")
        print(f"  ▸ Subwords: {thai_subwords}")
        print(f"  ▸ Token IDs: {thai_ids}")
        print(f"  ▸ Decoded from IDs: {thai_decoded}")
        print(f"[VIET] {viet_text}")
        print(f"  ▸ Subwords: {viet_subwords}")
        print(f"  ▸ Token IDs: {viet_ids}")
        print(f"  ▸ Decoded from IDs: {viet_decoded}")
        print("------")

    all_results.extend(zip(batch, translations))
    all_thai_tokens.extend(thai_tokens)
    all_viet_tokens.extend(viet_tokens)

    del inputs, outputs
    torch.cuda.empty_cache()
    gc.collect()

# Chuẩn bị kết quả cuối cùng
thai_list, viet_list = zip(*all_results)
result_df = pd.DataFrame({
    "thai": thai_list,
    "viet_text": viet_list,
    "thai_tokens": [str(x) for x in all_thai_tokens],
    "viet_tokens": [str(x) for x in all_viet_tokens]
})

# Tạo bảng nếu chưa tồn tại
con.execute("""
    CREATE TABLE IF NOT EXISTS translated_tokenized (
        thai TEXT,
        thai_tokens TEXT,
        viet_text TEXT,
        viet_tokens TEXT
    )
""")

# Ghi dữ liệu
con.register("result_df_view", result_df)
con.execute("INSERT INTO translated_tokenized SELECT * FROM result_df_view")

print("\n==== Kết quả dịch thử ====")
for i, row in result_df.iterrows():
    print(f"[THAI] {row['thai']}")
    print(f"[VIET] {row['viet_text']}")
    print(f"[THAI TOKENS] {row['thai_tokens']}")
    print(f"[VIET TOKENS] {row['viet_tokens']}")
    print("------")

# Dọn dẹp
torch.cuda.empty_cache()
gc.collect()
con.close()