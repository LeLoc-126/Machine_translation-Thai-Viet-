from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Các ký tự bị thiếu
missing_thai_chars = [
    'ฌ', 'ฦ', 'ำ', '\u0e3b', '\u0e3c', '\u0e3d', '\u0e3e', '฿', 'ๅ', '๎', '๏',
    '๐', '๑', '๒', '๓', '๔', '๕', '๖', '๗', '๘', '๙', '๚', '๛', '\u0e5c', '\u0e5d',
    '\u0e5e', '\u0e5f', '\u0e60', '\u0e61', '\u0e62', '\u0e63', '\u0e64', '\u0e65',
    '\u0e66', '\u0e67', '\u0e68', '\u0e69', '\u0e6a', '\u0e6b', '\u0e6c', '\u0e6d',
    '\u0e6e', '\u0e6f', '\u0e70', '\u0e71', '\u0e72', '\u0e73', '\u0e74', '\u0e75',
    '\u0e76', '\u0e77', '\u0e78', '\u0e79', '\u0e7a', '\u0e7b', '\u0e7c', '\u0e7d',
    '\u0e7e', '\u0e7f'
]

# Convert Unicode escape sequences to actual characters
missing_thai_chars = [chr(int(c[2:], 16)) if c.startswith('\\u') else c for c in missing_thai_chars]

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B")

# Lọc những token chưa có trong tokenizer
new_tokens = [c for c in missing_thai_chars if tokenizer.convert_tokens_to_ids(c) == tokenizer.unk_token_id]
print(f"Số ký tự cần thêm: {len(new_tokens)}")
print(f"Ký tự cần thêm: {new_tokens}")

# Thêm token mới
tokenizer.add_tokens(new_tokens)
print("✅ Tokenizer đã được mở rộng.")

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B")

# Resize model embeddings
model.resize_token_embeddings(len(tokenizer))

# Test tokenization
test_input = "tha_Thai สวัสดี ฌ ฦ ำ ๑ ฿"
tokenized = tokenizer(test_input, return_tensors="pt")
print("Tokenized IDs:", tokenized.input_ids)
print("Tokens:", tokenizer.convert_ids_to_tokens(tokenized.input_ids[0]))

# Lưu tokenizer và model
tokenizer.save_pretrained("~/nllb-1.3B-thai-extended-tokenizer")
model.save_pretrained("~/nllb-1.3B-thai-extended-model")
print("✅ Tokenizer và model đã được lưu.")

# Note: Fine-tuning is required next (add your dataset and fine-tuning code here)