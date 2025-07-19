from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# Đọc tệp unk_chars_thai.txt và lưu vào danh sách newdict2
newdict2 = []
with open('unk_chars_thai.txt', 'r', encoding='utf-8') as f:
    for line in f:
        token = line.strip()  # Loại bỏ ký tự thừa (như dấu cách hoặc dòng trống)
        newdict2.append(token)

# Tải mô hình và tokenizer từ Hugging Face
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Kiểm tra các token hiện tại trong tokenizer
curdict = tokenizer.get_vocab()

# Thêm các token mới vào tokenizer nếu chúng chưa có trong từ điển
for tok in tqdm(newdict2):
    if tok not in curdict:
        print(f"Adding: {tok} to tokenizer")
        tokenizer.add_tokens([tok])

# Sau khi thêm token mới, mô hình cũng cần được điều chỉnh lại kích thước embedding
model.resize_token_embeddings(len(tokenizer))


tokenizer.save_pretrained('/home/leloc/Document/USTH/Thesis/Machine_translation-Thai-Viet-/tokenizer')
model.save_pretrained('/home/leloc/Document/USTH/Thesis/Machine_translation-Thai-Viet-/model')

print(f"Total vocabulary size after update: {len(tokenizer)}")
