import os
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
import pickle
import numpy as np

# --- Cấu hình giữ nguyên ---
INPUT_CSV = '/home/leloc/Document/USTH/Thesis/Data/final_filtered_v2.csv' 
OUTPUT_FILE = '/home/leloc/Document/USTH/Thesis/Data/tokenized_10k_sample.pkl' 
SAMPLE_SIZE = 10000 
THAI_COL = 'Thai'  
VIET_COL = 'Viet'  
MODEL_PATH = '/home/leloc/Document/USTH/Thesis/Machine_translation-Thai-Viet-/my_updated_nllb_tokenizer'
MAX_LENGTH = 128

def main():
    # Các bước 1, 2, 3 không thay đổi
    print(f"🔄 Đang lấy mẫu {SAMPLE_SIZE} dòng ngẫu nhiên từ '{INPUT_CSV}'...")
    try:
        num_lines = sum(1 for _ in open(INPUT_CSV, encoding='utf-8')) - 1
        if num_lines < SAMPLE_SIZE:
            df_sample = pd.read_csv(INPUT_CSV, encoding='utf-8')
        else:
            skip_indices = np.random.choice(np.arange(1, num_lines + 1), size=num_lines - SAMPLE_SIZE, replace=False)
            df_sample = pd.read_csv(INPUT_CSV, encoding='utf-8', skiprows=skip_indices)
        print(f"✅ Đã tải thành công {len(df_sample)} dòng mẫu.")
    except Exception as e:
        print(f"❌ Lỗi khi đọc file CSV: {str(e)}")
        return

    print(f"🔄 Đang tải tokenizer '{MODEL_PATH}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, src_lang="tha_Thai", tgt_lang="vie_Latn")
    except Exception as e:
        print(f"❌ Lỗi khi tải tokenizer: {str(e)}")
        return

    thai_texts = df_sample[THAI_COL].fillna('').astype(str).tolist()
    viet_texts = df_sample[VIET_COL].fillna('').astype(str).tolist()
    
    print("⏳ Bắt đầu tokenize dữ liệu mẫu...")
    model_inputs = tokenizer(thai_texts, max_length=MAX_LENGTH, padding="max_length", truncation=True, return_tensors="pt")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(viet_texts, max_length=MAX_LENGTH, padding="max_length", truncation=True, return_tensors="pt")
    model_inputs['labels'] = labels['input_ids']
    print("✅ Tokenization hoàn thành.")

    # Bước 4: Lưu kết quả
    print(f"💾 Đang lưu kết quả vào file '{OUTPUT_FILE}'...")
    try:
        with open(OUTPUT_FILE, 'wb') as f:
            pickle.dump(model_inputs, f)
        print(f"✅ Đã lưu thành công!")
    except Exception as e:
        print(f"❌ Lỗi khi lưu file pickle: {str(e)}")
        return

    # === BƯỚC 5: KẾT THÚC SCRIPT TẠI ĐÂY ===
    print("\n🎉 Hoàn thành! File tokenized_10k_sample.pkl đã được tạo thành công.")
    print("Bỏ qua bước kiểm tra để tránh lỗi hết bộ nhớ.")

if __name__ == "__main__":
    main()