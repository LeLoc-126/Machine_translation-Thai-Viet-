import pandas as pd
import os
import logging # Import logging ở đầu file

# ======================= CẤU HÌNH =======================
INPUT_CSV = '/home/leloc/Document/USTH/Thesis/Data/final_fully_cleaned_data.csv'
NUM_TEST_SAMPLES = 2000
THAI_COL = 'Thai'
VIET_COL = 'Viet'
OUTPUT_DIR = '/home/leloc/Document/USTH/Thesis/Data/opensubtitles_test_split'
TRAIN_OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'opensubtitles_train.csv')
TEST_THAI_OUTPUT = os.path.join(OUTPUT_DIR, 'tha_Thai.devtest')
TEST_VIET_OUTPUT = os.path.join(OUTPUT_DIR, 'vie_Latn.devtest')
RANDOM_SEED = 42
# ==========================================================

# === SỬA LỖI: Thiết lập logger ở phạm vi toàn cục ===
# Bằng cách này, mọi hàm trong script đều có thể sử dụng biến 'logger'.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# ==================================================

def create_in_domain_test_set():
    """
    Tách một phần dữ liệu từ file CSV lớn để tạo thành tập test
    và lưu phần còn lại làm dữ liệu training.
    """
    logger.info(f"Đang đọc file dữ liệu lớn từ: {INPUT_CSV}")
    try:
        df = pd.read_csv(INPUT_CSV)
        df = df.dropna(subset=[THAI_COL, VIET_COL])
        logger.info(f"Đã đọc thành công {len(df)} cặp câu.")
    except FileNotFoundError:
        logger.error(f"Lỗi: Không tìm thấy file '{INPUT_CSV}'. Vui lòng kiểm tra lại đường dẫn.")
        return
    except Exception as e:
        logger.error(f"Lỗi khi đọc file CSV: {e}")
        return

    if len(df) < NUM_TEST_SAMPLES:
        logger.error(f"Lỗi: Dữ liệu không đủ ({len(df)} dòng) để tạo tập test {NUM_TEST_SAMPLES} dòng.")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        logger.info(f"Đã tạo thư mục output tại: {OUTPUT_DIR}")

    logger.info(f"Đang lấy ngẫu nhiên {NUM_TEST_SAMPLES} mẫu để làm tập test...")
    test_df = df.sample(n=NUM_TEST_SAMPLES, random_state=RANDOM_SEED)

    train_df = df.drop(test_df.index)
    logger.info(f"Tập training còn lại {len(train_df)} mẫu.")

    # --- LƯU CÁC FILE ---
    try:
        train_df.to_csv(TRAIN_OUTPUT_CSV, index=False, encoding='utf-8')
        logger.info(f"✅ Đã lưu dữ liệu training vào: {TRAIN_OUTPUT_CSV}")
    except Exception as e:
        logger.error(f"Lỗi khi lưu file training: {e}")

    try:
        with open(TEST_THAI_OUTPUT, 'w', encoding='utf-8') as f_thai:
            for line in test_df[THAI_COL]:
                f_thai.write(str(line) + '\n')
        logger.info(f"✅ Đã lưu tập test tiếng Thái vào: {TEST_THAI_OUTPUT}")

        with open(TEST_VIET_OUTPUT, 'w', encoding='utf-8') as f_viet:
            for line in test_df[VIET_COL]:
                f_viet.write(str(line) + '\n')
        logger.info(f"✅ Đã lưu tập test tiếng Việt vào: {TEST_VIET_OUTPUT}")

    except Exception as e:
        logger.error(f"Lỗi khi lưu các file test: {e}")
        
    print("\n🎉 Hoàn thành! Quá trình tách dữ liệu đã xong.")


if __name__ == "__main__":
    # Khối này bây giờ chỉ còn nhiệm vụ gọi hàm chính
    create_in_domain_test_set()