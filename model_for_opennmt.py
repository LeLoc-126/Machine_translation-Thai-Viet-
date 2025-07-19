from onmt.bin.translate import main as onmt_translate
# Dòng trên có vẻ không cần thiết, nhưng chúng ta sẽ dùng một công cụ khác
# Thực tế, công cụ chuyển đổi được cài đặt cùng ctranslate2
import os
import ctranslate2

print("Bắt đầu chuyển đổi mô hình...")

# Lệnh này sẽ tự động tải model từ Hub và chuyển đổi
# --model: Tên model trên Hugging Face
# --output_dir: Thư mục để lưu model đã chuyển đổi
# --quantization: Sử dụng float16 để tiết kiệm dung lượng
command = "ct2-transformers-converter --model facebook/nllb-200-distilled-600M --output_dir nllb-600m-ct2 --quantization float16 --force"

# Chạy lệnh từ shell
os.system(command)

print(f"✅ Chuyển đổi hoàn tất! Mô hình đã được lưu tại thư mục 'nllb-600m-ct2'")