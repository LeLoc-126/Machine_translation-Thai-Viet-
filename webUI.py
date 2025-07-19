import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load mô hình đã fine-tune (local hoặc từ Hugging Face Hub)
model_name = "facebook/nllb-200-distilled-1.3B"  # hoặc 'your-username/nllb-finetuned-vi-th'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Mapping friendly names to NLLB language codes
language_map = {
    "Tiếng Thái": "tha_Thai",
    "Tiếng Việt": "vie_Latn",
}

st.title("🌐 Chatbot Dịch Ngôn Ngữ")
st.write("Sử dụng mô hình NLLB-200 đã fine-tune để dịch giữa Thái, Việt")

with st.form("translate_form"):
    text = st.text_area("Nhập văn bản:")
    col1, col2 = st.columns(2)
    with col1:
        src_lang = st.selectbox("Ngôn ngữ nguồn", list(language_map.keys()), index=0)
    with col2:
        tgt_lang = st.selectbox("Ngôn ngữ đích", list(language_map.keys()), index=1)

    submitted = st.form_submit_button("Dịch")

if submitted and text.strip() != "":
    with st.spinner("Đang dịch..."):
        src_code = language_map[src_lang]
        tgt_code = language_map[tgt_lang]

        inputs = tokenizer(text, return_tensors="pt")
        inputs["forced_bos_token_id"] = tokenizer.convert_tokens_to_ids(tgt_code)

        output_tokens = model.generate(**inputs, max_length=512)
        translated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

        st.success("🎉 Dịch thành công!")
        st.text_area("Kết quả dịch:", translated_text, height=150)
