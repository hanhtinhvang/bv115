import streamlit as st
from transformers import pipeline
import docx2txt
import PyPDF2

def load_file(file):
    if file.type == "text/plain":
        return file.read().decode("utf-8")
    elif file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return docxtxt.process(file)
    return ""

st.title("Chatbot QA Local - Không dùng ChatGPT")
uploaded_file = st.file_uploader("Upload file (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"])

if uploaded_file:
    raw_text = load_file(uploaded_file)
    if raw_text.strip():
        # Có thể thay bằng model tiếng Việt của Huggingface, nếu bạn muốn
        qa = pipeline("question-answering", model="deepset/roberta-base-squad2")
        st.success("File đã được xử lý, nhập câu hỏi về nội dung file...")
        question = st.text_input("Bạn hỏi gì?")
        if st.button("Trả lời") and question:
            # Chia nhỏ raw_text nếu quá dài (transformers giới hạn ~500 token context!)
            context = raw_text[:2000]
            result = qa(question=question, context=context)
            st.write("🤖", result["answer"])
    else:
        st.error("Không đọc được nội dung file!")
else:
    st.info("Vui lòng upload file để bắt đầu.")