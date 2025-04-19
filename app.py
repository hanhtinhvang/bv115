import streamlit as st
from haystack.nodes import FARMReader
from haystack.nodes import PreProcessor
from haystack.document_stores import InMemoryDocumentStore
import docx2txt
import PyPDF2

def load_file(file):
    if file.type == "text/plain":
        return file.read().decode("utf-8")
    elif file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return docx2txt.process(file)
    return ""

st.title("🤖 Chatbot hỏi đáp theo file tài liệu (không dùng ChatGPT/OpenAI)")

uploaded_file = st.file_uploader("Tải file TXT, DOCX, PDF", type=['txt', 'pdf', 'docx'])
if not uploaded_file:
    st.info("Vui lòng tải file văn bản.")
    st.stop()

text = load_file(uploaded_file)
if not text.strip():
    st.error("Không đọc được nội dung file.")
    st.stop()

st.info("Đang xử lý tài liệu (quá trình này chỉ tải mô hình AI về lần đầu)...")

# Chuẩn bị dữ liệu
preprocessor = PreProcessor(
    split_length=250,
    split_overlap=20,
    split_respect_sentence_boundary=True
)
docs = [{'content': chunk} for chunk in preprocessor.process([{'content': text}])[0]['splits']]
document_store = InMemoryDocumentStore(use_bm25=True)
document_store.write_documents(docs)

# Tải mô hình extractive QA (chạy trên CPU)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)

st.success("Sẵn sàng! Đặt câu hỏi về tài liệu.")

question = st.text_input("Bạn hỏi gì?")
if st.button("Trả lời") and question:
    prediction = reader.predict_on_texts(question, [d['content'] for d in docs])
    answers = prediction['answers']
    if answers and answers[0].answer:
        st.write("🤖 **Trả lời:**", answers[0].answer)
    else:
        st.write("Không tìm ra câu trả lời phù hợp.")