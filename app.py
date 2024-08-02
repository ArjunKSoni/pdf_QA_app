import warnings
from dotenv import load_dotenv
import os
load_dotenv()

from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import streamlit as st
from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import io

warnings.filterwarnings("ignore")



st.title("PDF Question Answering App")

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro",google_api_key=os.getenv("GEMINI_API_KEY"),
                             temperature=0.2,convert_system_message_to_human=True)

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    pdf_reader = PdfReader(io.BytesIO(uploaded_file.read()))
    context = ""
    for page_num in range(len(pdf_reader.pages)):
        context += pdf_reader.pages[page_num].extract_text()
     
     
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    texts = text_splitter.split_text(context)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=os.getenv("GEMINI_API_KEY"))

    vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k":5})



    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    qa_chain = RetrievalQA.from_chain_type(
        model,
        retriever=vector_index,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )
    
    question = st.text_input("Enter your question about the PDF:")

    if st.button("Get Answer"):
        result = qa_chain({"query": question})
        st.write("AI Response:")
        st.write(result["result"])