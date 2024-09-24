import os
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain_community.chat_models.gigachat import GigaChat
from htmlTemplates import css, bot_template, user_template
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch


API_KEY = os.getenv('credentials')
checkpoint = "bond005/FRED-T5-large-instruct-v0.1"
model_path = "./models/FRED-T5-large-instruct-v0.1"


tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir="./models")
base_model = AutoModelForSeq2SeqLM.from_pretrained(
        checkpoint,
        device_map="auto",
        torch_dtype=torch.float32,
        cache_dir="./models"
    )
persist_directory = "vectorbase"


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(model_name=model_name,
                                      model_kwargs=model_kwargs,
                                      encode_kwargs=encode_kwargs)

    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    embedding_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    return embedding_retriever


def get_conversation_chain(embedding_retriever):
    llm = GigaChat(credentials=API_KEY,
                   model='GigaChat:latest',
                   verify_ssl_certs=False,
                   profanity_check=False
                   )
    prompt = ChatPromptTemplate.from_template('''Ответь на вопрос пользователя. \
    Используй при этом только информацию из контекста. Если в контексте нет \
    информации для ответа, сообщи об этом пользователю.
    Контекст: {context}
    Вопрос: {input}
    Ответ:'''
                                              )

    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )
    conversation_chain = create_retrieval_chain(embedding_retriever, document_chain)

    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation.invoke({'input': user_question})

    st.write('Ответ на Ваш вопрос:' + bot_template.replace(
        "{{MSG}}", response['answer']), unsafe_allow_html=True)


@st.cache_resource
def local_llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model=base_model,
        tokenizer=tokenizer,
        max_length=256,
        do_sample=True,
        temperature=0.3,
        top_p=0.95
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

@st.cache_resource
def qa_llm():
    llm = local_llm_pipeline()
    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(model_name=model_name,
                                       model_kwargs=model_kwargs,
                                       encode_kwargs=encode_kwargs)

    db = Chroma(persist_directory=persist_directory,
                embedding_function=embeddings)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        return_source_documents=True
    )
    return qa

# Если работаем с локальной БД и model_type == GigaChat
def giga_with_local_db():
    llm = GigaChat(credentials=API_KEY,
                   model='GigaChat:latest',
                   verify_ssl_certs=False,
                   profanity_check=False
                   )
    prompt = ChatPromptTemplate.from_template('''Ответь на вопрос пользователя. \
        Используй при этом только информацию из контекста. Если в контексте нет \
        информации для ответа, сообщи об этом пользователю.
        Контекст: {context}
        Вопрос: {input}
        Ответ:'''
                                              )

    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(model_name=model_name,
                                       model_kwargs=model_kwargs,
                                       encode_kwargs=encode_kwargs)
    db = Chroma(persist_directory=persist_directory,
                embedding_function=embeddings)
    embedding_retriever = db.as_retriever()
    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )
    conversation_chain = create_retrieval_chain(embedding_retriever, document_chain)

    return conversation_chain


def process_answer(instruction):
    qa = qa_llm()
    generated_text = qa.invoke(instruction)
    answer = generated_text['result']
    source_documents = generated_text['source_documents'][0]
    return answer, generated_text, source_documents


def main():
    load_dotenv()
    st.set_page_config(page_title='Финансовый AI-ассистент', page_icon=':books:')

    st.write(css, unsafe_allow_html=True)

    if 'conversation' not in st.session_state:
        st.session_state.conversation = None

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None

    st.header("Финансовый AI-ассистент :books:")
    st.write("""Это AI-ассистент в сфере финансовых вопросов.
                Вы можете задать вопрос по имеющейся базе знаний
                или загрузить свои файлы в формате PDF и найти
                ответ на интересующий Вас вопрос.
                При работе с локальной базой знаний вы можете использовать
                как локальную LLM, так и обращаться по API к GigaChat.""")
    # Выбор источника знаний
    knowledge_source = st.radio(
        "Использовать:",
        ("Существующую базу знаний", "Свои документы"),
        index=0,
        horizontal=True
    )

    if knowledge_source == "Существующую базу знаний":
        # Выбор модели
        model_type = st.radio(
            "Модель:",
            ("GigaChat", "Local FRED-T5-large-instruct-v0.1"),
            index=0,
            horizontal=True
        )

        user_quetion = st.text_input("""Задайте свой вопрос""")

        if model_type == "Local FRED-T5-large-instruct-v0.1":
            if user_quetion:
                answer, metadata, doc = process_answer(user_quetion)
                st.write('Ответ на Ваш вопрос:' + bot_template.replace(
                    "{{MSG}}", answer), unsafe_allow_html=True)
                st.write('Ниже Вы можете увидеть метаданные запроса.')
                #st.write(metadata)
                st.write(doc)

        if model_type == 'GigaChat':
            st.session_state.conversation = giga_with_local_db()
            if user_quetion:
                handle_userinput(user_quetion)

    if knowledge_source == "Свои документы":
        user_quetion = st.text_input("""Задайте свой вопрос""")
        if user_quetion:
            handle_userinput(user_quetion)

        with st.sidebar:
            st.subheader('Ваши документы')
            pdf_docs = st.file_uploader(
                'Выберите PDF-документы и нажмите "Обработать"', accept_multiple_files=True)
            if st.button('Обработать'):
                with st.spinner('Загрузка...'):

                    row_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(row_text)
                    vector_store = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vector_store)


if __name__ == '__main__':
    main()
