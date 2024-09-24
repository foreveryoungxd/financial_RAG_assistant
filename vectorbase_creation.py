from langchain_community.document_loaders import PyPDFLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

persist_directory = 'vectorbase'


def main():
    documents = []
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(root, file))
                new_docs = loader.load()
                documents.extend(new_docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # create embeddings
    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(model_name=model_name,
                                       model_kwargs=model_kwargs,
                                       encode_kwargs=encode_kwargs)
    print('Successfully created emdeddings for vectorstore')

    db = Chroma.from_documents(texts,
                               embeddings,
                               persist_directory=persist_directory
                               )

    print('Successfully created Chroma db')
    db.persist()
    db = None


if __name__ == '__main__':
    main()