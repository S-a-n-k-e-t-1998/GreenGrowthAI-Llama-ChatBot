from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 


from params import (CHROMA_DB_SETTINGS,EMBEDDING_MODEL_NAME,LLM_MODEL_PATH,SOURCE_DIR,PERSIST_DIR)

# create vector database

def create_vector_db():
    
    loader=DirectoryLoader(SOURCE_DIR,
                           glob='*.pdf',
                           loader_cls=PyPDFLoader)
    documents=loader.load()

    # chunking the document
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,
                                                 chunk_overlap=10)
    texts=text_splitter.split_documents(documents)
    # print(texts)

    # create the embedding
    embeddings=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME,
                                     model_kwargs={'device':'cpu'})
    
    db=Chroma.from_documents(texts,
                             embeddings,
                             persist_directory=PERSIST_DIR,
                             client_settings=CHROMA_DB_SETTINGS)
    db.persist()
    db=None

if __name__=="__main__":
    create_vector_db()
    print("Embedding Store Successfully...")