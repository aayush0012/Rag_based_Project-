import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from langchain_community.document_loaders import TextLoader, DirectoryLoader ## read text files,ppts,docsx files 
from langchain_text_splitters import RecursiveCharacterTextSplitter # for chunking the documents
from langchain_openai import OpenAIEmbeddings  ## vector embeddings 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma  # vector database
from dotenv import load_dotenv
load_dotenv()



def load_documents(folder_name="docs"):

    # folder ke andar ki saari txt files load karega
    loader = DirectoryLoader(
        path=folder_name,      # konsa folder read karna h
        glob="*.txt",          # sirf txt files load hongi
        loader_cls=TextLoader  # txt files ko read karega
    )

    # files ko Document objects me convert karega
    documents = loader.load()
    # agar folder empty hua
    if len(documents) == 0:
        raise FileNotFoundError("No txt files found")
    # return type -> list[Document]
    return documents




def split_documents(documents,
                    chunk_size=1000,
                    chunk_overlap=0):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    return chunks



# CONVERT ALL THE CHUNKS INTO VECTOR EMBEDDINGS AND STORE IN CHROMADB VECTOR DATABASE
def create_vector_store(chunks,
                        persist_directory="db/chroma_db"):

    # text ko vector embeddings me convert karega
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # chunks + embeddings ko ChromaDB me store karega
    vectorstore = Chroma.from_documents(

        documents=chunks,  # jo chunks banaye the

        embedding=embedding_model,  
        # embedding model jo text ko vectors me convert karega

        persist_directory=persist_directory,
        # vector database kis folder me save hogi

        collection_metadata={"hnsw:space": "cosine"}
        # cosine similarity use hogi vectors compare karne ke liye
    )

    # return type -> Chroma object
  # print("Done boss !")
    return vectorstore



def main() :
    documents= load_documents(folder_name="docs") 
    chunks = split_documents(documents); 
    vector  = create_vector_store(chunks, persist_directory="db/chroma_db")
    
    
    
if __name__ == "__main__":
    main() 
    