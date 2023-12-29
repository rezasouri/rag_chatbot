from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import os
# os.environ['CURL_CA_BUNDLE'] = ''

FILE_PATH = '../documents/Rich-Dad-Poor-Dad.pdf'

# Create loader
loader = PyPDFLoader(file_path=FILE_PATH)

# split document
pages = loader.load_and_split()
# os.environ['CURL_CA_BUNDLE'] = ''
# embbeding function
embedding_function = SentenceTransformerEmbeddings(
    model_name = "all-MiniLM-L6-v2"
)

# create vector store
vectordb = Chroma.from_documents(
    documents = pages,
    embedding = embedding_function,
    persist_directory = "../vectordb",
    collection_name = "rich_dad_poor_dad"
)

# make persistant
vectordb.persist()