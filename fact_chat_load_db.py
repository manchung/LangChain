from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(
    #  model="text-embedding-3-large",
)


text_splitter = CharacterTextSplitter(
    separator='\n',
    chunk_size=200,
    chunk_overlap=0
)

loader = TextLoader('facts.txt')
docs = loader.load_and_split(
    text_splitter=text_splitter
)

db = Chroma.from_documents(
    docs,
    embedding=embeddings,
    persist_directory='emb'
)

# results = db._similarity_search_with_relevance_scores('What is an interesting fact about the English language')

# for r in results:
#     print(r[1])
#     print(r[0].page_content)
#     print('\n')