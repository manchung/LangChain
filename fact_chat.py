from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from redundant_filter_retriever import RedundantFilterRetriever
from dotenv import load_dotenv
import langchain

langchain.debug = True

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4.1-nano",
)

embeddings = OpenAIEmbeddings()
db = Chroma(
    persist_directory='emb',
    embedding_function=embeddings
)
# retriever = db.as_retriever()
retriever = RedundantFilterRetriever(
    embeddings=embeddings, 
    chroma=db
)

chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type='stuff',
)

result = chain.invoke({
        'query': 'Do you know interesting facts about the English language?'
    })
print(result)