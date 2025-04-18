from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4.1-nano",
)

embeddings = OpenAIEmbeddings()
db = Chroma(
    persist_directory='emb',
    embedding_function=embeddings
)
retriever = db.as_retriever()

chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type='stuff',
)

result = chain.invoke({
        'query': 'Do you know anything about panda?'
    })
print(result)