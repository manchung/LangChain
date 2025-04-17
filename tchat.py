from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4.1-nano",
)

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ('placeholder', '{history}'),
    ('user', '{input}'),
    ])

base_chain = chat_prompt | llm
memory_store = InMemoryChatMessageHistory()

chat_chain = RunnableWithMessageHistory(
    base_chain,
    lambda session_id: memory_store,
    input_messages_key="input",
    history_messages_key="history"
)
session_id = "user-123"

while True:
    user_input = input('>> ')
    response = chat_chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )
    print(response.content)
