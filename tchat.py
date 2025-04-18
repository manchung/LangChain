from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain.memory import ConversationSummaryMemory
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4.1-nano",
)

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    # ('system', 'Conversation so far: {summary}'),
    ('placeholder', '{history}'),
    ('user', '{input}'),
    ])

base_chain = chat_prompt | llm

# Use in-memory store
# memory_store = InMemoryChatMessageHistory()

# Use file-based memory store
memory_store = FileChatMessageHistory("messages.json")

# Use summary memory store
# memory_store = ConversationSummaryMemory(
#     llm=llm,
#     memory_key="summary",
#     return_messages=False
# )

chat_chain = RunnableWithMessageHistory(
    base_chain,
    lambda session_id: memory_store,
    input_messages_key="input",
    history_messages_key="history",
)
session_id = "user-123"

while True:
    user_input = input('>> ')
    response = chat_chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )
    print(response.content)
