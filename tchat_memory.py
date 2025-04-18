from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4.1-nano",
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ('placeholder', '{history}'),
    ('user', '{input}'),
    ])

# Use file-based memory store
memory_store = FileChatMessageHistory("messages.json")

chat_chain = RunnableWithMessageHistory(
    prompt | llm,
    lambda session: memory_store,
    input_messages_key="input",
    history_messages_key="history",
)

MAX_TURNS = 2      # summarise when you exceed this

def summarise_if_needed(chain_input):
    """Mutates the on‑disk history in place, replacing many turns with 1 summary."""

    if len(memory_store.messages) < MAX_TURNS:
        return False                          # nothing to do

    # Build a tiny summarisation chain
    summary_prompt = ChatPromptTemplate.from_messages([
        ('placeholder', '{history}'),
        ("user", "Condense the above dialog into 1-2 sentences, "
                 "keeping all concrete facts about the user."),
    ])
    summary_chain = summary_prompt | llm

    summary_msg = summary_chain.invoke({"history": memory_store.messages})
    memory_store.clear()                      # drop all old messages
    memory_store.add_message(summary_msg)     # store the summary instead
    return True                       # optional debug flag

smart_chain = (
    RunnablePassthrough.assign(_=summarise_if_needed)   # side‑effect first
    | chat_chain                                        # real chat work
)

print("🤖 Chat with summarizing memory. Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() in {"exit", "quit"}:
        break

    response = smart_chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": "user-123"}}  # required for memory to work
    )
    print(f"AI: {response.content}\n")
    # print(memory_store.messages)

