from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import create_openai_functions_agent, AgentExecutor
from dotenv import load_dotenv
from tools.sql import run_query_tool, describe_table_tool, list_tables
from tools.report import write_report_tool
from handlers.chat_model_start_handler import ChatModelStartHandler

load_dotenv()

handler = ChatModelStartHandler()
llm = ChatOpenAI(
    model="gpt-4.1-nano",
    callbacks=[handler,]
)

prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template((
            'You are an AI that has access to a SQLite database, with the following tables: '
            f'{", ".join(list_tables())}. '
            'Do not make assumptions about what tables are in the database, or what columns there are in each table. '
            'Instead, use the "describe_table" tool to query the exact schema of each table.'
        )),
        MessagesPlaceholder(variable_name='history'),
        HumanMessagePromptTemplate.from_template('{input}'),
        MessagesPlaceholder(variable_name='agent_scratchpad'),

    ]
)

tools=[run_query_tool, describe_table_tool, write_report_tool]

agent = create_openai_functions_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(
    agent=agent,
    verbose=True,
    tools=tools,
)

memory_store = InMemoryChatMessageHistory()
chat_chain = RunnableWithMessageHistory(
    agent_executor,
    lambda session: memory_store,
    input_messages_key="input",
    history_messages_key="history",
)
cfg = {"configurable": {"session_id": "user_1"}} 
# agent_executor('How many users are in the database?')
# response = chat_chain.invoke(
#     {'input': 'How many users did not provide a shipping address?'},
#     config=cfg)
# response = agent_executor.invoke({'input': 'Give me the addresses of users who placed more than five orders?'})
# response = agent_executor.invoke({'input': 'How many users placed more than five orders? Write a report of the addresses of those users.'})
# response = agent_executor.invoke({'input': 'Do we have more users from Oregon or Minnesota?'})
response = chat_chain.invoke(
    {'input': 'How many orders are there? Write an html report to local disk.'},
    config=cfg)
# print(response['output'])

# response = chat_chain.invoke(
#     {'input': 'Repeat the exact same process for users.'},
#     config=cfg)
print(response['output'])