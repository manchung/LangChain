from langchain_openai import ChatOpenAI
# from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
import argparse

load_dotenv()
argparser = argparse.ArgumentParser(description="LangChain CLI")
argparser.add_argument(
    "--language",
    type=str,
    default="python",
    help="Programming language to use for the program",
)
argparser.add_argument(
    "--task",
    type=str,
    default="calculate the sum of two numbers",
    help="Task to perform",
)
args = argparser.parse_args()
llm = ChatOpenAI(
    model="gpt-4.1-nano",
)

code_prompt = PromptTemplate(
    template="Write a very short {language} function that will {task}. Skip all explanation and comments. Just output the code.",
    input_variables=["language", "task"],
)
code_chain = code_prompt | llm

test_prompt = PromptTemplate(
    template="Write a test for the following {language} code. Skip all explanation and comments. Just output the test:\n {code}",
    input_variables=["language", "code"],
)
test_chain = test_prompt | llm

def wrap_code(chain):
    return RunnableLambda(
        lambda d: {**d, "code": chain.invoke(d).content}
    )

def wrap_test(chain):
    return RunnableLambda(
        lambda d: {**d, "test": chain.invoke(d).content}
    )

full_chain = wrap_code(code_chain) | wrap_test(test_chain)
response = full_chain.invoke(
    {
        "language": args.language,
        "task": args.task,
    }
)
# response = llm.invoke("Write a very very short poem about this girl named Zoie")
print(response)
print(f'GENERATED CODE>>>>>\n\n{response["code"]}\n\n\nGENERATED TEST>>>>>\n\n{response["test"]}')

