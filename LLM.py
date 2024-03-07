import os
import getpass

os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter Your OpenAI API Key:")
# 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
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from langchain.agents import load_huggingface_tool
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.tools import DuckDuckGoSearchRun
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_community.tools.google_search.tool import (
    GoogleSearchResults,
    GoogleSearchRun,
)
from langchain.agents import AgentType, load_tools

# os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass("HF Token:")

llm = OpenAI(temperature=0.1)
# search = DuckDuckGoSearchRun()
# search_tool = tool(
#     name="search\_tool",
#     description="A search tool used to query DuckDuckGo for search results when trying to find information from the internet.",
#     func=search.run,
# )
llm = HuggingFaceHub(
    repo_id="huggingfaceh4/zephyr-7b-alpha",
    model_kwargs={"temperature": 0.5, "max_length": 512, "max_new_tokens": 512},
)

query = """
Give a pubmed query that finds articles with "radiomic*", "papillary thyroid cancer" and "lymph node metastasis" and the necessary MeSH terms.

"""

prompt = f"""
 <|system|>
You are an AI assistant that follows instruction extremely well.
Please be truthful and give direct answers
</s>
 <|user|>
 {query}
 </s>
 <|assistant|>
"""

response = llm.predict(prompt)
print(response)

llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)


# tools = load_tools(["ddg-search"], llm=llm)
# tools += [tool]
agent = initialize_agent(tools, llm, verbose=True)

agent.run("What is LangChain and how does it work? ")


# Creating a chain

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

hf = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 10},
)
from langchain.prompts import PromptTemplate

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)

chain = prompt | hf

question = "What is LangChain and how does it work?"

print(chain.invoke({"question": question}))

# Teste 2

repo_id = "bigscience/bloom"
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory

conversation = ConversationChain(
    llm=llm, verbose=True, memory=ConversationBufferMemory()
)

conversation.predict(
    input="Give a pubmed query that finds articles with 'radiomic*', 'papillary thyroid cancer' and 'lymph node metastasis' and the necessary MeSH terms."
)
