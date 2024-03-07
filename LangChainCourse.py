from langchain.llms import OpenAI

import os
import getpass

os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter Your OpenAI API Key:")
llm = OpenAI(model="davinci-002", temperature=0.9)

text = "Suggest a personalized workout routine for someone looking to improve cardiovascular endurance and prefers outdoor activities."
print(llm(text))
