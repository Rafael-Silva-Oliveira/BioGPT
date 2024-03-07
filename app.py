from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import streamlit as st
import PyPDF2

from transformers import set_seed
from transformers import BioGptTokenizer, BioGptForCausalLM
import os
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from dotenv import load_dotenv

load_dotenv()


class BioinformaticsAgent:
    def __init__(self, model, task):
        self.system_prompt = self.get_system_prompt()
        self.human_prompt = HumanMessagePromptTemplate.from_template("{question}")

        complete_prompt = ChatPromptTemplate.from_messages(
            [self.system_prompt, self.human_prompt]
        )
        self.model = model
        self.task = task

        llm = self._load_model(model, task)
        self.chat = llm
        self.chain = LLMChain(llm=self.chat, prompt=complete_prompt)

    def get_system_prompt(self):
        system_prompt = """
		You are an expert in summarizing data. Please be truthful and give direct answers.\n 

		Task: {task}

        Context: {context}

		
		"""
        return SystemMessagePromptTemplate.from_template(system_prompt)

    def run_chain(self, task, context, question):

        return self.chain.run(task=task, context=context, question=question)

    def _load_model(self, model_name, task):
        with st.status(f"Downloading model {model}..."):
            st.write("Please wait...")

            if model_name == "microsoft/biogpt":

                # model = BioGptForCausalLM.from_pretrained(model_name)
                # tokenizer = BioGptTokenizer.from_pretrained(model_name)
                # llm = pipeline(task, model=model, tokenizer=tokenizer)

                llm = HuggingFaceHub(
                    repo_id=model_name, model_kwargs={"temperature": 1e-10}
                )

                set_seed(42)
                return llm
            if model_name == "facebook/bart-large-cnn":
                llm = HuggingFaceHub(
                    repo_id=model_name, model_kwargs={"temperature": 1e-10}
                )

                set_seed(42)
                return llm
            status.update(label="Download complete!", state="complete", expanded=False)


# create a streamlit app
st.title("BioGPT from documents.")


def retrieve_pdf_text(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# create a upload file widget for a pdf
pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# if a pdf file is uploaded
if pdf_file:
    # retrieve the text from the pdf
    if "context" not in st.session_state:
        st.session_state.context = retrieve_pdf_text(pdf_file)

# create a button that clears the context
if st.button("Clear context"):
    st.session_state.__delitem__("context")
    st.session_state.__delitem__("response")
# create a selectbox widget for the model in the sidebar
model = st.sidebar.selectbox(
    "Select a model", ["microsoft/biogpt", "facebook/bart-large-cnn"]
)

# create a selectbox widget for the task in the sidebar
task = st.sidebar.selectbox("Select a task", ["text-generation", "question-answering"])
# if there's context, proceed
if "context" in st.session_state:

    # create a text input widget for a question
    question = st.chat_input("Write the prompt")
    if question:
        st.write(f"{question}")
    # create a button to run the model
    if st.button("Run"):
        # run the model
        st.session_state.BioinformaticsAgent = BioinformaticsAgent(
            model=model, task=task
        )
        response = st.session_state.BioinformaticsAgent.run_chain(
            task=task, context=st.session_state.context, question=question
        )

        if "response" not in st.session_state:
            st.session_state.response = response

        else:
            st.session_state.response = response

# display the response
if "response" in st.session_state:
    with st.chat_message("assistant"):
        st.markdown(response)
else:
    st.warning("Choose a model and a prompt first", icon="⚠️")
