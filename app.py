import streamlit as st
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain import PromptTemplate, LLMChain

#Function to return the response
def load_answer(question, huggingfacehub_api_token):
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
    llm = HuggingFaceEndpoint(repo_id=repo_id,
                            #max_length=128,
                            temperature=0.1,
                            token=huggingfacehub_api_token)
    answer = llm.invoke(question)
    return answer


#App UI starts here
st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
st.header("Simple Question Answering App")

#Gets the user input
def get_text():
    input_text = st.text_input("You: ", key="input")
    return input_text

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
huggingfacehub_api_token = os.getenv("huggingfacehub_api_token")

user_question = get_text()

submit = st.button('Generate')  

#If generate button is clicked
if submit:
    response = load_answer(user_question, huggingfacehub_api_token)    

    st.subheader("Answer:")

    st.write(response)