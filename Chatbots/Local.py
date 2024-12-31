from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv
import time

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user's queries."),
        ("user", "Question: {question}")
    ]
)

llm = Ollama(model="llama3.2:1b")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

st.title('Chat Bot : llama3.2:1b')

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

for user_input, bot_response in st.session_state['chat_history']:
    st.write(f"**User**: {user_input}")
    st.write(f"**Bot**: {bot_response}")

input_text = st.text_input("Ask a question:")

if input_text:
    st.session_state['chat_history'].append((input_text, None))

    thinking_placeholder = st.empty()
    thinking_placeholder.text("Bot : Thinking...")

    with st.spinner("Generating response..."):
        try:
            bot_response = chain.invoke({"question": input_text})

            if isinstance(bot_response, dict):
                bot_response = bot_response.get('text', 'Sorry, something went wrong.')
            
            bot_response = output_parser.parse(bot_response) if isinstance(bot_response, str) else str(bot_response)

        except Exception as e:
            st.error(f"Error: {str(e)}")
            bot_response = "Sorry, something went wrong."

    thinking_placeholder.empty()

    st.session_state['chat_history'][-1] = (input_text, bot_response)

    st.write(f"**Bot**: {bot_response}")
