import streamlit as st
from langchain import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from io import StringIO

# LLM and key loading function
def load_LLM(openai_api_key):
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    return llm

# Function to get OpenAI API Key
def get_openai_api_key():
    input_text = st.text_input(label="OpenAI API Key", placeholder="Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input", type="password")
    return input_text

# Function to summarize text
def summarize_text(file_input, openai_api_key):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n"], 
            chunk_size=2000,  # Reduced chunk size
            chunk_overlap=200  # Adjusted overlap
        )
        splitted_documents = text_splitter.create_documents([file_input])
        llm = load_LLM(openai_api_key=openai_api_key)

        summarize_chain = load_summarize_chain(
            llm=llm, 
            chain_type="map_reduce"
        )

        summary_output = summarize_chain.run(splitted_documents)
        return summary_output
    except Exception as e:
        st.error(f"An error occurred during summarization: {e}")
        return None

# Page title and header
st.set_page_config(page_title="AI Long Text Summarizer")
st.header("AI Long Text Summarizer")
st.markdown("The free version of ChatGPT cannot summarize long texts. Now you can do it with this app."
            "Just enter your openAI API key and upload the .txt file")

# Input OpenAI API Key
st.markdown("## Enter Your OpenAI API Key")
openai_api_key = get_openai_api_key()

# Input file upload
st.markdown("## Upload the text file you want to summarize")
uploaded_file = st.file_uploader("Choose a file", type="txt")

# Check if both API key and file are provided
inputs_ready = openai_api_key and uploaded_file

# Add a button to trigger summarization
if inputs_ready:
    if st.button("Summarize"):
        file_input = StringIO(uploaded_file.getvalue().decode("utf-8")).read()

        if len(file_input.split(" ")) > 20000:
            st.write("Please enter a shorter file. The maximum length is 20000 words.")
        else:
            with st.spinner('Summarizing...'):
                summary_output = summarize_text(file_input, openai_api_key)
                if summary_output:
                    st.write(summary_output)
else:
    st.button("Summarize", disabled=True)
