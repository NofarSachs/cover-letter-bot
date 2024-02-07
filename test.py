# Import os to set API key
import os
# Import OpenAI as main LLM service
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
# Bring in streamlit for UI/app interface
import streamlit as st

# Import PDF document loaders...there's other ones as well!
from langchain.document_loaders import PyPDFLoader
# Import chroma as the vector store
from langchain.vectorstores import Chroma
import chromadb

# Import vector store stuff
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

# Set APIkey for OpenAI Service
# Can sub this out for other LLM providers
os.environ['OPENAI_API_KEY'] = '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'

# Create instance of OpenAI LLM
llm = OpenAI(temperature=0.9, verbose=True)
embeddings = OpenAIEmbeddings()

# Create a prompt
cover_letter_template = PromptTemplate(
    input_variables=['job_description', 'professional_background'],
    template='Generate a short cover letter based on the following professional background: {professional_background} '
             'and job_description: {job_description} '
)
# Create and load PDF Loader
loader = PyPDFLoader('professional_background.pdf')
# Split pages from pdf
pages = loader.load_and_split()

# Load documents into vector database aka ChromaDB
persistent_client = chromadb.PersistentClient()
collection = persistent_client.get_or_create_collection("collection_name")

store = Chroma.from_documents(pages, embeddings, client=persistent_client, collection_name='my_collection')

# Create vectorstore info object - metadata repo?
vectorstore_info = VectorStoreInfo(
    name="professional_background",
    description="a file listing the professional background as a pdf",
    vectorstore=store
)
# Convert the document store into a langchain toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

# Add the toolkit to an end-to-end LC
# agent_executor = create_vectorstore_agent(
#     llm=llm,
#     toolkit=toolkit,
#     verbose=True
# )
# Create a chain
cover_letter_chain = LLMChain(llm=llm, prompt=cover_letter_template, verbose=True, output_key='cover_letter')
st.title('👩🏻‍💻 🧐 Why am I a good fit for your company 🤓 🧠')
# Create a text input box for the user
prompt = st.text_input('Insert Job description here')

# If the user hits enter
if prompt:
    # Then pass the prompt to the LLM
    cover_letter = cover_letter_chain.run(job_description=prompt, professional_background=toolkit)
    # ...and write it out to the screen
    st.write(cover_letter)

