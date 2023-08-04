import os
import openai

# parser
from docparser import *
import PyPDF2

# LangChain
from langchain.chat_models import ChatOpenAI

from langchain.docstore.document import Document

from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.utilities.zapier import ZapierNLAWrapper
from langchain.agents import AgentType
from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory
from langchain.agents import initialize_agent, Tool

from langchain import LLMMathChain, SerpAPIWrapper

from langchain.llms import OpenAI


# Scheme imports ------------------------------
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

# Utilities ------------------------------
def get_text_from_pdf(fileobj):
    #create reader variable that will read the pdffileobj
    reader = PyPDF2.PdfReader(fileobj)
    
    #This will store the number of pages of this pdf file
    num_pages = len(reader.pages)
    
    combined_text = ''
    for i in range(num_pages):
        #create a variable that will select the selected number of pages
        pageobj = reader.pages[i]
        text = pageobj.extract_text()
        combined_text += text
        combined_text += '\n'

    return combined_text

# LangChain --------------------------------
def get_agent(resumes):

    # 1. construct database
    resume_database, raw_resumes, retrieval_chains = get_database_from_resume(resumes, method='not_retrieval')
    # resume_database = get_complete_database(resume_database, raw_resumes)

    resume_database_text = get_text_from_json(resume_database)

    # ----------------------------------------------------------------
    # save it for inspection
    fname = 'output.txt'
    with open(fname, 'w') as f:
        f.write(resume_database_text)

    fname = 'output.txt'
    with open(fname, 'r') as f:
        resume_database_text = f.read()
    # ----------------------------------------------------------------

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=0)
    splits = text_splitter.split_text(resume_database_text)

    docs = [Document(page_content=t) for t in splits[:]]

    # get vector database
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings()
    vectorstore = Chroma.from_documents(docs, embeddings, collection_name="resume_database")

    # 2. create toolsets ----------------------------------------------------------------

    llm = ChatOpenAI(temperature=0)

    llm_math_chain = LLMMathChain.from_llm(llm=llm)
    retrieval_chain = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
        )
    zapier = ZapierNLAWrapper()
    zapier_toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier).get_tools()

    tools = [
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math and numerical problems.",
        ),
        Tool(
            name='resume_database',
            func=retrieval_chain.run,
            description=f"useful for when you need to answer questions about any person or candidates.",
        ),
    ]

    tools += zapier_toolkit

    # 3. set up agent --------------------------------
    llm = OpenAI(temperature=0)    
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history", 
        k=2)
    
    system_message = SystemMessage(
        content="""
        Always respond using the tools. Say you do not know if you do not know the answer.
        """
        )

    agent = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, 
        verbose=True, 
        memory=memory,
        agent_kwargs={"system_message": system_message.content},
    )

    return agent

    # 3. set up agent --------------------------------
    llm = OpenAI(temperature=0)    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    agent = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, 
        verbose=True, 
        memory=memory,
    )

    return agent

    # 3. set up agent --------------------------------
    llm = ChatOpenAI(temperature=0)
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history", 
        k=2,
        return_messages=True)

    agent = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, 
        verbose=True, 
        handle_parsing_errors=True,
        early_stopping_methods='generate', 
        memory=memory,
        max_iterations=4,
    )

    return agent

