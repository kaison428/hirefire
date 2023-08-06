import os
import openai

# parser
from docparser import *
import PyPDF2

# LangChain
from langchain.chat_models import ChatOpenAI

from langchain.docstore.document import Document

from langchain.chains import RetrievalQA, LLMChain, SequentialChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.utilities.zapier import ZapierNLAWrapper
from langchain.agents import AgentType
from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory, SimpleMemory
from langchain.agents import initialize_agent, Tool

from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain import LLMMathChain, SerpAPIWrapper

from langchain.llms import OpenAI


# Scheme imports ------------------------------
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.prompts import PromptTemplate

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
def get_agent(resumes, use_zapier=False):

    # 1. construct database
    resume_database, raw_resumes, retrieval_chains = get_database_from_resume(resumes, method='non-retrieval', summarize=True)
    # resume_database = get_complete_database(resume_database, raw_resumes)

    resume_database_text = get_text_from_json(resume_database)
    resume_database_df = get_df_from_json(resume_database)

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
    retrieval_chain = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
        )
    
    tools = [
        Tool(
            name='resume_database',
            func=retrieval_chain.run,
            description=f"useful for when you need to answer questions about any person or candidates.",
        ),
    ]

    if use_zapier:
        zapier = ZapierNLAWrapper()
        zapier_toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier).get_tools()
        tools += zapier_toolkit

    # 3. set up agent --------------------------------
    llm = ChatOpenAI(temperature=0, model='gpt-4')
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

    # 4. set up a chain for email idenfication -----------------------
    # chain 1 --------------------------------
    llm = ChatOpenAI(temperature=0)
    template_string = '''
        Identify if the prompt is about drafting, sending email and/or creating calendar. Output 'Y' if it does and 'N' if it does not.
        
        Prompt: ```{prompt}```
    '''
    prompt_template = PromptTemplate(
        input_variables=["prompt"],
        template=template_string,
    )
    email_id_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="is_email")

    # chain 2 --------------------------------
    llm = ChatOpenAI(temperature=0)
    template_string = '''
        is_email: {is_email}
        Prompt: ```{prompt}```

        If is_email is N, output the Prompt immediately and skip the following instructions.

        If is_email is Y, check if the Prompt contains any email address. 
        If the Prompt contains email addresses, output the Prompt immediately.
        If the Prompt does not contain email addresses, output the Final Message delimited by triple backticks.

        Final Message: ```Repeat this message in your output: Please enter at least one email address.```
        
    '''
    prompt_template = PromptTemplate(
        input_variables=["prompt", "is_email"],
        template=template_string,
    )
    address_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="answer")

    # final chain --------------------------------
    overall_chain = SequentialChain(
        memory=SimpleMemory(),
        chains=[email_id_chain, address_chain],
        input_variables=["prompt"],
        output_variables=["answer"],
        verbose=True)
    
    return overall_chain, agent, resume_database_df
