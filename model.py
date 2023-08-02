import os
import openai

# parser
import PyPDF2

# LangChain
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

from langchain.docstore.document import Document

from langchain.chains.summarize import load_summarize_chain

from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader

from langchain.chains.question_answering import load_qa_chain


# Scheme imports ------------------------------
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

