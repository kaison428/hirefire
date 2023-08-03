from os import listdir
from os.path import isfile, join

from data import *

import ast
import copy

from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.agents import Tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document

def get_database_from_resume(resumes):

    # get template ------------------------------
    prompt_template = ChatPromptTemplate.from_template(TEMPLATE_STRING)

    # define llm --------------------------------
    llm = ChatOpenAI(temperature=0)

    # resume database
    resume_database = {}
    raw_resumes = {}

    for i, resume in enumerate(resumes):

        # split text --------------------------------
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=0)
        splits = text_splitter.split_text(resume)
        splits = [Document(page_content=t) for t in splits[:]]

        # create vectorstore for retrieval --------------------------------
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(splits, embeddings, collection_name=f'candiate{i}')

        retrieval_chain = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
        )

        # create vectorstore for retrieval --------------------------------
        question = 'Provide the full name of the person in the document in this format:{Full Name}'
        candidate = retrieval_chain.run(question)

        resume_data = parse_resume(llm, resume, RESUME_SAMPLE, QUESTION_SCHEMA, ANSWER_DATA, prompt_template)

        # save data ----------------------------------------------------------------
        resume_database[candidate] = resume_data
        raw_resumes[candidate] = resume

    return resume_database, raw_resumes

def get_complete_database(resume_database, raw_resumes):

    # get template ------------------------------
    prompt_template = ChatPromptTemplate.from_template(TEMPLATE_STRING)

    complete_resume_database = copy.deepcopy(resume_database)
    section_queries = {
                    'Work Experience': {
                        'schema': WORK_EXPERIENCE_SCHEMA,
                        'answer': WORK_EXPERIENCE_ANSWER,
                    }, 
                    'Projects': {
                        'schema': PROJECT_SCHEMA,
                        'answer': PROJECT_ANSWER,
                    },
                }

    for app_name in resume_database:
        resume = raw_resumes[app_name]
        data_obj= resume_database[app_name]

        for target_key in section_queries:
            schema = section_queries[target_key]['schema']
            answer = section_queries[target_key]['answer']
            
            section_data = get_item_info(target_key, data_obj, resume, RESUME_SAMPLE, schema, answer, prompt_template)

            complete_resume_database[app_name][target_key] = section_data

    return complete_resume_database

def parse_resume(llm, resume, resume_sample, question_schema, answer_data, prompt_template):
    parsed_resume = {}
    for key in question_schema:
        question = question_schema[key]
        answer_sample = answer_data[key]
        
        query = prompt_template.format_messages(
                    resume_sample=resume_sample,
                    question=question,
                    answer_sample=answer_sample,
                    resume=resume)
        
        data = llm(query).content
        parsed_resume[key] = data

    return parsed_resume

def parse_resume_from_retrieval(retrieval_chain, question_schema, prompt_template):
    parsed_resume = {}
    for key in question_schema:
        question = question_schema[key]
        
        query = prompt_template.format_messages(
                    question=question,
                )
        
        data = retrieval_chain.run(query)
        parsed_resume[key] = data

    return parsed_resume

def get_item_info(target_key, data_obj, resume, resume_sample, question_schema, answer_data, prompt_template):

    # define llm --------------------------------
    llm = ChatOpenAI(temperature=0)
    
    items = data_obj[target_key]
    parsed_items = [i.strip() for i in ast.literal_eval(items)]
    
    parsed_section = {}

    for item in parsed_items:
        parsed_item = {}

        for key in question_schema:
            question_template = ChatPromptTemplate.from_template(question_schema[key])
            answer_sample = answer_data[key]

            question = question_template.format_messages(
                        entity=item)[0].content
            
            query = prompt_template.format_messages(
                        resume_sample=resume_sample,
                        question=question,
                        answer_sample=answer_sample,
                        resume=resume)
            
            data = llm(query).content
            parsed_item[key] = data
        
        parsed_section[item] = parsed_item

    return parsed_section


def get_text_from_json(database):

    text = ''

    text += 'List of candidates:'
    for person in database:
        text += f'{person},'
    text += '\n\n'

    for person in database:
        for key, info in database[person].items():
            if type(info) == str:
                text += f"{person}'s {key}: {info}"
                text += '\n'
            else:
                for entity, entity_info in info.items():
                    for term, detail in entity_info.items():
                        text += f"{person}'s {key} at {entity}: {term}: {detail}"
                        text += '\n'
        text += '\n'

    return text