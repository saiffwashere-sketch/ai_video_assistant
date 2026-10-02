import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda, RunnableSequence, RunnableMap,RunnablePassthrough
from core.vector_store import build_vector_store,load_vector_store, get_retriever

def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b", 
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3
    )

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def build_rag_chain(transcript: str):
    vector_store = build_vector_store(transcript)
    retriever = get_retriever(vector_store , k=4)

    llm = get_llm()

    prompt= ChatPromptTemplate.from_messages(
        [
            ("system", 
            """You are an expert assistant that answers questions based on the provided meeting transcript.
    You are an expert  assistant that answers questions based on the provided meeting transcript.
    If the answer is not present in the transcript, respond with "I don't know".

    always answer in a concise and clear manner.

    Context from the meeting transcript is provided below. Use this context to answer the question.
    {context}""",
    ),
    ("human","{question}"),
        ]
    )

    #full LCEL rag pipeline

    rag_chain = (
        {"context": retriever | RunnableLambda(format_docs),
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

def load_rag_chain():
    vector_store = load_vector_store()
    retriever = get_retriever(vector_store , k=4)

    llm = get_llm()

    prompt= ChatPromptTemplate.from_messages(
        [
            ("system", 
            """You are an expert assistant that answers questions based on the provided meeting transcript.
    You are an expert  assistant that answers questions based on the provided meeting transcript.
    If the answer is not present in the transcript, respond with "I don't know".

    always answer in a concise and clear manner.

    Context from the meeting transcript is provided below. Use this context to answer the question.
    {context}""",
    ),
    ("human","{question}"),
        ]
    )

    #full LCEL rag pipeline

    rag_chain = (
        {"context": retriever | RunnableLambda(format_docs),
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain  

def ask_question(rag_chain, question: str)-> str:
    print(f"Question: {question}")
    answer = rag_chain.invoke(question)
    print(f"Answer: {answer}")
    return answer