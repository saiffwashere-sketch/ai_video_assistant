import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b", 
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3
    )


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )
    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    llm = get_llm()

    map_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize this portion of a meeting transcript concisely."),
        ("human", "{text}"),
    ])
    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)
    print(f"Summarizing {len(chunks)} chunk(s)...")

    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"  Summarizing chunk {i + 1}/{len(chunks)}...")
        chunk_summaries.append(map_chain.invoke({"text": chunk}))
        if i < len(chunks) - 1:
            time.sleep(2)  # 2s pause between chunks to avoid 429

    combined = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert meeting summarizer. Combine these partial summaries "
            "into one final professional meeting summary in bullet points.",
        ),
        ("human", "{text}"),
    ])
    combined_chain = combined_prompt | llm | StrOutputParser()

    time.sleep(2)  # pause before final combine call too
    return combined_chain.invoke({"text": combined})


def generate_title(transcript: str) -> str:
    llm = get_llm()

    title_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Based on the meeting transcript, generate a short professional meeting title "
            "(max 8 words). Only return the title, nothing else.",
        ),
        ("human", "{text}"),
    ])
    title_chain = title_prompt | llm | StrOutputParser()

    time.sleep(4)  # pause before title call
    return title_chain.invoke({"text": transcript[:2000]})


if __name__ == "__main__":
    print(summarize("This is a test transcript. The meeting discussed project updates and team performance."))


