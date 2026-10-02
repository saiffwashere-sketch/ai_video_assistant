# actionable items, questions, decision points
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b", 
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3
    )


def build_chain(system_prompt: str):
    llm = get_llm()
    # FIX: chain now accepts a plain string directly via the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}"),
    ])
    return prompt | llm | StrOutputParser()


def extract_action_items(transcript: str) -> str:
    system_prompt = (
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'"
    )
    chain = build_chain(system_prompt)
    # FIX: pass dict with "text" key directly to the prompt template
    return chain.invoke({"text": transcript})


def extract_key_decisions(transcript: str) -> str:
    system_prompt = (
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all decision points. For each provide:\n"
        "- Decision description\n"
        "- Who made the decision (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No decision points found.'"
    )
    chain = build_chain(system_prompt)
    return chain.invoke({"text": transcript})


def extract_questions(transcript: str) -> str:
    system_prompt = (
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all questions asked. For each provide:\n"
        "- Question text\n"
        "- Who asked the question (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No questions found.'"
    )
    chain = build_chain(system_prompt)
    return chain.invoke({"text": transcript})


# FIX: removed the bare print() call that was firing on every import
# If you want to test this file standalone, use:
# if __name__ == "__main__":
#     print(extract_action_items("Test transcript."))
