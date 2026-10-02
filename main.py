from dotenv import load_dotenv
from core.transcriber import transcribe_all
from utils.audio_processor import process_input
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_questions, extract_key_decisions
from core.vector_store import build_vector_store
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()


def run_pipeline(source: str, language: str = "english") -> dict:
    print("Starting AI video assistant...")

    # Step 1: Process the audio file
    chunks = process_input(source)

    # Step 2: Transcribe the audio
    transcript = transcribe_all(chunks, language=language)
    print(f"Raw transcription (first 300 chars): {transcript[:300]}")

    # Step 3: Summarize the transcript
    summary = summarize(transcript)

    # Step 4: Generate a title
    title = generate_title(transcript)

    # Step 5: Extract action items, questions, key decisions
    action_items = extract_action_items(transcript)
    questions = extract_questions(transcript)
    decisions = extract_key_decisions(transcript)

    # Step 6: Build the RAG chain
    rag_chain = build_rag_chain(transcript)

    return {
        "transcript": transcript,
        "summary": summary,
        "title": title,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


if __name__ == "__main__":
    source = input("Enter the path to the audio file or YouTube URL: ")
    language = input("Enter the language (default is 'english'): ") or "english"
    result = run_pipeline(source, language)

    print("\n" + "=" * 40)
    print(f"Title: {result['title']}")
    print(f"\nSummary:\n{result['summary']}")
    print(f"\nAction Items:\n{result['action_items']}")
    print(f"\nKey Decisions:\n{result['key_decisions']}")
    print(f"\nOpen Questions:\n{result['open_questions']}")
    print("=" * 40 + "\n")

    print("You can now ask questions about the transcript. Type 'exit' to quit.")
    rag_chain = result["rag_chain"]
    while True:
        question = input("Enter your question: ").strip()
        if question.lower() == "exit":
            print("Exiting.")
            break
        if not question:
            continue
        # FIX: ask_question expects a plain string, not a dict
        answer = ask_question(rag_chain, question)
        print(f"Answer: {answer}\n")