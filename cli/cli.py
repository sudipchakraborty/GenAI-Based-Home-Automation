from contentLoader import load_documents
from utils import print_docs_pretty
from textSplitter import split_text_character
##########################################
def run_cli():
    print("CLI started. Type 'exit' to quit.")
    docs = load_documents("Content/Attendance.txt", source_type="text")
    print_docs_pretty(docs)  # line-by-line output with source header
    chunks = split_text_character(docs, chunk_size=10, chunk_overlap=2)
    print(chunks)

    while True:
        user_input = input(">>> ")  # Read string from terminal

        if user_input.lower() == "exit":
            print("Exiting program... Goodbye!")
            break
        else:
            print(f"You entered: {user_input}")
###########################################
