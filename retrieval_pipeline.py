from dotenv import load_dotenv
# Vector database
from langchain_chroma import Chroma
# Embedding model
from langchain_huggingface import HuggingFaceEmbeddings # vector embedding ke liye open ai wala paid tha 
# LLM
from langchain_ollama import ChatOllama # local LLM ke liye, jo humne Ollama se setup kiya tha
# Chat message types
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage
)
load_dotenv()
# Text -> Vector conversion

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
#  vector database
db = Chroma(
    persist_directory="db/chroma_db",
    embedding_function=embedding_model
)

# local LLM

model = ChatOllama(model="llama3.2:latest")
# Store previous conversation
chat_history = []


#  ask question

def ask_question(user_question):
    print(f"\nYou Asked: {user_question}")

   
# Convert follow-up question
    if len(chat_history) > 0:
        rewrite_messages = [
            SystemMessage(
                content="""
                Rewrite the user's question so that it becomes
                a complete standalone searchable question.
                Only return the rewritten question.
                """
            )
        ]

        # Add previous conversation
        rewrite_messages += chat_history
        
        # Add current question
        rewrite_messages.append(
            HumanMessage(content=user_question)
        )

        # LLM rewrites question
        rewritten_result = model.invoke(rewrite_messages)

        search_question = rewritten_result.content

        print(f"Search Question: {search_question}")

    else:
        search_question = user_question
    #  Retrieve relevant documents
    retriever = db.as_retriever(
        search_kwargs={"k": 3}
    )
    docs = retriever.invoke(search_question)

    print(f"\nRetrieved {len(docs)} documents\n")

    # Print small preview
    # for i, doc in enumerate(docs, start=1):

    #     preview = doc.page_content[:120]

    #     print(f"Document {i}:")
    #     print(preview)
    #     print()

    # --------------------------------------------------
    # --------------------------------------------------

    documents_text = ""

    for doc in docs:
        documents_text += doc.page_content
        documents_text += "\n\n"

    # Creating   the final prompt
    final_prompt = f"""
Answer the question using ONLY the documents below.

Question:
{user_question}

Documents:
{documents_text}

If the answer is not present in the documents,
say you don't have enough information.
"""

   
    # Now we have given the data to the llm and now asking for the response from the llm

    final_messages = [
        SystemMessage(
            content="""
            You are a helpful AI assistant.
            Answer only from provided documents.
            """
        )
    ]

    # Add previous conversation
    final_messages += chat_history

    # Add final prompt
    final_messages.append(
        HumanMessage(content=final_prompt)
    )

    # Generate answer
    result = model.invoke(final_messages)

    answer = result.content
    #  Save conversation
    chat_history.append(
        HumanMessage(content=user_question)
    )

    chat_history.append(
        AIMessage(content=answer)
    )

    # Print answer
    print("\nAnswer:")
    print(answer)

    return answer
# Chat loop

def start_chat():

    print("RAG Chat Started")
    print("Type 'quit' to exit\n")

    while True:

        user_question = input("Your Question: ")

        if user_question.lower() == "quit":
            print("Goodbye!")
            break

        ask_question(user_question)


# ==================================================
# Run program
# ==================================================
if __name__ == "__main__":
    start_chat()