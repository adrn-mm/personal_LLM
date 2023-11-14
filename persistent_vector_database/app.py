from dotenv import load_dotenv
from langchain.llms import HuggingFaceHub
from langchain.chat_models import ChatOpenAI
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
import qdrant_client

def get_vector_store():
    client = qdrant_client.QdrantClient(
        os.getenv('QDRANT_HOST'),
        api_key=os.getenv('QDRANT_API_KEY')
    )
    embeddings = OpenAIEmbeddings()
    vector_store = Qdrant(
        client=client,
        collection_name="pdfs", 
        embeddings=embeddings,
    )
    return vector_store

def main():
    load_dotenv()

    st.set_page_config(page_title="Ask Qdrant")
    st.header("Ask your remote database 🪣")

    vector_store = get_vector_store()

    # create chain 
    qa = RetrievalQA.from_chain_type(
        llm = ChatOpenAI(),
        # llm=HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512}),
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
    )

    # show user input
    user_question = st.text_input("Ask your question about your database:")
    if user_question:
        st.write("Questions:", user_question)
        answer = qa.run(user_question)
        st.write("Answer:",answer)


if __name__ == "__main__":
    main()