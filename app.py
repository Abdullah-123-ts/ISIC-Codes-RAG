# rag_app.py
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# ---------------- CONFIG ----------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VECTOR_STORE_PATH = "isic_faiss_store"

# List of Excel sheets in search order
SHEETS_ORDER = ["dafza.xlsx", "meydan.xlsx", "spc.xlsx", "isic.xlsx"]

# ---------------- LOAD SHEETS ----------------
@st.cache_data
def load_sheets():
    """Load all Excel sheets into DataFrames with normalized column names."""
    dataframes = {}
    for sheet in SHEETS_ORDER:
        df = pd.read_excel(sheet)
        # Normalize column names
        df.columns = [c.strip().lower() for c in df.columns]
        # Ensure required columns exist
        if "activity name" not in df.columns or "class" not in df.columns:
            raise ValueError(f"Sheet {sheet} must contain 'Activity Name' and 'Class' columns.")
        dataframes[sheet] = df
    return dataframes

# ---------------- VECTOR STORE ----------------
@st.cache_resource
def load_or_create_vectorstore(isic_df: pd.DataFrame):
    """Create or load FAISS vector store for ISIC activities."""
    if os.path.exists(VECTOR_STORE_PATH):
        vectorstore = FAISS.load_local(
            VECTOR_STORE_PATH,
            OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
            allow_dangerous_deserialization=True
        )
    else:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        docs = [
            Document(
                page_content=row["activity name"],
                metadata={"class": row["class"]}
            )
            for _, row in isic_df.iterrows()
        ]
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(VECTOR_STORE_PATH)
    return vectorstore

# ---------------- SEARCH LOGIC ----------------
def search_activity(query, dataframes, vectorstore):
    """Search for activity across sheets using exact match + vector fallback."""
    # Step 1: Exact match in all sheets except ISIC
    for sheet in SHEETS_ORDER[:-1]:
        df = dataframes[sheet]
        match = df[df["activity name"].str.lower() == query.lower()]
        if not match.empty:
            row = match.iloc[0]
            return {
                "source": sheet,
                "activity": row["activity name"],
                "code": row["class"],
                "method": "Exact Match"
            }

    # Step 2: Vector similarity search on ISIC
    docs = vectorstore.similarity_search(query, k=5)
    candidates = [
        {"activity": d.page_content, "code": d.metadata.get("class")}
        for d in docs
    ]


    # Step 3: Use LLM to refine best choice
    llm = ChatOpenAI(model="gpt-5-mini", openai_api_key=OPENAI_API_KEY)
    prompt = f"""
    The user is searching for an activity: "{query}".
    Here are candidate ISIC activities with their codes:
    {candidates}

    Please choose the single most appropriate code based on business context.
    Respond in JSON as: {{"activity": "...", "code": "..."}}.
    """
    response = llm.invoke(prompt)

    return {
        "source": "ISIC.xlsx",
        "method": "Vector + GPT",
        "raw_candidates": candidates,
        "llm_response": response.content
    }

# ---------------- STREAMLIT APP ----------------
def main():
    st.title("Activity Code Finder (RAG)")
    st.write("Search for activity codes across multiple Excel sheets with exact + vector search fallback.")

    dataframes = load_sheets()
    vectorstore = load_or_create_vectorstore(dataframes["isic.xlsx"])
    query = st.text_input("Enter activity name/description:")

    if st.button("Search") and query:
        result = search_activity(query, dataframes, vectorstore)
        st.subheader("Result")
        st.json(result)

if __name__ == "__main__":
    main()
