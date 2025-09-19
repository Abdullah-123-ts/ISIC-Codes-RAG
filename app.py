# rag_app.py
import os
import pandas as pd
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()
# ---------------- CONFIG ----------------
OPENAI_api_key = os.getenv("OPENAI_API_KEY")
print(OPENAI_api_key)
VECTOR_STORE_PATH = "isic_faiss_store"

# List of Excel sheets in search order
SHEETS_ORDER = ["dafza.xlsx", "meydan.xlsx", "spc.xlsx", "ISIC.xlsx"]

# ---------------- LOAD SHEETS ----------------
@st.cache_data
def load_sheets():
    dataframes = {}
    for sheet in SHEETS_ORDER:
        df = pd.read_excel(sheet)
        dataframes[sheet] = df
    return dataframes

# ---------------- VECTOR STORE ----------------
@st.cache_resource
def load_or_create_vectorstore(isic_df):
    if os.path.exists(VECTOR_STORE_PATH):
        vectorstore = FAISS.load_local(
            VECTOR_STORE_PATH,
            OpenAIEmbeddings(openai_api_key=OPENAI_api_key),
            allow_dangerous_deserialization=True
        )
    else:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_api_key)
        vectorstore = FAISS.from_pandas(
            isic_df,
            embeddings,
            text_column="Activity name"
        )
        vectorstore.save_local(VECTOR_STORE_PATH)
    return vectorstore

# ---------------- SEARCH LOGIC ----------------
def search_activity(query, dataframes, vectorstore):
    # Step 1: Exact match search (excluding last sheet)
    for sheet in SHEETS_ORDER[:-1]:
        df = dataframes[sheet]
        match = df[df["Activity name"].str.lower() == query.lower()]
        if not match.empty:
            row = match.iloc[0]
            return {
                "source": sheet,
                "activity": row["Activity name"],
                "code": row.get("Class"),
                "method": "Exact Match"
            }

    # Step 2: Vector similarity search on ISIC
    isic_df = dataframes["ISIC.xlsx"]
    docs = vectorstore.similarity_search(query, k=5)

    candidates = [
        {"activity": d.page_content, "code": d.metadata.get("Code") or d.metadata.get("Class")}
        for d in docs
    ]

    # Use GPT to refine choice
    llm = ChatOpenAI(model="gpt-5-mini", openai_api_key=openai_api_key)
    prompt = f"""
    The user is searching for an activity: "{query}".
    Here are the candidate ISIC activities with their codes:

    {candidates}

    Please choose the single most appropriate code based on business context.
    Respond in JSON as: {{"activity": "...", "code": "..."}}.
    """
    response = llm.invoke(prompt)
    return {"source": "ISIC.xlsx", "method": "Vector + GPT", **response.dict()}

# ---------------- STREAMLIT APP ----------------
def main():
    st.title("Activity Code Finder (RAG)")
    st.write("Search for activity codes across multiple Excel sheets with exact + vector search fallback.")

    dataframes = load_sheets()
    vectorstore = load_or_create_vectorstore(dataframes["ISIC.xlsx"])
    query = st.text_input("Enter activity name/description:")

    if st.button("Search") and query:
        result = search_activity(query, dataframes, vectorstore)

        st.subheader("Result")
        st.json(result)


if __name__ == "__main__":
    main()
