# =============== CONFIG ==================
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


import os
import re
import pandas as pd
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# ---------------- CONFIG ----------------
VECTOR_STORE_PATHS = {
    "DAFZA.xlsx": "store_dafza",
    "ISIC.xlsx": "store_isic"
}
SEARCH_HIERARCHY = ["DAFZA.xlsx", "ISIC.xlsx"]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
knowledge_base = {}  # {file: {"db": ..., "df": ...}}

# ---------------- HELPERS ----------------
def load_dataframe(file_path):
    if file_path.endswith(".xlsx"):
        try:
            return pd.read_excel(file_path)
        except Exception as e:
            st.error(f"Error loading {file_path}: {e}")
            return None
    return None

def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"[\(\)\-\–]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def build_documents_from_df(df, file_tag):
    docs = []
    for _, row in df.iterrows():
        if file_tag == "DAFZA.xlsx":
            text = f"{row.get('Activity List','')} {row.get('ISIC Description','')}"
            code = str(row.get("Class", ""))
        elif file_tag == "ISIC.xlsx":
            text = str(row.get("Activity name", ""))
            code = str(row.get("Class", ""))
        else:
            text = " ".join([str(v) for v in row.values])
            code = ""
        doc = Document(page_content=normalize_text(text), metadata={"code": code})
        docs.append(doc)
    return docs

# ----------- Vectorstore builder -----------
@st.cache_resource
def create_or_update_vectorstore(file_path, store_path):
    df = load_dataframe(file_path)
    if df is None:
        return None, None

    docs = build_documents_from_df(df, file_path)

    if os.path.exists(store_path):
        db = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
    else:
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(store_path)

    return db, df

# ----------- Search functions -----------
def search_in_dataframe(df, query):
    query_norm = normalize_text(query)
    for col in df.columns:
        if any(k in col.lower() for k in ["activity", "description"]):
            for idx, val in df[col].astype(str).items():
                if normalize_text(val) == query_norm:
                    return str(df.loc[idx, "Class"])
    return None

def vector_match_search(db, query, top_k=5):
    """Vector search + LLM reranking for best match."""
    if db is None:
        return None

    query_norm = normalize_text(query)
    results = db.similarity_search(query_norm, k=top_k)

    if not results:
        return None

    # Prepare candidate codes for LLM reranking
    candidates = [f"Code: {r.metadata.get('code','')} | Activity: {r.page_content}" for r in results]

    prompt = f"""
You are given a query: "{query}".
And a list of candidate activities with their codes:
{chr(10).join(candidates)}

Rank these candidates from most relevant to least relevant for the query, considering exact business meaning. 
Return only the code of the topmost relevant activity.
"""
    try:
        response = llm(prompt)
        top_code = re.findall(r'\d+', response.content)
        if top_code:
            return top_code[0]
    except Exception:
        # fallback if LLM fails
        return results[0].metadata.get("code", None)
    return None

def query_hierarchy(query):
    """Exact → vector with LLM reranking following your hierarchy."""
    # 1️⃣ Exact in DAFZA
    code = search_in_dataframe(knowledge_base["DAFZA.xlsx"]["df"], query)
    if code:
        return f"✅ Exact match in DAFZA.xlsx → Code: {code}"

    # 2️⃣ Vector in DAFZA
    code = vector_match_search(knowledge_base["DAFZA.xlsx"]["db"], query)
    if code:
        return f"ℹ️ Nearest match in DAFZA.xlsx → Code: {code}"

    # 3️⃣ Exact in ISIC
    code = search_in_dataframe(knowledge_base["ISIC.xlsx"]["df"], query)
    if code:
        return f"✅ Exact match in ISIC.xlsx → Code: {code}"

    # 4️⃣ Vector in ISIC
    code = vector_match_search(knowledge_base["ISIC.xlsx"]["db"], query)
    if code:
        return f"ℹ️ Nearest match in ISIC.xlsx → Code: {code}"

    return "❌ No relevant activity code found."

# ---------------- Streamlit UI ----------------
def main():
    st.title("RAG Knowledge Base for ISIC codes")

    for f in SEARCH_HIERARCHY:
        if f in VECTOR_STORE_PATHS:
            db, df = create_or_update_vectorstore(f, VECTOR_STORE_PATHS[f])
            if db:
                knowledge_base[f] = {"db": db, "df": df}

    st.success("Knowledge base ready! (cached)")

    query = st.text_input("Enter Activity / Description")
    if st.button("Search") and query:
        with st.spinner("Searching knowledge base..."):
            answer = query_hierarchy(query)
        st.write("### 🔍 Answer:")
        st.write(answer)

if __name__ == "__main__":
    main()
