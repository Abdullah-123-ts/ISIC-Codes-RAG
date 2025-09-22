# # =============== CONFIG ==================
# import os
# from dotenv import load_dotenv
# load_dotenv()
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


# import os
# import re
# import pandas as pd
# import streamlit as st
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_community.vectorstores import FAISS
# from langchain.schema import Document

# # ---------------- CONFIG ----------------
# VECTOR_STORE_PATHS = {
#     "DAFZA.xlsx": "store_dafza",
#     "ISIC.xlsx": "store_isic"
# }
# SEARCH_HIERARCHY = ["DAFZA.xlsx", "ISIC.xlsx"]

# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# knowledge_base = {}  # {file: {"db": ..., "df": ...}}

# # ---------------- HELPERS ----------------
# def load_dataframe(file_path):
#     if file_path.endswith(".xlsx"):
#         try:
#             return pd.read_excel(file_path)
#         except Exception as e:
#             st.error(f"Error loading {file_path}: {e}")
#             return None
#     return None

# def normalize_text(text):
#     text = str(text).lower()
#     text = re.sub(r"[\(\)\-\–]", " ", text)
#     text = re.sub(r"\s+", " ", text).strip()
#     return text

# def build_documents_from_df(df, file_tag):
#     docs = []
#     for _, row in df.iterrows():
#         if file_tag == "DAFZA.xlsx":
#             text = f"{row.get('Activity List','')} {row.get('ISIC Description','')}"
#             code = str(row.get("Class", ""))
#         elif file_tag == "ISIC.xlsx":
#             text = str(row.get("Activity name", ""))
#             code = str(row.get("Class", ""))
#         else:
#             text = " ".join([str(v) for v in row.values])
#             code = ""
#         doc = Document(page_content=normalize_text(text), metadata={"code": code})
#         docs.append(doc)
#     return docs

# # ----------- Vectorstore builder -----------
# @st.cache_resource
# def create_or_update_vectorstore(file_path, store_path):
#     df = load_dataframe(file_path)
#     if df is None:
#         return None, None

#     docs = build_documents_from_df(df, file_path)

#     if os.path.exists(store_path):
#         db = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
#     else:
#         db = FAISS.from_documents(docs, embeddings)
#         db.save_local(store_path)

#     return db, df

# # ----------- Search functions -----------
# def search_in_dataframe(df, query):
#     query_norm = normalize_text(query)
#     for col in df.columns:
#         if any(k in col.lower() for k in ["activity", "description"]):
#             for idx, val in df[col].astype(str).items():
#                 if normalize_text(val) == query_norm:
#                     return str(df.loc[idx, "Class"])
#     return None

# def vector_match_search(db, query, top_k=5):
#     """Vector search + LLM reranking for best match."""
#     if db is None:
#         return None

#     query_norm = normalize_text(query)
#     results = db.similarity_search(query_norm, k=top_k)

#     if not results:
#         return None

#     # Prepare candidate codes for LLM reranking
#     candidates = [f"Code: {r.metadata.get('code','')} | Activity: {r.page_content}" for r in results]

#     prompt = f"""
# You are given a query: "{query}".
# And a list of candidate activities with their codes:
# {chr(10).join(candidates)}

# Rank these candidates from most relevant to least relevant for the query, considering exact business meaning. 
# Return only the code of the topmost relevant activity.
# """
#     try:
#         response = llm(prompt)
#         top_code = re.findall(r'\d+', response.content)
#         if top_code:
#             return top_code[0]
#     except Exception:
#         # fallback if LLM fails
#         return results[0].metadata.get("code", None)
#     return None

# def query_hierarchy(query):
#     """Exact → vector with LLM reranking following your hierarchy."""
#     # 1️⃣ Exact in DAFZA
#     code = search_in_dataframe(knowledge_base["DAFZA.xlsx"]["df"], query)
#     if code:
#         return f"✅ Exact match in DAFZA.xlsx → Code: {code}"

#     # 2️⃣ Vector in DAFZA
#     code = vector_match_search(knowledge_base["DAFZA.xlsx"]["db"], query)
#     if code:
#         return f"ℹ️ Nearest match in DAFZA.xlsx → Code: {code}"

#     # 3️⃣ Exact in ISIC
#     code = search_in_dataframe(knowledge_base["ISIC.xlsx"]["df"], query)
#     if code:
#         return f"✅ Exact match in ISIC.xlsx → Code: {code}"

#     # 4️⃣ Vector in ISIC
#     code = vector_match_search(knowledge_base["ISIC.xlsx"]["db"], query)
#     if code:
#         return f"ℹ️ Nearest match in ISIC.xlsx → Code: {code}"

#     return "❌ No relevant activity code found."

# # ---------------- Streamlit UI ----------------
# def main():
#     st.title("RAG Knowledge Base for ISIC codes")

#     for f in SEARCH_HIERARCHY:
#         if f in VECTOR_STORE_PATHS:
#             db, df = create_or_update_vectorstore(f, VECTOR_STORE_PATHS[f])
#             if db:
#                 knowledge_base[f] = {"db": db, "df": df}

#     st.success("Knowledge base ready! (cached)")

#     query = st.text_input("Enter Activity / Description")
#     if st.button("Search") and query:
#         with st.spinner("Searching knowledge base..."):
#             answer = query_hierarchy(query)
#         st.write("### 🔍 Answer:")
#         st.write(answer)

# if __name__ == "__main__":
#     main()




















##############dafza->mydan->spc->isic, at end vector + llm on all to pick best##############

# rag_app.py
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import re

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"&", "and", text)       # replace & with "and"
    text = re.sub(r"\s+", " ", text)       # collapse multiple spaces
    return text


from rapidfuzz import fuzz

def fuzzy_exact_match(query, df, threshold=90):
    best_row = None
    best_score = 0
    for _, row in df.iterrows():
        score = fuzz.ratio(normalize_text(query), normalize_text(row["activity name"]))
        if score > best_score:
            best_score, best_row = score, row
    if best_score >= threshold:
        return best_row, best_score
    return None, None
# ---------------- CONFIG ----------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

VECTOR_STORE_DIR = "vectorstores"
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# List of Excel sheets in search order
SHEETS_ORDER = ["dafza.xlsx", "meydan.xlsx", "spc.xlsx", "isic.xlsx"]

# ---------------- LOAD SHEETS ----------------
@st.cache_data
def load_sheets():
    """Load all Excel sheets into DataFrames with normalized column names."""
    dataframes = {}
    for sheet in SHEETS_ORDER:
        df = pd.read_excel(sheet)
        df.columns = [c.strip().lower() for c in df.columns]
        if "activity name" not in df.columns or "class" not in df.columns:
            raise ValueError(f"Sheet {sheet} must contain 'Activity Name' and 'Class' columns.")
        dataframes[sheet] = df
    return dataframes

# ---------------- VECTOR STORE ----------------
@st.cache_resource
def load_or_create_vectorstore(sheet_name: str, df: pd.DataFrame):
    """Create or load FAISS vector store for a given sheet."""
    store_path = os.path.join(VECTOR_STORE_DIR, f"{sheet_name}_faiss")
    if os.path.exists(store_path):
        vectorstore = FAISS.load_local(
            store_path,
            OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
            allow_dangerous_deserialization=True
        )
    else:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        docs = [
            Document(page_content=row["activity name"], metadata={"class": row["class"]})
            for _, row in df.iterrows()
        ]
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(store_path)
    return vectorstore

# ---------------- SEARCH LOGIC ----------------
def search_activity(query, dataframes, vectorstores):
    """Search across all sheets with exact match first, then vector + LLM fallback."""

    # # Step 1: Exact match in all sheets
    # query_norm = normalize_text(query)

    # for sheet in SHEETS_ORDER:
    #     df = dataframes[sheet]
    #     df["normalized"] = df["activity name"].apply(normalize_text)
    #     match = df[df["normalized"] == query_norm]
    #     if not match.empty:
    #         row = match.iloc[0]
    #         return {
    #             "source": sheet,
    #             "activity": row["activity name"],
    #             "code": row["class"],
    #             "method": "Exact Match (Normalized)"
    #         }

    for sheet in SHEETS_ORDER:
        df = dataframes[sheet]
        row, score = fuzzy_exact_match(query, df)
        if row is not None:
            return {
                "source": sheet,
                "activity": row["activity name"],
                "code": row["class"],
                "method": f"Fuzzy Exact Match ({score}%)"
            }

    # Step 2: Vector similarity search in all sheets
    all_candidates = []
    for sheet in SHEETS_ORDER:
        docs = vectorstores[sheet].similarity_search(query, k=3)  # top 3 per sheet
        candidates = [
            {"sheet": sheet, "activity": d.page_content, "code": d.metadata.get("class")}
            for d in docs
        ]
        all_candidates.extend(candidates)

    # Step 3: Use LLM to refine best choice
    llm = ChatOpenAI(model="gpt-5-mini", openai_api_key=OPENAI_API_KEY, temperature=1)

    prompt = f"""
    The user is searching for an activity: "{query}".
    Here are candidate activities with their codes from different sheets:
    {all_candidates}

    Please choose the single most appropriate code and activity based on business context.
    Respond in JSON as: {{"sheet": "...", "activity": "...", "code": "..."}}.
    """
    response = llm.invoke(prompt)

    return {
        "source": "Multiple Sheets",
        "method": "Vector + GPT",
        "raw_candidates": all_candidates,
        "llm_response": response.content
    }

# ---------------- STREAMLIT APP ----------------
def main():
    st.title("Activity Code Finder (RAG)")
    st.write("Search for activity codes across multiple Excel sheets with exact + vector search fallback.")

    dataframes = load_sheets()
    vectorstores = {sheet: load_or_create_vectorstore(sheet, df) for sheet, df in dataframes.items()}

    query = st.text_input("Enter activity name/description:")

    if st.button("Search") and query:
        result = search_activity(query, dataframes, vectorstores)
        st.subheader("Result")
        st.json(result)

if __name__ == "__main__":
    main()
