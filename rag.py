
# # Improved rag_app.py — returns structured (human-friendly) results instead of raw JSON
# # - Robust fuzzy exact match
# # - FAISS vectorstore per sheet
# # - LLM refinement with strict JSON instruction and robust parsing fallback
# # - Streamlit UI shows a readable result instead of raw LLM JSON

# import os
# import re
# import json
# import ast
# import pandas as pd
# import streamlit as st
# from dotenv import load_dotenv

# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_community.vectorstores import FAISS
# from langchain.schema import Document
# from rapidfuzz import fuzz

# # ---------------- HELPERS ----------------

# def normalize_text(text: str) -> str:
#     if not isinstance(text, str):
#         return ""
#     text = text.lower().strip()
#     text = re.sub(r"&", "and", text)       # replace & with "and"
#     text = re.sub(r"\s+", " ", text)       # collapse multiple spaces
#     return text


# def fuzzy_exact_match(query: str, df: pd.DataFrame, threshold: int = 95):
#     """Return the best-matching row (Series) and score if score >= threshold."""
#     best_row = None
#     best_score = 0
#     qn = normalize_text(query)
#     for _, row in df.iterrows():
#         candidate = normalize_text(row.get("activity name", ""))
#         score = fuzz.ratio(qn, candidate)
#         if score > best_score:
#             best_score = score
#             best_row = row
#     if best_score >= threshold:
#         return best_row, int(best_score)
#     return None, None


# # ---------------- CONFIG ----------------
# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# VECTOR_STORE_DIR = "vectorstores"
# os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# # Provide the Excel file names (same as original) — adjust to your paths if needed
# SHEETS_ORDER = ["dafza.xlsx", "meydan.xlsx", "spc.xlsx", "isic.xlsx"]


# # ---------------- LOAD SHEETS ----------------
# @st.cache_data
# def load_sheets():
#     """Load all Excel sheets into DataFrames with normalized column names."""
#     dataframes = {}
#     for sheet in SHEETS_ORDER:
#         df = pd.read_excel(sheet)
#         # normalize column names to lowercase and trimmed
#         df.columns = [c.strip().lower() for c in df.columns]
#         if "activity name" not in df.columns or "class" not in df.columns:
#             raise ValueError(f"Sheet {sheet} must contain 'Activity Name' and 'Class' columns.")
#         dataframes[sheet] = df
#     return dataframes


# # ---------------- VECTOR STORE ----------------
# @st.cache_resource
# def load_or_create_vectorstore(sheet_name: str, df: pd.DataFrame):
#     """Create or load FAISS vector store for a given sheet."""
#     store_path = os.path.join(VECTOR_STORE_DIR, f"{sheet_name}_faiss")
#     embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

#     if os.path.exists(store_path):
#         vectorstore = FAISS.load_local(
#             store_path,
#             embeddings,
#             allow_dangerous_deserialization=True,
#         )
#     else:
#         docs = [
#             Document(page_content=str(row["activity name"]), metadata={"class": str(row["class"])})
#             for _, row in df.iterrows()
#         ]
#         vectorstore = FAISS.from_documents(docs, embeddings)
#         vectorstore.save_local(store_path)
#     return vectorstore


# # ---------------- SEARCH LOGIC ----------------

# def parse_json_like(text: str):
#     """Try to parse a JSON object from arbitrary LLM text with multiple fallbacks."""
#     text = text.strip()
#     # Direct JSON
#     try:
#         return json.loads(text)
#     except Exception:
#         pass

#     # Extract first {...} block and try parse
#     m = re.search(r"\{.*\}", text, re.S)
#     if m:
#         candidate = m.group(0)
#         try:
#             return json.loads(candidate)
#         except Exception:
#             try:
#                 # fallback to python literal
#                 return ast.literal_eval(candidate)
#             except Exception:
#                 return {"llm_raw": text}
#     # nothing found
#     return {"llm_raw": text}


# def search_activity(query: str, dataframes: dict, vectorstores: dict):
#     """Search across all sheets with fuzzy-exact first, then vector + LLM fallback.

#     Returns a structured dict with keys: source, activity, code, method, and optional reason/score.
#     """

#     # 1) Fuzzy exact match across sheets (fast + deterministic)
#     for sheet in SHEETS_ORDER:
#         df = dataframes[sheet]
#         row, score = fuzzy_exact_match(query, df, threshold=95)
#         if row is not None:
#             return {
#                 "source": sheet,
#                 "activity": row["activity name"],
#                 "code": row["class"],
#                 "method": f"Fuzzy Exact Match ({score}%)",
#                 "score": score,
#             }

#     # 2) Vector similarity search in all sheets (collect top candidates)
#     all_candidates = []
#     for sheet in SHEETS_ORDER:
#         docs = vectorstores[sheet].similarity_search(query, k=3)
#         candidates = [
#             {"sheet": sheet, "activity": d.page_content, "code": d.metadata.get("class")}
#             for d in docs
#         ]
#         all_candidates.extend(candidates)

#     # 3) Use LLM to refine best choice — ask for strict JSON
#     llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY, temperature=0)
#     prompt = f"""
# You are given a user's query and candidate activities with their sheet names and codes.
# Choose the single best match from the candidates and return ONLY a JSON object with these keys:
#   - sheet: the filename (string)
#   - activity: the exact activity string you pick (string)
#   - code: the class/code for that activity (string)
#   - reason: a one-line reason for your choice (string)
# Candidates: {all_candidates}
# User query: "{query}"
# Respond with a JSON object and nothing else.
# """

#     response = llm.invoke(prompt)
#     parsed = parse_json_like(response.content)

#     # If LLM returned the expected keys, use it
#     if isinstance(parsed, dict) and all(k in parsed for k in ("sheet", "activity", "code")):
#         return {
#             "source": parsed.get("sheet"),
#             "activity": parsed.get("activity"),
#             "code": parsed.get("code"),
#             "method": "Vector + LLM",
#             "reason": parsed.get("reason"),
#             "llm_raw": response.content,
#         }

#     # Fallback: choose the candidate with best fuzzy score to the query
#     if all_candidates:
#         best = max(
#             all_candidates,
#             key=lambda c: fuzz.ratio(normalize_text(query), normalize_text(c.get("activity", "")))
#         )
#         score = fuzz.ratio(normalize_text(query), normalize_text(best.get("activity", "")))
#         return {
#             "source": best.get("sheet"),
#             "activity": best.get("activity"),
#             "code": best.get("code"),
#             "method": "Vector (best candidate fallback)",
#             "score": int(score),
#             "llm_raw": response.content,
#         }

#     # Last resort: nothing found
#     return {
#         "source": None,
#         "activity": None,
#         "code": None,
#         "method": "No match",
#         "llm_raw": response.content,
#     }


# # ---------------- STREAMLIT APP ----------------

# def main():
#     st.title("Activity Code Finder (RAG)")
#     st.write("Search for activity codes across multiple Excel sheets with fuzzy + vector + LLM fallback.")

#     try:
#         dataframes = load_sheets()
#     except Exception as e:
#         st.error(f"Failed to load sheets: {e}")
#         return

#     # Create/load vectorstores for each sheet
#     vectorstores = {sheet: load_or_create_vectorstore(sheet, df) for sheet, df in dataframes.items()}

#     query = st.text_input("Enter activity name/description:")

#     if st.button("Search") and query:
#         result = search_activity(query, dataframes, vectorstores)

#         st.subheader("Result")
#         if result.get("activity"):
#             st.success(result.get("activity"))
#             st.write("**Code:**", result.get("code"))
#             st.write("**Source sheet:**", result.get("source"))
#             st.write("**Method:**", result.get("method"))
#             if result.get("score") is not None:
#                 st.write("**Score:**", result.get("score"))
#             if result.get("reason"):
#                 st.write("**Reason:**", result.get("reason"))

#         else:
#             st.warning("No confident match found.")



# if __name__ == "__main__":
#     main()







# rag_app_updated.py — Fast Multi-Activity Search
# - Fast mode (fuzzy + vector) for 1–2 sec responses
# - Optional LLM refinement (slower but more accurate)
# - Handles multiple activities at once

import os
import re
import json
import ast
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from rapidfuzz import fuzz


# ---------------- HELPERS ----------------

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"&", "and", text)       # replace & with "and"
    text = re.sub(r"\s+", " ", text)       # collapse multiple spaces
    return text


def fuzzy_exact_match(query: str, df: pd.DataFrame, threshold: int = 95):
    """Return the best-matching row (Series) and score if score >= threshold."""
    best_row = None
    best_score = 0
    qn = normalize_text(query)
    for _, row in df.iterrows():
        candidate = normalize_text(row.get("activity name", ""))
        score = fuzz.ratio(qn, candidate)
        if score > best_score:
            best_score = score
            best_row = row
    if best_score >= threshold:
        return best_row, int(best_score)
    return None, None


def parse_json_like(text: str):
    """Try to parse a JSON object from arbitrary LLM text with multiple fallbacks."""
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        candidate = m.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            try:
                return ast.literal_eval(candidate)
            except Exception:
                return {"llm_raw": text}
    return {"llm_raw": text}


# ---------------- CONFIG ----------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

VECTOR_STORE_DIR = "vectorstores"
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

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
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    if os.path.exists(store_path):
        vectorstore = FAISS.load_local(
            store_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    else:
        docs = [
            Document(page_content=str(row["activity name"]), metadata={"class": str(row["class"])} )
            for _, row in df.iterrows()
        ]
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(store_path)
    return vectorstore


# ---------------- SEARCH LOGIC ----------------

def search_activity(query: str, dataframes: dict, vectorstores: dict, use_llm: bool = False):
    """
    Search for a single activity.
    If use_llm=False -> fast (1–2 sec).
    If use_llm=True  -> slower, with LLM refinement.
    """

    # 1) Fuzzy exact match (fastest)
    for sheet in SHEETS_ORDER:
        df = dataframes[sheet]
        row, score = fuzzy_exact_match(query, df, threshold=95)
        if row is not None:
            return {
                "source": sheet,
                "query" : query,
                "activity": row["activity name"],
                "code": row["class"],
                "method": f"Fuzzy Exact Match ({score}%)",
                "score": score,
            }

    # 2) Vector similarity search
    all_candidates = []
    for sheet in SHEETS_ORDER:
        docs = vectorstores[sheet].similarity_search(query, k=5)
        for d in docs:
            all_candidates.append({
                "query" : query,
                "sheet": sheet,
                "activity": d.page_content,
                "code": d.metadata.get("class")
            })

    # Fast mode: return first/best candidate directly
    if not use_llm:
        if all_candidates:
            best = all_candidates[0]
            return {
                "query" : query,
                "source": best["sheet"],
                "activity": best["activity"],
                "code": best["code"],
                "method": "Vector (fast mode)",
            }
        return {"source": None, "activity": None, "code": None, "method": "No match"}

    # 3) LLM refinement (slower)
    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY, temperature=0)
    prompt = f"""
You are given a user's query and candidate activities with their sheet names and codes.
Choose the single best match and return ONLY a JSON object with keys:
  - sheet
  - activity
  - code
  - reason
Candidates: {all_candidates}
User query: "{query}"
"""
    response = llm.invoke(prompt)
    parsed = parse_json_like(response.content)

    if isinstance(parsed, dict) and all(k in parsed for k in ("sheet", "activity", "code")):
        return {
            "query" : query,
            "source": parsed.get("sheet"),
            "activity": parsed.get("activity"),
            "code": parsed.get("code"),
            "method": "Vector + LLM",
            "reason": parsed.get("reason"),
            "llm_raw": response.content,
        }

    return {"source": None, "activity": None, "code": None, "method": "No match", "llm_raw": response.content}


def search_multiple_activities(queries, dataframes, vectorstores, use_llm=False):
    """Search codes for multiple activity queries."""
    results = []
    for q in queries:
        q = q.strip()
        if not q:
            continue
        result = search_activity(q, dataframes, vectorstores, use_llm=use_llm)
        results.append(result)
    return results


# ---------------- STREAMLIT APP ----------------

def main():
    st.title("Activity Code Finder (RAG)")
    st.write("Search activity codes across multiple Excel sheets. Supports multiple activities at once.")

    try:
        dataframes = load_sheets()
    except Exception as e:
        st.error(f"Failed to load sheets: {e}")
        return

    vectorstores = {sheet: load_or_create_vectorstore(sheet, df) for sheet, df in dataframes.items()}

    query = st.text_area("Enter one or multiple activities (comma or newline separated):")
    use_llm = st.checkbox("Use LLM refinement (slower, more accurate)", value=False)

    if st.button("Search") and query:
        queries = [q.strip() for q in re.split(r"[,\n]", query) if q.strip()]
        results = search_multiple_activities(queries, dataframes, vectorstores, use_llm=use_llm)

        for res in results:
            st.subheader(res.get("activity") or "No match")
            st.write("**Query:**", res.get("query"))
            st.write("**Code:**", res.get("code"))
            st.write("**Source sheet:**", res.get("source"))
            st.write("**Method:**", res.get("method"))
            if res.get("score") is not None:
                st.write("**Score:**", res.get("score"))
            if res.get("reason"):
                st.write("**Reason:**", res.get("reason"))


if __name__ == "__main__":
    main()
