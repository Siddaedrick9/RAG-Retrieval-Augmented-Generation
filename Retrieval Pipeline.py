#  ## CHATBOT WITH Excel Directory, 
#  ## Excel workbook sheet cell level data Changes or Updates Detection and Auto Reindexing the FAISS Vector Store and BM25, 
#  ## with Embedding Model = paraphrase-multilingual-MiniLM-L12-v2 with hindi marathi english support to understand user level input query,
#  ## Then Cross-Encoder based Re-ranking, Threshold Score of User Query based Filtering and Google Gemini LLM for Answer Generation,
#  ## with Document Metadata in Answer Generation Block,
#  ## and Chat Memory for 5 Previous Conversations History
#  ## and without Chat Memory for Answer Generation Block as well!
#  ## Adding Streamlit UI for this code in future. (minimal)

import os
import json
import hashlib
import pickle
import time
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferWindowMemory
import google.generativeai as genai


genai.configure(api_key= "YOUR_GOOGLE_API_KEY_HERE" )
try:
    model_llm =  model_llm = genai.GenerativeModel(
    model_name="models/gemini-2.0-flash-001",
    generation_config=genai.types.GenerationConfig(
        temperature=0.9,
        max_output_tokens=5000,
        top_k=30,
        top_p=0.95
    )
)
except:
    model_llm = genai.GenerativeModel(
    model_name="models/gemini-1.5-pro",
    generation_config=genai.types.GenerationConfig(
        temperature=0.9,
        max_output_tokens=5000,
        top_k=30,
        top_p=0.95
    )
)

# === Paths ===
EXCEL_DIR = "C:/Users/SiddhantMutha/Desktop/Excel Documents"
FAISS_DIR = "C:/Users/SiddhantMutha/Desktop/faiss_indexes11"
BM25_DIR = "C:/Users/SiddhantMutha/Desktop/BM25_Index_Files11"
BM25_INDEX_PATH = os.path.join(BM25_DIR, "bm25.pkl")
BM25_DOCS_PATH = os.path.join(BM25_DIR, "documents.pkl")
BM25_META_PATH = os.path.join(BM25_DIR, "file_hashes.pkl")

# === Embeddings and Reranker Setup ===
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"
tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# === Excel Hashing ===
def compute_directory_excel_hash(directory):
    sha = hashlib.sha256()
    for file_path in sorted(Path(directory).glob("*.xlsx")):
        sha.update(file_path.name.encode())
        xls = pd.ExcelFile(file_path)
        for sheet in xls.sheet_names:
            df = xls.parse(sheet)
            sha.update(df.to_csv(index=False).encode())
    return sha.hexdigest()

# === FAISS ===
def load_excel_docs(directory):
    docs = []
    for filename in os.listdir(directory):
        if filename.endswith(".xlsx"):
            filepath = os.path.join(directory, filename)
            xls = pd.ExcelFile(filepath)
            for sheet_name in xls.sheet_names:
                df = xls.parse(sheet_name)
                for index, row in df.iterrows():
                    row_with_headers = [f"{col}: {val}" for col, val in row.items()]
                    page_content = " | ".join(row_with_headers)
                    metadata = {
                        "source": filename,
                        "sheet": sheet_name,
                        "row_index": index,
                        "columns": list(df.columns),
                        "row_data": row.to_dict()
                    }
                    docs.append(Document(page_content=page_content, metadata=metadata))
    return docs

def create_faiss_index(docs, save_path):
    index = FAISS.from_documents(docs, embedding_model)
    index.save_local(save_path)
    print(f"✅ FAISS index saved to {save_path}")

def load_or_create_faiss_index(current_hash, stored_hash):
    if current_hash != stored_hash or not os.path.exists(os.path.join(FAISS_DIR, "index.faiss")):
        print("🔁 FAISS: Index outdated or missing. Rebuilding...")
        docs = load_excel_docs(EXCEL_DIR)
        print(f"Loaded {len(docs)} rows.")
        create_faiss_index(docs, FAISS_DIR)
    else:
        print("✅ FAISS: Index is up to date.")
    return FAISS.load_local(FAISS_DIR, embedding_model, index_name="index", allow_dangerous_deserialization=True)

# === BM25 ===
def save_bm25_metadata(hash_val):
    os.makedirs(BM25_DIR, exist_ok=True)
    with open(BM25_META_PATH, "w") as f:
        json.dump({"excel_hash": hash_val}, f)

def load_bm25_metadata():
    if not os.path.exists(BM25_META_PATH) or os.path.getsize(BM25_META_PATH) == 0:
        return None
    try:
        with open(BM25_META_PATH, "r") as f:
            return json.load(f).get("excel_hash")
    except json.JSONDecodeError:
        print("⚠️ BM25 metadata file is corrupted. Rebuilding index.")
        return None

def save_bm25_index(index, raw_docs):
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(index, f)
    with open(BM25_DOCS_PATH, "wb") as f:
        pickle.dump(raw_docs, f)

def build_bm25_index_from_excel():
    raw_docs = []
    for file_path in sorted(Path(EXCEL_DIR).glob("*.xlsx")):
        try:
            xls = pd.ExcelFile(file_path)
            for sheet in xls.sheet_names:
                df = xls.parse(sheet)
                for _, row in df.iterrows():
                    content = " ".join([str(x) for x in row if pd.notna(x)]).strip()
                    if content:
                        raw_docs.append({
                            "text": content,
                            "metadata": {"filename": file_path.name, "sheet": sheet}
                        })
        except Exception as e:
            print(f"⚠️ Failed to read or parse {file_path.name}: {e}")

    if not raw_docs:
        raise ValueError("❌ No valid rows found in Excel files. BM25 index cannot be built.")

    tokenized_corpus = [doc["text"].lower().split() for doc in raw_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    save_bm25_index(bm25, raw_docs)
    return bm25, raw_docs

# === Reranking ===
def rerank_and_filter(query, docs, threshold=0.95, top_n=30):
    pairs = [(query, doc["content"] if isinstance(doc, dict) else doc.page_content) for doc in docs]
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        scores = model(**inputs).logits.squeeze().tolist()
    if isinstance(scores, float): scores = [scores]
    scored = list(zip(docs, scores))
    filtered = [x for x in scored if x[1] >= threshold]
    reranked = sorted(filtered, key=lambda x: x[1], reverse=True)
    if reranked:
        print(f"\n✅ {len(reranked)} results above threshold.")
    else:
        print(f"\n❌ 0 results above threshold ({threshold}). Total: {len(scored)}.")
    return reranked[:top_n]

# === Hybrid Search ===
def hybrid_query(query, faiss_k=200, bm25_k=200, threshold=0.95, top_n=30):
    current_hash = compute_directory_excel_hash(EXCEL_DIR)
    stored_hash = load_bm25_metadata()

    try:
        db = load_or_create_faiss_index(current_hash, stored_hash)
    except Exception as e:
        print(f"❌ FAISS loading failed: {e}")
        return

    if stored_hash != current_hash or not os.path.exists(BM25_INDEX_PATH):
        print("🔁 BM25: Index outdated or missing. Rebuilding...")
        bm25, bm25_docs = build_bm25_index_from_excel()
        save_bm25_metadata(current_hash)
    else:
        try:
            with open(BM25_INDEX_PATH, "rb") as f1, open(BM25_DOCS_PATH, "rb") as f2:
                bm25 = pickle.load(f1)
                bm25_docs = pickle.load(f2)
        except Exception as e:
            print(f"❌ BM25 loading failed: {e}")
            return

    faiss_docs = db.similarity_search(query, k=faiss_k)
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_ranked = sorted(zip(bm25_docs, bm25_scores), key=lambda x: x[1], reverse=True)[:bm25_k]
    bm25_wrapped = [{"content": d["text"], "metadata": d["metadata"]} for d, _ in bm25_ranked]

    merged = faiss_docs + bm25_wrapped
    seen = set()
    unique_docs = []
    for doc in merged:
        content = doc.page_content if hasattr(doc, "page_content") else doc["content"]
        if content not in seen:
            seen.add(content)
            unique_docs.append(doc)

    reranked_results = rerank_and_filter(query, unique_docs, threshold, top_n)
    if not reranked_results:
        print("⚠️ No relevant answers found. Try rephrasing.")
        return

    for i, (doc, score) in enumerate(reranked_results):
        content = doc.page_content if hasattr(doc, "page_content") else doc["content"]
        metadata = doc.metadata if hasattr(doc, "metadata") else doc["metadata"]
        print(f"\n🔹 Result {i+1} (Score: {score:.4f})")
        print("📄 Content:", content)
        print("🗂 Metadata:", metadata)
        print("🔍 Reranked Score:", score   )

    return reranked_results[:top_n]




###  Content WITH Metadata in Answer Generation Block Function Below

# context = "\n\n".join(
#     f"Content: {doc.page_content if hasattr(doc, 'page_content') else doc['content']}\n"
#     f"Metadata: {json.dumps(doc.metadata if hasattr(doc, 'metadata') else doc['metadata'], indent=2)}"
#     for doc, _ in reranked_results)
 


###  Content WITHOUT Metadata in Answer Generation Block Function Below

# context = "\n\n".join(
#     f"Content: {doc.page_content if hasattr(doc, 'page_content') else doc['content']}"
#     for doc, _ in reranked_results)






# === Step 3: Define Answer Generation Chain WITH Chat Memory  ===


# memory = ConversationBufferWindowMemory(k=5, return_messages=True)
# def generate_custom_answer(user_query, reranked_results, model_llm, memory):
#     chat_history = memory.load_memory_variables({})["history"]  # Get last k interactions
#     history_text = "\n".join([f"User: {msg.content}" if msg.type == "human" else f"Bot: {msg.content}" for msg in chat_history])

#     context = "\n\n".join(
#         f"Content: {doc.page_content if hasattr(doc, 'page_content') else doc['content']}\n"
#         f"Metadata: {json.dumps({k: v for k, v in (doc.metadata if hasattr(doc, 'metadata') else doc['metadata']).items() if k in ['filename', 'sheet', 'row_data', 'row_index', 'columns']}, indent=2)}"
#         for doc, _ in reranked_results
#     )

#     prompt = f"""You are a helpful student counselor. Use only the following context and prior conversation to answer the user's question.

# ### Previous Conversation:
# {history_text}

# ### Context:
# {context}

# ### Question:
# {user_query}

# ### Answer:"""

#     try:
#         response = model_llm.generate_content(prompt)
#         print(f"\n🧠 Answer:\n{response.text.strip()}")
#         memory.save_context({"input": user_query}, {"output": response.text.strip()})
#         return response.text.strip()
#     except Exception as e:
#         print(f"❌ Generation error: {e}")
#         return "⚠️ Sorry, I couldn't generate an answer at this moment."






# === Step 3: Define Answer Generation Chain without Chat Memory and WITH Metadata ======   use other content line for without metadata defined and commented out above

def generate_custom_answer(user_query, reranked_results, model_llm):
    context = "\n\n".join(
        f"Content: {doc.page_content if hasattr(doc, 'page_content') else doc['content']}\n"
        f"Metadata: {json.dumps({k: v for k, v in (doc.metadata if hasattr(doc, 'metadata') else doc['metadata']).items() if k in ['filename', 'sheet', 'columns', 'row_index', 'row_data']}, indent=2)}"
        for doc, _ in reranked_results
)

    prompt = f"""You are a helpful student counselor. Use only the following context to answer the user's question concisely and helpfully. Strictly do not perform web search and reasoning capabilities from your side.

    ### Context:
    {context}

    ### Question:
    {user_query}

    ### Answer:"""

    try:
        response = model_llm.generate_content(prompt)
        print(f"\n🧠 Answer:\n{response.text.strip()}")
        return response.text.strip()
    except Exception as e:
        print(f"❌ Generation error: {e}")
        return "⚠️ Sorry, I couldn't generate an answer at this moment."


# === CLI ===
if __name__ == "__main__":
    while True:
        user_query = input("\nEnter your query (or type 'exit' or 'quit' to exit): ").strip()
        if user_query.lower() in {"quit", "exit"}:
            print("👋 Exiting...")
            break
        if user_query:
            t1 = time.time()
            reranked_results = hybrid_query(user_query)
            if reranked_results:
                answer = generate_custom_answer(user_query, reranked_results, model_llm) ##Use this line for WITHOUT Chat Memory
                ##answer = generate_custom_answer(user_query, reranked_results, model_llm, memory) ##Use this line for Chat Memory
            t2 = time.time()
            print(f"\n⏱ Time taken: {t2 - t1:.2f} seconds")