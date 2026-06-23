import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Zyro Dynamics HR Help Desk", page_icon="\U0001F4BC")
st.title("\U0001F4BC Zyro Dynamics HR Help Desk")
st.caption("Ask me anything about company HR policies. I will only answer from official policy documents.")

# ----------------------------------------------------------------------------
# Config (Cell 2 equivalent)
# ----------------------------------------------------------------------------
LLM_PROVIDER = "groq"  # "groq" | "gemini" | "openai"
LLM_MODEL = "llama-3.3-70b-versatile"

# The official Kaggle competition path (only exists when running inside a
# Kaggle notebook/kernel — NOT available on Streamlit Cloud).
_KAGGLE_CORPUS_PATH = "/kaggle/input/competitions/niat-masterclass-rag-challenge/zyro-dynamics-hr-corpus/"

# Local fallback path for deployment (Streamlit Cloud, or anywhere else that
# isn't Kaggle) — the 11 PDFs must be committed into a `corpos` folder next
# to app.py in your repo.
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_LOCAL_CORPUS_PATH = os.path.join(_APP_DIR, "corpos")

if "CORPUS_PATH" in os.environ:
    CORPUS_PATH = os.environ["CORPUS_PATH"]
elif os.path.isdir(_KAGGLE_CORPUS_PATH):
    CORPUS_PATH = _KAGGLE_CORPUS_PATH
else:
    CORPUS_PATH = _LOCAL_CORPUS_PATH

# ----------------------------------------------------------------------------
# API keys
# ----------------------------------------------------------------------------
# On Streamlit Cloud, set these under "Secrets" in the app settings:
# GROQ_API_KEY = "..."
if LLM_PROVIDER == "groq" and "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
elif LLM_PROVIDER == "gemini" and "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
elif LLM_PROVIDER == "openai" and "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

if "LANGCHAIN_API_KEY" in st.secrets:
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "zyro-rag-challenge"

# ----------------------------------------------------------------------------
# Debug panel — helps diagnose "corpos/ not found" issues on first deploy
# ----------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Diagnostics")
    st.caption(f"Corpus path: `{CORPUS_PATH}`")
    if os.path.isdir(CORPUS_PATH):
        pdfs = [f for f in os.listdir(CORPUS_PATH) if f.lower().endswith(".pdf")]
        st.caption(f"PDFs found: {len(pdfs)}")
        if pdfs:
            with st.expander("Files"):
                for p in sorted(pdfs):
                    st.write(f"- {p}")
    else:
        st.caption("Folder does not exist yet.")


# ----------------------------------------------------------------------------
# Build the RAG pipeline once and cache it across reruns
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading HR policy documents and building the knowledge base...")
def build_pipeline():
    if not os.path.isdir(CORPUS_PATH):
        st.error(
            f"Corpus folder not found at `{CORPUS_PATH}`.\n\n"
            "If you're running this on **Streamlit Cloud**, the Kaggle "
            "competition path isn't accessible there — you need to commit "
            "a `corpos` folder containing the 11 HR policy PDFs to your repo, "
            "next to `app.py`.\n\n"
            "If you're running this **inside Kaggle**, double-check the "
            "competition data path under the Data tab."
        )
        st.stop()

    loader = PyPDFDirectoryLoader(CORPUS_PATH)
    documents = loader.load()

    if not documents:
        st.error(
            f"No PDF files were found in `{CORPUS_PATH}`.\n\n"
            "Check that the 11 HR policy PDFs were actually committed to "
            "git (not just present locally) and that they have a `.pdf` extension."
        )
        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        st.error("Documents were loaded but produced zero chunks. Check the PDF contents.")
        st.stop()

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5}
    )

    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        llm = ChatGroq(model=LLM_MODEL, temperature=0.1, max_tokens=512)
    elif LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.1, max_output_tokens=512)
    elif LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0.1, max_tokens=512)
    else:
        raise ValueError("Unsupported LLM provider.")

    return retriever, llm


retriever, llm = build_pipeline()

RAG_PROMPT = ChatPromptTemplate.from_template(
    "You are the HR Help Desk assistant for Zyro Dynamics.\n"
    "Answer the employee question using ONLY the information in the context below, "
    "which is drawn from the company's internal HR policy documents.\n\n"
    "Rules:\n"
    "- Base your answer strictly on the provided context. Do not use outside knowledge.\n"
    "- If the context does not contain enough information to answer the question, "
    "say: I don't have enough information in the HR policy documents to answer that.\n"
    "- Be clear, concise, and cite the relevant policy by name when possible.\n"
    "- Use a professional, helpful tone, as if speaking to an employee.\n\n"
    "Context:\n{context}\n\n"
    "Question:\n{question}\n\n"
    "Answer:"
)

OOS_PROMPT = ChatPromptTemplate.from_template(
    "You are a strict scope classifier for an HR Help Desk chatbot at Zyro Dynamics.\n\n"
    "The chatbot may ONLY answer questions about internal HR policies, such as: "
    "leave policy, work from home policy, code of conduct, performance reviews, "
    "compensation and benefits, IT and data security, POSH policy, onboarding and "
    "separation, and travel and expense policy.\n\n"
    "The chatbot must REFUSE questions that are NOT about these internal HR "
    "policies, including (but not limited to): general company financials, "
    "product or feature details, recruitment or hiring process details not covered "
    "by HR policy documents, comparisons to other companies, or any unrelated "
    "general-knowledge question.\n\n"
    "Classify the employee question below as either IN_SCOPE or OUT_OF_SCOPE. "
    "Respond with exactly one word: IN_SCOPE or OUT_OF_SCOPE.\n\n"
    "Question: {question}\n\n"
    "Classification:"
)

REFUSAL_MESSAGE = (
    "I can only answer HR-related questions from Zyro Dynamics policy documents. "
    "This question falls outside that scope, so I'm unable to help with it. "
    "Please reach out to the relevant team or your HR representative for assistance."
)


def format_docs(docs):
    formatted = []
    for d in docs:
        source = d.metadata.get("source", "Unknown source")
        page = d.metadata.get("page", "N/A")
        formatted.append(f"[Source: {source} | Page: {page}]\n{d.page_content}")
    return "\n\n---\n\n".join(formatted)


def ask_bot(question: str):
    classifier_chain = OOS_PROMPT | llm | StrOutputParser()
    classification = classifier_chain.invoke({"question": question}).strip().upper()

    if "OUT_OF_SCOPE" in classification:
        return {"answer": REFUSAL_MESSAGE, "sources": []}

    docs = retriever.invoke(question)
    if not docs:
        return {"answer": REFUSAL_MESSAGE, "sources": []}

    context = format_docs(docs)
    chain = RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    sources = sorted(set(d.metadata.get("source", "Unknown source") for d in docs))

    return {"answer": answer, "sources": sources}


# ----------------------------------------------------------------------------
# Chat interface
# ----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm your HR Help Desk assistant. Ask me about leave, WFH, benefits, performance reviews, and other Zyro Dynamics policies.", "sources": []}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("\U0001F4C4 Sources"):
                for s in msg["sources"]:
                    st.write(f"- {os.path.basename(s)}")

user_input = st.chat_input("Ask an HR policy question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input, "sources": []})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Looking through HR policies..."):
            result = ask_bot(user_input)
        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander("\U0001F4C4 Sources"):
                for s in result["sources"]:
                    st.write(f"- {os.path.basename(s)}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })