# youtube_summarizer.py

import os
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.chat_models import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv

# ─── Load environment variables from .env (if you use one) ────────────────────
load_dotenv()  # pip install python-dotenv

# ─── Configuration ─────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME   = "mixtral-8x7b-32768"  # or llama3-8b-8192 for faster/cheaper

if not GROQ_API_KEY:
    st.error("🚨 Please set your GROQ_API_KEY environment variable and restart.")
    st.stop()

# ─── Helpers ───────────────────────────────────────────────────────────────────
def extract_video_id(url: str) -> str | None:
    """Extracts the 11‑char YouTube video ID from a URL."""
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    return None

def fetch_transcript(video_id: str) -> str | None:
    """Fetches the auto‑caption transcript via the YouTube Transcript API."""
    try:
        raw = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([seg["text"] for seg in raw])
    except Exception:
        return None

def summarize_text(text: str) -> str:
    """Splits the transcript into chunks and runs a map_reduce summarization chain."""
    # 1) chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.create_documents([text])

    # 2) init Groq LLM
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=MODEL_NAME,
        temperature=0.3
    )

    # 3) load & run summarize chain
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    return chain.run(docs)

# ─── Streamlit UI ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="YouTube Summarizer", layout="wide")
st.title("🎥 YouTube Video Summarizer (LangChain + ChatGroq)")

url = st.text_input("Enter a YouTube video URL", placeholder="https://www.youtube.com/watch?v=...")
if st.button("Generate Summary"):
    vid = extract_video_id(url)
    if not vid:
        st.error("❌ Invalid YouTube URL.")
    else:
        with st.spinner("⏳ Fetching transcript..."):
            transcript = fetch_transcript(vid)

        if not transcript:
            st.error("❌ Transcript not available for this video.")
        else:
            with st.spinner("🤖 Summarizing via ChatGroq..."):
                summary = summarize_text(transcript)
            st.subheader("📝 Summary")
            st.write(summary)
