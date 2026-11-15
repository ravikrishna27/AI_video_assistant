
import os
import time
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


# ============================================================
# 1. PAGE CONFIGURATION AND ENVIRONMENT VARIABLES
# ============================================================

st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()

# Load Streamlit Cloud secrets before importing backend modules.
# This is important because transcriber.py reads its API key
# when the module is imported.

try:
    for key in [
        "MISTRAL_API_KEY",
        "SARVAM_API_KEY",
        "WHISPER_MODEL",
        "SARVAM_STT_MODEL"
    ]:
        if key in st.secrets:
            os.environ[key] = str(st.secrets[key])
except Exception:
    pass


# ============================================================
# 2. EXISTING BACKEND IMPORTS — PRESERVED
# ============================================================

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions
)
from core.rag_engine import build_rag_chain, ask_question


# ============================================================
# 3. PREMIUM UI STYLING
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --background: #09090f;
    --surface: #12121c;
    --surface-light: #191927;
    --border: #29293b;
    --purple: #8b5cf6;
    --cyan: #22d3ee;
    --text: #f4f4f8;
    --muted: #9999ad;
}

.stApp {
    background: var(--background);
    color: var(--text);
    font-family: 'Inter', sans-serif;
}

[data-testid="stSidebar"] {
    background: #10101a;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3 {
    color: var(--text) !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
}

p, li {
    color: #d1d1df;
    line-height: 1.7;
}

.hero {
    padding: 25px 0 20px 0;
}

.hero-label {
    color: var(--cyan);
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.hero-title {
    font-size: clamp(30px, 5vw, 48px);
    font-weight: 800;
    letter-spacing: -2px;
    line-height: 1.15;
    background: linear-gradient(100deg, #ffffff, #a78bfa, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-description {
    color: var(--muted);
    font-size: 15px;
    margin-top: 12px;
}

.section-card {
    background: linear-gradient(145deg, #171725, #11111b);
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 22px;
    margin-bottom: 15px;
}

.section-label {
    color: #a78bfa;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.feature-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 13px;
    padding: 18px;
    min-height: 125px;
}

.feature-icon {
    font-size: 23px;
    margin-bottom: 12px;
}

.feature-title {
    color: #f4f4f8;
    font-size: 15px;
    font-weight: 700;
}

.feature-description {
    color: var(--muted);
    font-size: 12px;
    margin-top: 6px;
}

.stButton > button {
    background: linear-gradient(110deg, #7c3aed, #5b21b6);
    color: white;
    border: none;
    border-radius: 9px;
    font-weight: 700;
    min-height: 43px;
    transition: 0.2s ease;
}

.stButton > button:hover {
    border: 1px solid #a78bfa;
    box-shadow: 0 5px 20px #7c3aed44;
}

.stTextInput input,
.stSelectbox div[data-baseweb="select"] > div {
    background: #191927;
    border-color: var(--border);
    border-radius: 8px;
    color: white;
}

.stProgress > div > div > div {
    background: linear-gradient(90deg, #8b5cf6, #22d3ee);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    border-bottom: 1px solid var(--border);
}

.stTabs [data-baseweb="tab"] {
    color: var(--muted);
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    color: #a78bfa !important;
}

div[data-testid="stChatMessage"] {
    background: #151522;
    border: 1px solid var(--border);
    border-radius: 12px;
}

.stDownloadButton button {
    background: #191927;
    color: white;
    border: 1px solid var(--border);
    border-radius: 8px;
}

hr {
    border-color: var(--border);
}

.small-muted {
    color: var(--muted);
    font-size: 12px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 4. SESSION STATE
# ============================================================

defaults = {
    "result": None,
    "chat_history": [],
    "pipeline_steps": {},
    "pipeline_done": False
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# 5. SIDEBAR — INPUT CONTROLS
# ============================================================

with st.sidebar:

    st.markdown("## 🎬 Video Intelligence")

    st.caption("AI-powered meeting analysis")

    st.divider()

    st.markdown("### Input Source")

    input_type = st.radio(
        "Choose your source",
        ["YouTube URL", "Upload Audio / Video"],
        label_visibility="collapsed"
    )

    source = None
    uploaded_file = None

    if input_type == "YouTube URL":

        source = st.text_input(
            "YouTube URL",
            placeholder="Paste your YouTube link here"
        )

    else:

        uploaded_file = st.file_uploader(
            "Upload your media file",
            type=[
                "mp3", "wav", "m4a",
                "mp4", "mov", "mkv", "webm"
            ]
        )

    st.markdown("### Transcription Language")

    language = st.selectbox(
        "Select language",
        ["english", "hinglish"],
        label_visibility="collapsed"
    )

    st.markdown("")

    run_btn = st.button(
        "⚡ Analyse Media",
        use_container_width=True
    )

    st.divider()

    st.markdown("### What this app does")

    st.markdown("""
    - 🎧 Audio extraction
    - 📝 Speech transcription
    - 📋 AI summarization
    - ✅ Action item extraction
    - 🔑 Key decision extraction
    - 💬 RAG-powered Q&A
    """)

    st.divider()

    st.markdown(
        '<p class="small-muted">AI Video Assistant · Meeting Intelligence</p>',
        unsafe_allow_html=True
    )


# ============================================================
# 6. MAIN HERO SECTION
# ============================================================

st.markdown("""
<div class="hero">
    <div class="hero-label">AI-powered productivity</div>
    <div class="hero-title">AI Video Assistant</div>
    <div class="hero-description">
        Transform long videos and meetings into structured,
        searchable knowledge.
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()


# ============================================================
# 7. FEATURE CARDS
# ============================================================

feature_columns = st.columns(4)

features = [
    ("🎧", "Transcription", "Convert speech into text"),
    ("📋", "Summarization", "Get concise meeting summaries"),
    ("🎯", "Insights", "Extract decisions and tasks"),
    ("🧠", "Ask AI", "Chat with your transcript")
]

for column, feature in zip(feature_columns, features):

    icon, title, description = feature

    with column:
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-icon">{icon}</div>
            <div class="feature-title">{title}</div>
            <div class="feature-description">{description}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("")


# ============================================================
# 8. PIPELINE EXECUTION
# ============================================================

if run_btn:

    if input_type == "YouTube URL" and (
        not source or not source.strip()
    ):
        st.error("Please enter a valid YouTube URL.")

    elif input_type == "Upload Audio / Video" and uploaded_file is None:
        st.error("Please upload an audio or video file.")

    else:

        temporary_upload_path = None

        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.pipeline_steps = {}
        st.session_state.pipeline_done = False

        progress = st.progress(0)
        status = st.empty()

        stages = [
            "Processing audio/video",
            "Transcribing audio",
            "Generating title",
            "Creating summary",
            "Extracting insights",
            "Building RAG engine"
        ]

        def update_stage(index, message):

            status.info(f"⏳ {message}")

            st.session_state.pipeline_steps[stages[index]] = "completed"

            progress.progress((index + 1) / len(stages))

        try:

            # Save uploaded file temporarily.
            if input_type == "Upload Audio / Video":

                suffix = Path(uploaded_file.name).suffix

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as temp_file:

                    temp_file.write(uploaded_file.getbuffer())
                    temporary_upload_path = temp_file.name

                media_source = temporary_upload_path

            else:

                media_source = source.strip()

            # STEP 1: Audio Processing
            status.info("⏳ Processing audio/video...")

            chunks = process_input(media_source)

            update_stage(0, "Audio/video processing completed")

            # STEP 2: Transcription
            status.info("⏳ Transcribing audio...")

            transcript = transcribe_all(chunks, language)

            if not transcript or not transcript.strip():
                raise ValueError("No transcript was generated.")

            update_stage(1, "Transcription completed")

            # STEP 3: Title Generation
            status.info("⏳ Generating title...")

            title = generate_title(transcript)

            update_stage(2, "Title generation completed")

            # STEP 4: Summarization
            status.info("⏳ Generating summary...")

            summary = summarize(transcript)

            update_stage(3, "Summarization completed")

            # STEP 5: Information Extraction
            status.info("⏳ Extracting meeting insights...")

            action_items = extract_action_items(transcript)

            decisions = extract_key_decisions(transcript)

            questions = extract_questions(transcript)

            update_stage(4, "Information extraction completed")

            # STEP 6: RAG Engine
            status.info("⏳ Building RAG engine...")

            rag_chain = build_rag_chain(transcript)

            update_stage(5, "RAG engine ready")

            # Store results in Streamlit session state.
            st.session_state.result = {
                "title": title,
                "transcript": transcript,
                "summary": summary,
                "action_items": action_items,
                "key_decisions": decisions,
                "open_questions": questions,
                "rag_chain": rag_chain
            }

            st.session_state.pipeline_done = True

            status.success("✅ Analysis completed successfully!")

            time.sleep(1)

            st.rerun()

        except Exception as e:

            status.error(f"Something went wrong: {str(e)}")

        finally:

            if temporary_upload_path and os.path.exists(
                temporary_upload_path
            ):
                os.remove(temporary_upload_path)


# ============================================================
# 9. DISPLAY RESULTS
# ============================================================

if st.session_state.result:

    result = st.session_state.result

    st.divider()

    st.markdown("## 📊 Analysis Results")

    st.markdown(
        f'<div class="section-card">'
        f'<div class="section-label">Generated Session Title</div>'
        f'<h3>{result["title"]}</h3>'
        f'</div>',
        unsafe_allow_html=True
    )

    # Result tabs
    overview_tab, transcript_tab, insights_tab, chat_tab = st.tabs([
        "📊 Overview",
        "📝 Transcript",
        "🎯 Meeting Insights",
        "💬 Ask AI"
    ])

    # --------------------------------------------------------
    # OVERVIEW TAB
    # --------------------------------------------------------

    with overview_tab:

        st.markdown("### Executive Summary")

        st.markdown(
            '<div class="section-card">',
            unsafe_allow_html=True
        )

        st.write(result["summary"])

        st.markdown("</div>", unsafe_allow_html=True)

        st.download_button(
            "⬇ Download Summary",
            data=str(result["summary"]),
            file_name="meeting_summary.txt",
            mime="text/plain"
        )

    # --------------------------------------------------------
    # TRANSCRIPT TAB
    # --------------------------------------------------------

    with transcript_tab:

        st.markdown("### Full Transcript")

        st.caption("Generated from your uploaded media.")

        st.text_area(
            "Transcript",
            value=result["transcript"],
            height=400,
            label_visibility="collapsed"
        )

        st.download_button(
            "⬇ Download Transcript",
            data=result["transcript"],
            file_name="meeting_transcript.txt",
            mime="text/plain"
        )

    # --------------------------------------------------------
    # MEETING INSIGHTS TAB
    # --------------------------------------------------------

    with insights_tab:

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown("### ✅ Action Items")
            st.write(result["action_items"])

        with col2:

            st.markdown("### 🔑 Key Decisions")
            st.write(result["key_decisions"])

        with col3:

            st.markdown("### ❓ Open Questions")
            st.write(result["open_questions"])

        # Download all extracted insights.
        insights_text = f"""
ACTION ITEMS
{result["action_items"]}

KEY DECISIONS
{result["key_decisions"]}

OPEN QUESTIONS
{result["open_questions"]}
"""

        st.download_button(
            "⬇ Download Meeting Insights",
            data=insights_text,
            file_name="meeting_insights.txt",
            mime="text/plain"
        )

    # --------------------------------------------------------
    # RAG CHAT TAB
    # --------------------------------------------------------

    with chat_tab:

        st.markdown("### 💬 Chat with your Meeting")

        st.caption(
            "Ask questions and get answers based on your transcript."
        )

        # Display existing chat messages.
        for message in st.session_state.chat_history:

            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Streamlit chat input.
        user_question = st.chat_input(
            "Ask something about your meeting..."
        )

        if user_question:

            st.session_state.chat_history.append({
                "role": "user",
                "content": user_question
            })

            with st.chat_message("user"):
                st.markdown(user_question)

            with st.chat_message("assistant"):

                with st.spinner("Thinking..."):

                    try:

                        answer = ask_question(
                            result["rag_chain"],
                            user_question
                        )

                        st.markdown(answer)

                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer
                        })

                    except Exception as e:

                        st.error(f"Unable to generate an answer: {e}")

            st.rerun()

        if st.session_state.chat_history:

            if st.button("🗑 Clear Conversation"):

                st.session_state.chat_history = []
                st.rerun()


# ============================================================
# 10. EMPTY STATE
# ============================================================

else:

    st.markdown("")

    st.markdown("""
    <div class="section-card" style="text-align:center;padding:55px 20px;">

        <div style="font-size:48px;margin-bottom:15px;">🎬</div>

        <h2 style="margin-bottom:10px;">
            Ready to transform your videos?
        </h2>

        <p style="color:#9999ad;">
            Upload an audio/video file or paste a YouTube URL
            using the sidebar to get started.
        </p>

        <p style="color:#8b5cf6;font-size:13px;margin-top:20px;">
            TRANSCRIBE &nbsp; • &nbsp;
            SUMMARIZE &nbsp; • &nbsp;
            EXTRACT &nbsp; • &nbsp;
            ASK AI
        </p>

    </div>
    """, unsafe_allow_html=True)
