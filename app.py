import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure NLTK data is downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Text Summarization System",
    page_icon="📝",
    layout="wide"
)

# High-Contrast CSS Customizations
st.markdown("""
    <style>
    /* Main Page Background */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Section Headers */
    .section-tag {
        background-color: #1e3a8a;
        color: #ffffff !important;
        padding: 8px 16px;
        border-radius: 6px;
        font-size: 15px !important;
        font-weight: 700 !important;
        display: inline-block;
        margin-bottom: 15px;
    }
    
    /* Display Content Box for Dataset Articles */
    .article-display-box {
        background-color: #ffffff;
        border: 2px solid #cbd5e1;
        border-radius: 8px;
        padding: 16px;
        height: 240px;
        overflow-y: auto;
        color: #0f172a !important;
        font-size: 15px !important;
        line-height: 1.6;
        font-weight: 500;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.05);
    }
    
    /* Output Summary Box */
    .summary-box {
        background-color: #eff6ff;
        border-left: 6px solid #2563eb;
        padding: 20px;
        border-radius: 8px;
        font-size: 16px !important;
        line-height: 1.7;
        color: #1e293b !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }

    /* Labels & Input Visibility */
    label, p, span, div {
        color: #0f172a !important;
        font-size: 16px !important;
        font-weight: 700 !important;
    }

    /* Primary Action Buttons */
    .stButton>button {
        background-color: #2563eb !important;
        color: #ffffff !important;
        font-size: 18px !important;
        font-weight: 800 !important;
        padding: 12px 28px !important;
        border-radius: 8px !important;
        border: none !important;
        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.3);
    }
    .stButton>button:hover {
        background-color: #1d4ed8 !important;
    }
    
    /* Hide Default Footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Core NLP Summarization Logic
# ---------------------------------------------------------
def summarize_text(text, num_sentences=3):
    if not text or not isinstance(text, str):
        return ""
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text

    stop_words = list(stopwords.words('english'))
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
        sentence_scores = tfidf_matrix.sum(axis=1).A1
        ranked_indices = sentence_scores.argsort()[::-1][:num_sentences]
        ranked_indices.sort()
        return " ".join([sentences[i] for i in ranked_indices])
    except Exception:
        return " ".join(sentences[:num_sentences])

# ---------------------------------------------------------
# Top Banner / Header (HIGH CONTRAST BRIGHT WHITE BOX)
# ---------------------------------------------------------
st.markdown("""
    <div style="background-color: #ffffff; padding: 30px; border-radius: 12px; border: 3px solid #2563eb; text-align: center; margin-bottom: 25px; box-shadow: 0 4px 10px rgba(0,0,0,0.08);">
        <h1 style="color: #0f172a !important; font-size: 42px !important; font-weight: 900 !important; margin: 0 0 8px 0 !important; font-family: sans-serif; letter-spacing: 1px;">
            📝 AUTOMATIC TEXT SUMMARIZATION SYSTEM
        </h1>
        <p style="color: #2563eb !important; font-size: 20px !important; font-weight: 700 !important; margin: 0 !important;">
            Extractive Natural Language Processing Engine using TF-IDF Vectorization
        </p>
    </div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------
tab_custom, tab_dataset = st.tabs(["📌 Live Custom Summarizer", "📊 BBC Dataset Evaluation"])

# ---------------------------------------------------------
# TAB 1: Live Custom Summarizer
# ---------------------------------------------------------
with tab_custom:
    col_left, col_right = st.columns([1, 1], gap="large")
    
    with col_left:
        st.markdown('<span class="section-tag">STEP 1: INPUT TEXT</span>', unsafe_allow_html=True)
        user_text = st.text_area(
            "Paste your long article or document text below:",
            placeholder="Paste your long news article, essay, or raw document text here...",
            height=260
        )
        num_sentences = st.slider("Select Summary Length (Target Sentences):", min_value=1, max_value=8, value=3)
        run_btn = st.button("⚡ Generate Summary", key="btn_custom_gen")

    with col_right:
        st.markdown('<span class="section-tag">STEP 2: EXTRACTED KEY INSIGHTS</span>', unsafe_allow_html=True)
        if run_btn:
            if user_text.strip():
                generated_summary = summarize_text(user_text, num_sentences=num_sentences)
                st.markdown(f'<div class="summary-box">{generated_summary}</div>', unsafe_allow_html=True)
                
                orig_words = len(user_text.split())
                sum_words = len(generated_summary.split())
                reduction = round((1 - (sum_words / orig_words)) * 100, 1) if orig_words > 0 else 0
                
                st.write("")
                st.markdown("### 📈 Text Compression Analytics")
                m1, m2, m3 = st.columns(3)
                m1.metric("Original Length", f"{orig_words} words")
                m2.metric("Summary Length", f"{sum_words} words")
                m3.metric("Space Saved", f"{reduction}%")
            else:
                st.warning("Please paste or enter some text first.")
        else:
            st.info("Paste text on the left and click **Generate Summary** to see results.")

# ---------------------------------------------------------
# TAB 2: BBC Dataset Evaluation
# ---------------------------------------------------------
with tab_dataset:
    st.markdown('<span class="section-tag">DATASET GROUND-TRUTH TESTING</span>', unsafe_allow_html=True)
    
    try:
        df = pd.read_csv('bbc_news_data.csv')
        max_idx = len(df) - 1
        
        # Article Selection Input
        col_select, col_empty = st.columns([1, 1])
        with col_select:
            article_idx = st.number_input(
                f"Enter Article Index Number (0 to {max_idx}):",
                min_value=0,
                max_value=max_idx,
                value=0,
                step=1
            )
            
        # Dynamically fetch article based on current typed index
        selected_article = str(df.iloc[article_idx]['Text'])
        reference_summary = str(df.iloc[article_idx]['Summary'])
        
        col_art, col_ref = st.columns([1, 1], gap="large")
        
        with col_art:
            st.markdown(f"**Original Dataset Article (Index #{article_idx}):**")
            st.markdown(f'<div class="article-display-box">{selected_article}</div>', unsafe_allow_html=True)
            
        with col_ref:
            st.markdown(f"**Human Reference Summary (Ground Truth):**")
            st.markdown(f'<div class="article-display-box">{reference_summary}</div>', unsafe_allow_html=True)
            
        st.write("")
        eval_btn = st.button("⚡ Extract & Compare AI Summary", key=f"btn_eval_{article_idx}")
        
        if eval_btn:
            gen_sum = summarize_text(selected_article, num_sentences=3)
            
            st.markdown("---")
            st.markdown("### 🤖 System Generated AI Summary")
            st.markdown(f'<div class="summary-box">{gen_sum}</div>', unsafe_allow_html=True)
            
            # Simple, clear, human-understandable metrics
            orig_words = len(selected_article.split())
            ref_words_list = reference_summary.lower().split()
            gen_words_list = gen_sum.lower().split()
            
            ref_words_set = set(ref_words_list)
            gen_words_set = set(gen_words_list)
            
            overlap_count = len(ref_words_set.intersection(gen_words_set))
            overlap_pct = round((overlap_count / len(ref_words_set)) * 100, 1) if ref_words_set else 0
            
            st.markdown("---")
            st.markdown("### 📊 Performance Metrics")
            k1, k2, k3 = st.columns(3)
            k1.metric("Original Article Words", f"{orig_words}")
            k2.metric("AI Summary Words", f"{len(gen_words_list)}")
            k3.metric("Key Vocabulary Overlap", f"{overlap_pct}% Match")
            
    except Exception as e:
        st.error(f"Error loading dataset file `bbc_news_data.csv`: {e}")