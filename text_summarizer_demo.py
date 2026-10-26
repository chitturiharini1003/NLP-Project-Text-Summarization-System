# =====================================================================
# PROJECT 4: AUTOMATIC TEXT SUMMARIZATION SYSTEM
# NLP Concepts: Tokenization (CO2), TF-IDF Feature Extraction (CO3/CO4), Summarization (CO5)
# =====================================================================

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------------------
# MODULE 1 & 2: Text Input & Sentence Tokenization (CO2)
# ---------------------------------------------------------------------
sample_article = """
US consumer confidence rose in December for the first time since July.
Consumers' confidence in the state of the US economy is at its highest level for five months.
The feel-good factor among US consumers rose due to strong post-Christmas retail sales.
US retailers have reported strong sales over the past 10 days after a slow start.
"""

print("--- MODULE 1: RAW INPUT TEXT ---")
print(sample_article.strip())

sentences = sent_tokenize(sample_article)
print(f"\n--- MODULE 2: SENTENCE TOKENIZATION (Total Sentences: {len(sentences)}) ---")
for idx, s in enumerate(sentences):
    print(f"Sentence [{idx}]: {s}")

# ---------------------------------------------------------------------
# MODULE 3: Feature Engineering - Stop Word Removal & TF-IDF Scoring (CO3 & CO4)
# ---------------------------------------------------------------------
stop_words = list(stopwords.words('english'))
vectorizer = TfidfVectorizer(stop_words=stop_words)

# Compute TF-IDF Matrix across sentences
tfidf_matrix = vectorizer.fit_transform(sentences)
sentence_scores = tfidf_matrix.sum(axis=1).A1  # Sum TF-IDF weights per sentence

print("\n--- MODULE 3: TF-IDF SENTENCE SCORES ---")
for idx, score in enumerate(sentence_scores):
    print(f"Sentence [{idx}] TF-IDF Importance Score: {score:.4f}")

# ---------------------------------------------------------------------
# MODULE 4 & 5: Important Sentence Extraction & Summary Generation (CO5)
# ---------------------------------------------------------------------
N_SENTENCES = 2
top_indices = sentence_scores.argsort()[::-1][:N_SENTENCES]
top_indices.sort()  # Maintain chronological order in summary

final_summary = " ".join([sentences[i] for i in top_indices])

print("\n--- MODULE 4 & 5: GENERATED EXTRACTIVE SUMMARY ---")
print(final_summary)