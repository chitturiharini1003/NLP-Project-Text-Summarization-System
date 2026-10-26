import pandas as pd
from rouge_score import rouge_scorer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

def summarize_text(text, num_sentences=3):
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text

    stop_words = list(stopwords.words('english'))
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    sentence_scores = tfidf_matrix.sum(axis=1).A1
    ranked_indices = sentence_scores.argsort()[::-1][:num_sentences]
    ranked_indices.sort()
    
    summary = " ".join([sentences[i] for i in ranked_indices])
    return summary

# 1. Load the BBC News dataset
print("Loading dataset...")
df = pd.read_csv('bbc_news_data.csv')
print(f"Dataset loaded! Total articles: {len(df)}")

# 2. Select the first article from the dataset
sample_row = df.iloc[0]
original_text = sample_row['Text']
reference_summary = sample_row['Summary']

# 3. Generate summary using our TF-IDF model
generated_summary = summarize_text(original_text, num_sentences=3)

print("\n--- ORIGINAL TEXT (First 300 chars) ---")
print(original_text[:300] + "...")

print("\n--- GENERATED TF-IDF SUMMARY ---")
print(generated_summary)

print("\n--- REFERENCE SUMMARY (Dataset) ---")
print(reference_summary[:300] + "...")

# 4. Calculate ROUGE evaluation metrics
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)

print("\n--- ROUGE EVALUATION METRICS ---")
print(f"ROUGE-1 Precision: {scores['rouge1'].precision:.4f} | Recall: {scores['rouge1'].recall:.4f} | F1-Score: {scores['rouge1'].fmeasure:.4f}")
print(f"ROUGE-2 Precision: {scores['rouge2'].precision:.4f} | Recall: {scores['rouge2'].recall:.4f} | F1-Score: {scores['rouge2'].fmeasure:.4f}")
print(f"ROUGE-L Precision: {scores['rougeL'].precision:.4f} | Recall: {scores['rougeL'].recall:.4f} | F1-Score: {scores['rougeL'].fmeasure:.4f}")