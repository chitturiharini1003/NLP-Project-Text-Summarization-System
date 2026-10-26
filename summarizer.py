import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

def summarize_text(text, num_sentences=2):
    # Tokenize into sentences
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text

    # Compute TF-IDF scores
    stop_words = list(stopwords.words('english'))
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # Calculate sentence scores based on TF-IDF sum
    sentence_scores = tfidf_matrix.sum(axis=1).A1
    
    # Rank sentences and pick top ones
    ranked_indices = sentence_scores.argsort()[::-1][:num_sentences]
    ranked_indices.sort()  # Keep original order
    
    summary = " ".join([sentences[i] for i in ranked_indices])
    return summary

if __name__ == "__main__":
    sample_text = (
        "Natural language processing is a subfield of linguistics, computer science, and artificial intelligence. "
        "It focuses on the interactions between computers and human language. "
        "Text summarization is one of the important applications of natural language processing. "
        "Extractive summarization selects important sentences directly from the document. "
        "Abstractive summarization generates new sentences to convey the main idea."
    )
    
    print("--- Original Text ---")
    print(sample_text)
    print("\n--- Summary ---")
    print(summarize_text(sample_text, num_sentences=2))