import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK resources
nltk.download('stopwords', quiet=True)

# 1. Load NLTK English Stopwords Corpus
STOPWORDS = set(stopwords.words('english'))

# 2. Initialize NLTK Porter Stemmer
stemmer = PorterStemmer()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # Lowercase & Tokenize using Regex
    tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    # Remove Stop Words using NLTK Corpus
    tokens = [t for t in tokens if t not in STOPWORDS]
    # Apply Official NLTK Porter Stemmer
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)

print("Reading bbc_news_data.csv...")
df = pd.read_csv('bbc_news_data.csv')

print("Preprocessing Text and Summaries using NLTK...")
df['Cleaned_Text'] = df['Text'].apply(preprocess_text)
if 'Summary' in df.columns:
    df['Cleaned_Summary'] = df['Summary'].apply(preprocess_text)

# Save preprocessed output to CSV and Excel
csv_file = 'bbc_news_preprocessed.csv'
excel_file = 'bbc_news_preprocessed.xlsx'

df.to_csv(csv_file, index=False)
try:
    df.to_excel(excel_file, index=False)
    print(f"\n✅ SUCCESS! Created Excel file: '{excel_file}'")
except Exception as e:
    print(f"\n✅ SUCCESS! Created CSV file: '{csv_file}'")

print("Preprocessing complete!")