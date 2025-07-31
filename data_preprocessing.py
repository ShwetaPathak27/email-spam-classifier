import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tqdm import tqdm
from multiprocess import Pool, cpu_count
import re

# Download NLTK data (only once needed)
nltk.download('stopwords')

# Initialize
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Define transformation function
def transform_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)        # remove URLs
    text = re.sub(r"[^a-zA-Z]", " ", text)            # remove special characters
    words = text.split()
    cleaned_words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(cleaned_words)

# Use multiprocessing for faster text cleaning
def parallel_apply(texts):
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(transform_text, texts), total=len(texts)))
    return results

# ✅ Required for Windows compatibility
if __name__ == '__main__':
    print("📂 Loading data...")
    df = pd.read_csv('email_spam_dataset.csv')
    
    # Drop missing rows
    df = df.dropna(subset=['label', 'text'])
    
    print("🚀 Cleaning in progress, please wait...")
    df['cleaned'] = parallel_apply(df['text'])

    # Save the cleaned data
    df.to_csv('cleaned_email_dataset.csv', index=False)
    print("✅ Cleaning completed and saved successfully.")
