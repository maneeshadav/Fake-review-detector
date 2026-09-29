import re
import nltk
from nltk.corpus import stopwords

# Ensure NLTK stopwords are downloaded silently
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Clean and preprocess raw text input for fake review detection:
    1. Convert to lowercase
    2. Remove HTML tags and URLs
    3. Remove punctuation and special characters
    4. Tokenize by whitespace
    5. Remove English stopwords
    6. Remove extra whitespace
    """
    if not isinstance(text, str) or not text:
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove HTML tags & URLs
    text = re.sub(r'http\S+|www\S+|<.*?>', ' ', text)
    
    # 3. Remove punctuation and special characters (keep alphanumeric and spaces)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # 4. Tokenize by splitting into words
    tokens = text.split()
    
    # 5. Remove stopwords and single-character tokens
    tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
    
    # 6. Join back into clean string
    return " ".join(tokens)

if __name__ == "__main__":
    sample = "This PRODUCT arrived on time & worked EXACTLY as described! http://example.com Great buy."
    print("Original:", sample)
    print("Cleaned: ", clean_text(sample))
