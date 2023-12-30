import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def donwload_nltk_resources(resource_name: str):
    try:
        # Check of the resource has already been downloaded
        nltk.data.find(resource_name)
    except LookupError:
        # If not, download the resource
        nltk.download(resource_name)

def checkForNLTKResources():
    # Resources to download
    resources = ['punkt', 'stopwords', 'wordnet']

    for resource in resources:
        donwload_nltk_resources(resource)

def cleanText(text: str)-> str:
    # Put text to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Remove digits
    text = re.sub(r'\d+', '', text)
    # Tokenization
    words = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    # Join words
    text = ' '.join(words)
    return text