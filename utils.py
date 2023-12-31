import re
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

from textblob import TextBlob

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

def cleanText(text: str, withStemming = False)-> str:
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
    # Stemming/Lemmatization
    if withStemming:
        # Stemming
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]
    else:
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
    # Join words
    text = ' '.join(words)
    return text

def computeNumericalFeatures(df: pd.DataFrame):
    ''' Receives a dataframe with a column named 'text' (already cleaned) 
    and computes the following features: number of words, number of characters,
    average word length, number of sentences and sentiment score.

    Input:
        - df: dataframe with a column named 'text' (already cleaned).
    
    Output:
        - df: dataframe with the new features.
    '''
    # Number of words
    df['num_words'] = df['text'].apply(lambda x: len(str(x).split()))
    # Character count
    df['num_chars'] = df['text'].apply(lambda x: len(str(x)))
    # Average word length
    df['avg_word_length'] = df['num_chars'] / df['num_words']
    # Sentence count
    df['num_sentences'] = df['text'].apply(lambda x: len(TextBlob(x).sentences))
    # Sentiment score
    df['sentiment_score'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)

    return df