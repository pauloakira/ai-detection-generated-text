import re
import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

from textblob import TextBlob

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from scipy.stats import kstest, norm

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

def cleanText(text: str, withStemming: bool = False)-> str:
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

def computeNumericalFeatures(df: pd.DataFrame)-> pd.DataFrame:
    ''' Receives a dataframe with a column named 'cleaned_text' (already cleaned) 
    and computes the following features: number of words, number of characters,
    average word length, number of sentences and sentiment score.

    Input:
        - df: dataframe with a column named 'text' (already cleaned).
    
    Output:
        - df: dataframe with the new features.
    '''
    # Number of words
    df['num_words'] = df['cleaned_text'].apply(lambda x: len(str(x).split()))
    # Character count
    df['num_chars'] = df['cleaned_text'].apply(lambda x: len(str(x)))
    # Average word length
    df['avg_word_length'] = df['num_chars'] / df['num_words']
    # Sentence count
    df['num_sentences'] = df['cleaned_text'].apply(lambda x: len(TextBlob(x).sentences))
    # Sentiment score
    df['sentiment_score'] = df['cleaned_text'].apply(lambda x: TextBlob(x).sentiment.polarity)

    return df

def tfidfVectorizer(df: pd.DataFrame, hasAdditionalFeatures: bool = False)-> csr_matrix:
    ''' Receives a dataframe with a column named 'cleaned_text' (already cleaned)
    and computes the tfidf matrix. If hasAdditionalFeatures is True, it also adds
    the following features: number of words, number of characters, average word length,
    number of sentences and sentiment score.
    
    Input:
        - df: dataframe with a column named 'text' (already cleaned).
        - hasAdditionalFeatures: boolean indicating if additional features should be added.

    Output:
        - tfidf_matrix: tfidf sparse matrix.
    '''
    # Create tfidf vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=1)
    # Fit the vectorizer
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['cleaned_text'])
    # Add additional features
    if hasAdditionalFeatures:
        # Convert the numerical features to a sparse matrix
        additional_features = csr_matrix(df[['num_words', 'num_chars', 'avg_word_length', 'num_sentences', 'sentiment_score']].values)
        # Horizontally stack the sparse matrix with the tfidf matrix
        tfidf_matrix = hstack([tfidf_matrix, additional_features])
    return tfidf_matrix

def computeKSTest(df: pd.DataFrame, column: str, alpha: float = 0.05)-> (float, float):
    ''' Computes the Kolmogorov-Smirnov test for a given column of a dataframe.
    
    Input:
        - df: dataframe.
        - column: column of the dataframe to compute the test.
    
    Output:
        - D: KS statistic.
        - p_value: p-value.
    '''
    # Compute the KS test
    data = df[column]
    DStat, p_value = kstest(data, 'norm', (np.mean(data), np.std(data)))
    if p_value > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')
    return DStat, p_value