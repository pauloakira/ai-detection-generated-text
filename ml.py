import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def donwload_nltk_resources(resource_name):
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