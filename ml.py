import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

class NaiveBayes():
    def __init__(self, tfidf_matrix: csr_matrix, df, hasAdditionalFeatures: bool = False):
        self.tfidf_matrix = tfidf_matrix
        self.df = df
        self.hasAdditionalFeatures = hasAdditionalFeatures

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.accuracy = None
        self.classification_report = None

    def train_multinomial(self, test_size: float = 0.2, random_state: int = 42)->MultinomialNB:
        ''' Trains a Multinomial Naive Bayes classifier. The dataset is split into train and test sets. It is important for the DataFrame to have a column named 'generated' with the labels.
        
        Input:
            - test_size: proportion of the dataset to include in the test split.
        
        Output:
            - model: trained model.
        '''

        # Split the dataset into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.tfidf_matrix, self.df['generated'], test_size=test_size, random_state=random_state)

        # Define and train the model
        model = MultinomialNB()
        model.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_test)
        self.accuracy = accuracy_score(self.y_test, y_pred)
        self.classification_report = classification_report(self.y_test, y_pred)

        return model

    def getScoreMetrics(self):
        ''' Returns the accuracy and the classification report.

        Input:
            None

        Output:
            - accuracy: accuracy of the model.
            - classification_report: classification report of the model.
        '''
        print(f"Accuracy: {self.accuracy}")
        print(f"Classification report:\n{self.classification_report}")
        return self.accuracy, self.classification_report     

