import pandas as pd
import numpy as np

import utils

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, KFold

from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, Dense, Flatten, Dropout, LSTM

from gensim.models import Word2Vec

class NaiveBayes():
    def __init__(self, tfidf_matrix: csr_matrix, df, hasAdditionalFeatures: bool = False):
        self.tfidf_matrix = tfidf_matrix
        self.df = df
        self.hasAdditionalFeatures = hasAdditionalFeatures

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.y_pred = None

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

        self.y_pred = model.predict(self.X_test)

        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        self.classification_report = classification_report(self.y_test, self.y_pred)

        return model

    def getScoreMetrics(self)->(float, str):
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
    
    def getTestPrediction(self)->np.ndarray:
        '''Returns the predictions on the test set.

        Input:
            None
        
        Output:
            - y_pred: predictions on the test set.
        
        '''
        return self.y_pred

    def crossValidation(self, model: MultinomialNB, k: int = 5)->(np.ndarray, float, float): # type: ignore
        ''' Performs k-fold cross validation on the model.

        Input:
            - model: trained model.
            - k: number of folds.

        Output:
            - scores: list of scores.
            - mean: mean of the scores.
            - std: standard deviation of the scores.
        '''
        scores = cross_val_score(model, self.tfidf_matrix, self.df['generated'], cv=k, scoring='accuracy')
        mean = np.mean(scores)
        std = np.std(scores)
        print(f"Cross validation scores: {scores}")
        print(f"Mean score: {mean}")
        print(f"Standard deviation: {std}")
        return scores, mean, std
    
    def evaluteModelErrors(self)->pd.DataFrame:
        ''' Evaluates the model errors. 

        Input:
            None

        Output:
            - df_errors: dataframe with the errors.
        '''
        error_df = pd.DataFrame({'Actual': self.y_test, 'Predicted': self.y_pred})
        error_df['Correct'] = error_df['Actual'] == error_df['Predicted']
        return error_df

class TextClassifier:
    def __init__(self, df: pd.DataFrame, word2vec_model: Word2Vec):
        self.df = df
        self.word2vec_model = word2vec_model

        self.tokenizer, self.X_padded = utils.tokenizeText(self.df)
        self.embedding_matrix = utils.createEmbeddingMatrix(self.tokenizer, word2vec_model)

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.y_pred = None

        self.accuracy = None
        self.classification_report = None
    
    

class CNN(TextClassifier):

    def buildModel(self)->Sequential:
        ''' Builds the CNN model.

        Input:
            None
        
        Output:
            - model: CNN model.
        '''
        # Emebedding layer parameters
        vocab_size = len(self.tokenizer.word_index) + 1
        embedding_dim = self.word2vec_model.vector_size
        max_length = self.X_padded.shape[1]

        # Model definition       
        model = Sequential()
        # Add the embedding layer
        model.add(Embedding(vocab_size, embedding_dim, weights=[self.embedding_matrix], input_length=max_length, trainable=False))
        # Add the convolutional layer
        model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
        # Add the pooling layer
        model.add(MaxPooling1D(pool_size=2))
        # Fully connected layer
        model.add(Flatten())
        model.add(Dense(10, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid')) # binary classifier

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model
    
    def train(self)->Sequential:
        ''' Trains the CNN model. The dataset is split into train and test sets. It is important for the DataFrame to have a column named 'generated' with the labels.

        Input:
            None
        
        Output:
            - model: trained model.
        '''
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_padded, self.df['generated'], test_size=0.2, random_state=42)

        # Build the model
        model = self.buildModel()

        # Train the model
        model.fit(self.X_train, self.y_train, epochs=10, validation_data=(self.X_test, self.y_test), batch_size=64)

        return model

    def evaluateModel(self, model: Sequential)->(float, str): # type: ignore
        ''' Evaluates the model.

        Input:
            - model: trained model.
        
        Output:
            - accuracy: accuracy of the model.
            - classification_report: classification report of the model.
        '''
        y_pred_prob = model.predict(self.X_test)
        # Convert the probabilities to binary labels
        self.y_pred = (y_pred_prob > 0.5).astype('int32')

        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        self.classification_report = classification_report(self.y_test, self.y_pred)

        print(f"Accuracy: {self.accuracy}")
        print(f"Classification report:\n{self.classification_report}")

        return self.accuracy, self.classification_report
    
    def crossValidation(self, num_folds: int=5)->[float, float]: # type: ignore
        ''' Performs k-fold cross validation on the model.

        Input:
            - num_folds: number of folds.
        
        Output:
            - mean_accuracy: mean accuracy across all folds.
            - std_accuracy: standard deviation of the accuracies.
        '''
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

        scores_list = []
        fold_no = 1
        for train, test in kfold.split(self.X_padded, self.df['generated']):
            # Build the model
            model = self.buildModel()
            
            # Select data for this fold
            X_train_fold, X_test_fold = self.X_padded[train], self.X_padded[test]
            y_train_fold, y_test_fold = self.df['generated'].iloc[train], self.df['generated'].iloc[test]

            print(f"Training for fold {fold_no}...")

            # Train the model
            model.fit(X_train_fold, y_train_fold, epochs=10, validation_data=(X_test_fold, y_test_fold), batch_size=64)

            # Evaluate the model
            scores = model.evaluate(X_test_fold, y_test_fold, verbose=0)
            scores_list.append(scores[1])
            print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')

            fold_no += 1
        
        mean_accuracy = np.mean(scores_list)
        std_accuracy = np.std(scores_list)
        print(f"Mean accuracy: {mean_accuracy}")
        print(f"Standard deviation: {std_accuracy}")

        return mean_accuracy, std_accuracy

    
    
class LSTM(TextClassifier):
    def buildModel(self)->Sequential:
        ''' Builds the CNN model.

        Input:
            None
        
        Output:
            - model: CNN model.
        '''
        # Emebedding layer parameters
        vocab_size = len(self.tokenizer.word_index) + 1
        embedding_dim = self.word2vec_model.vector_size
        max_length = self.X_padded.shape[1]

        # Model definition
        model = Sequential()
        # Add the embedding layer
        model.add(Embedding(vocab_size, embedding_dim, weights=[self.embedding_matrix], input_length=max_length, trainable=False))
        # Add the LSTM layer
        model.add(LSTM(64))
        # Fully connected layer
        model.add(Dense(10, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model
    
    def train(self)->Sequential:
        ''' Trains the CNN model. The dataset is split into train and test sets. It is important for the DataFrame to have a column named 'generated' with the labels.

        Input:
            None
        
        Output:
            - model: trained model.
        '''
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_padded, self.df['generated'], test_size=0.2, random_state=42)

        # Build the model
        model = self.buildModel()

        # Train the model
        model.fit(self.X_train, self.y_train, epochs=10, validation_data=(self.X_test, self.y_test), batch_size=64)

        return model

    def evaluateModel(self, model: Sequential)->(float, str): # type: ignore
        ''' Evaluates the model.

        Input:
            - model: trained model.
        
        Output:
            - accuracy: accuracy of the model.
            - classification_report: classification report of the model.
        '''
        y_pred_prob = model.predict(self.X_test)
        # Convert the probabilities to binary labels
        self.y_pred = (y_pred_prob > 0.5).astype('int32')

        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        self.classification_report = classification_report(self.y_test, self.y_pred)

        print(f"Accuracy: {self.accuracy}")
        print(f"Classification report:\n{self.classification_report}")

        return self.accuracy, self.classification_report
    
    def crossValidation(self, num_folds: int=5)->[float, float]: # type: ignore
        ''' Performs k-fold cross validation on the model.

        Input:
            - num_folds: number of folds.
        
        Output:
            - mean_accuracy: mean accuracy across all folds.
            - std_accuracy: standard deviation of the accuracies.
        '''
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

        scores_list = []
        fold_no = 1
        for train, test in kfold.split(self.X_padded, self.df['generated']):
            # Build the model
            model = self.buildModel()
            
            # Select data for this fold
            X_train_fold, X_test_fold = self.X_padded[train], self.X_padded[test]
            y_train_fold, y_test_fold = self.df['generated'].iloc[train], self.df['generated'].iloc[test]

            print(f"Training for fold {fold_no}...")

            # Train the model
            model.fit(X_train_fold, y_train_fold, epochs=10, validation_data=(X_test_fold, y_test_fold), batch_size=64)

            # Evaluate the model
            scores = model.evaluate(X_test_fold, y_test_fold, verbose=0)
            scores_list.append(scores[1])
            print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')

            fold_no += 1
        
        mean_accuracy = np.mean(scores_list)
        std_accuracy = np.std(scores_list)
        print(f"Mean accuracy: {mean_accuracy}")
        print(f"Standard deviation: {std_accuracy}")

        return mean_accuracy, std_accuracy
