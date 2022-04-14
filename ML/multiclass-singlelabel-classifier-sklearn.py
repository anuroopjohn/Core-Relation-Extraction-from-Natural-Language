import parameters
import pandas as pd
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from  sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import re
import nltk
from sklearn.datasets import load_files
import pickle
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC , SVC
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import GridSearchCV
import time as time
import matplotlib.pyplot as plt                        # For plotting data
#import seaborn as sns                        # For dataframes
import warnings

warnings.filterwarnings("ignore")

def load_data(path):
    '''
    Loading the train/test dataset from the directory path
    input : path of train/test dataset
    output : if the document is contains train dataset, function will return both the utterence and relations; if its test dataset, then function will return only the utterence 
    '''
    relations=[]
    df = pd.read_csv(path)
    utterence = df["utterances"]
    if "Core Relations" in df.columns:
        relations = df["Core Relations"]
        return utterence, relations
    else: 
        return utterence

def preprocess_data(X):
    '''
    Function to clean the data before passing into the model
    input : list of utterences
    output : list of preprocessed utterences
    '''
    documents = []
    stemmer = WordNetLemmatizer()
    for sen in range(0, len(X)):
        document = re.sub(r'\W', ' ', str(X[sen]))
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
        document = re.sub(r'\s+', ' ', document, flags=re.I)
        document = re.sub(r'^b\s+', '', document)
        document = document.lower()
        document = document.split()
        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)
        documents.append(document)
    return documents


def feature_extraction(utterences,vectorizer,tfidfconverter):
    '''
    Transforms the textual data into vectorized format.
    input : list of input uttrenences, allready initialized count vectorizer, tfidf converter.
    output : list of vectorized utterences, list of binarized relations. 
    '''
    X_counts = vectorizer.fit_transform(utterences).toarray()
    X_tfidf = tfidfconverter.fit_transform(X_counts).toarray()
    return X_tfidf


def train_val_split(utterences,vectorizer,validation_size):
    '''
    Function to split the training dataset into train and validation datasets.
    input : vectorized utterence, binarized relations
    output : input for training, input for validaton, output for training, out for validation.
    '''
    x_train, x_test, y_train, y_test = train_test_split(utterences, vectorizer, test_size=validation_size, random_state=0,shuffle=True)
    return x_train,x_test,y_train,y_test


def pipline_creation():
    '''
    function to create pipeline
    input : 
    '''
    pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidfconverter', TfidfTransformer()),
    ('classifier', SGDClassifier(random_state=0))
    ])

    parameters = {
        'vectorizer__max_df': (0.5, 0.75, 1.0),
        'vectorizer__max_features': (None, 5000, 10000, 50000),
        'vectorizer__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        'tfidfconverter__use_idf': (True, False),
        'tfidfconverter__norm': ('l1', 'l2')
    }
    return pipeline, parameters

def save_predictions(test_utterences,test_out_predicitons):
    '''
    function to save the test output predictions
    input : test utterances , test predictions from model
    '''
    sub_dt = pd.DataFrame(zip(test_utterences,test_out_predicitons), columns=['Utterences','Predicted Core realtions'])
    sub_dt.to_csv("predicted-test.csv", index=None)
    return 

def main(train_path,test_path,val_size):
    '''
    main function
    '''
    utterences,relations=load_data(train_path) # loading the training data
    preprocessed_utterences = preprocess_data(utterences) #preprocessing the training data
    x_train, x_val, y_train, y_val=train_val_split(preprocessed_utterences,relations,val_size)

    pipeline, parameters = pipline_creation()
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    print(parameters)
    t0 = time.time()
    grid_search.fit(x_train, y_train)
    print("Grid search time : %0.3fs" % (time.time() - t0))
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_val)
    print(accuracy_score(y_val, y_pred))
    print('Training accuarcy score: ' + str(pipeline.score(x_train,y_train)))
    print('Validation accuarcy: ' + str(pipeline.score(x_val,y_val)))
    test_utterences = load_data(test_path) # loading the test dataset
    preprocessed_test_utterences = preprocess_data(test_utterences) #preprocessing the test dataset
    y_test_pred = pipeline.predict(x_val)
    save_predictions(test_utterences,y_test_pred)
    return


main(parameters.train_path,parameters.test_path,parameters.validation_size)