import parameters
from sklearn.feature_extraction.text import CountVectorizer , TfidfTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import pandas as pd
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import re
import nltk
nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from skmultilearn.problem_transform import ClassifierChain
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
        for relation in df["Core Relations"]:relations.append(relation.split())
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


def feature_extraction(utterences,relations,vectorizer,tfidfconverter,multilabel_binarizer):
    '''
    Transforms the textual data into vectorized format.
    input : list of input uttrenences, list of relations, allready initialized count vectorizer, tfidf converter and multilabel_binarizer
    output : list of vectorized utterences, list of binarized relations. 
    '''
    X_counts = vectorizer.fit_transform(utterences).toarray()
    X_tfidf = tfidfconverter.fit_transform(X_counts).toarray()
    multilabel_binarizer.fit(relations)
    Y = multilabel_binarizer.transform(relations)
    return X_tfidf,Y


def train_val_split(X_tfidf,Y,validation_size):
    '''
    Function to split the training dataset into train and validation datasets.
    input : vectorized utterence, binarized relations
    output : input for training, input for validaton, output for training, out for validation.
    '''
    x_train_tfidf, x_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(X_tfidf, Y, test_size=validation_size, random_state=0,shuffle=True)
    return x_train_tfidf,x_test_tfidf,y_train_tfidf,y_test_tfidf

def OneVsRest_model(x_train_tfidf,y_train_tfidf):
    '''
    function to invoke OneVsRest model
    input : vectorized training utterence, binalized training relations
    output: trained OneVsRest model with SGD classifier
    '''
    classifier=SGDClassifier(random_state=0)
    clf = OneVsRestClassifier(classifier)
    clf.fit(x_train_tfidf, y_train_tfidf)
    return clf

def ClassifierChain_model(x_train_tfidf,y_train_tfidf):
    '''
    function to invoke ClassifierChain model
    input : vectorized training utterence, binalized training relations
    output: trained ClassifierChain model with SGD classifier
    '''
    clf = ClassifierChain(
    classifier = SGDClassifier(random_state=0),
    require_dense = [False, True])
    clf.fit(x_train_tfidf, y_train_tfidf)
    return clf


def prediction(x,clf):
    '''
    function to predict the output relations using the trained model
    input : vectorized testing utterence
    output: binarized relations predicted by the trained model
    '''
    y_pred = clf.predict(x)
    return y_pred

def save_predictions(test_utterences,test_out_predicitons):
    '''
    function to format and save the test output predictions
    '''
    formated_predictions = []
    for i in test_out_predicitons:
        if len(i)>1: formated_predictions.append((" ").join(sorted(i)))
        else: formated_predictions.append(i[0])
    sub_dt = pd.DataFrame(zip(test_utterences,formated_predictions), columns=['Utterences','Predicted Core realtions'])
    sub_dt.to_csv('predicted-test.csv', index=None)
    return formated_predictions

def main(train_path,test_path,val_size,classifier):
    '''
    main function
    '''
    vectorizer = CountVectorizer(ngram_range=(1,1),analyzer='word') # initilaizting the count vectorizer
    tfidfconverter = TfidfTransformer(norm='l2')  # initilaizting the tfidf transformer
    multilabel_binarizer = MultiLabelBinarizer() # initilaizting the multilabel binarizer

    utterences,relations=load_data(train_path) # loading the training data
    preprocessed_utterences = preprocess_data(utterences) #preprocessing the training data
    X_tfidf,Y = feature_extraction(preprocessed_utterences,relations,vectorizer,tfidfconverter,multilabel_binarizer) #vectorizing the input and output
    x_train_tfidf, x_val_tfidf, y_train_tfidf, y_val_tfidf=train_val_split(X_tfidf,Y,val_size) #spliting the training data into train and test dataset
    if classifier == "OneVsRest":clf=OneVsRest_model(x_train_tfidf,y_train_tfidf) # training the onevsrest model using the input and output data
    elif classifier == "ClassifierChain": clf=ClassifierChain_model(x_train_tfidf,y_train_tfidf) # # training the classifier chain model using the input and output data
    y_pred=prediction(x_val_tfidf,clf) # predicting the output of validation data using the trained model
    val_classification_report = classification_report(y_val_tfidf,y_pred)
    validation_accuracy = accuracy_score(y_val_tfidf, y_pred) #comparing the predicted validation outputs with the actual validation output to determine the model's accuracy
    print("Accuracy of Multi-class Multi-label model : ",validation_accuracy)
    
    test_utterences = load_data(test_path) # loading the test dataset
    preprocessed_test_utterences = preprocess_data(test_utterences) #preprocessing the test dataset
    test_utterence_count_vec = vectorizer.transform(preprocessed_test_utterences).toarray() #vectorizing the inputs using count vectorizer
    test_utterence_tfidf = tfidfconverter.transform(test_utterence_count_vec).toarray() #vectorizing the inputs using tfidf transformer.
    y_test_pred = prediction(test_utterence_tfidf,clf) # predicting the outputs of the test dataset.
    y_test_pred_transform=multilabel_binarizer.inverse_transform(y_test_pred) #converting the binarized outputs to core realtions
    y_test_pred_formated = save_predictions(test_utterences,y_test_pred_transform) # format and sabe the test predictions
    return y_test_pred_formated, validation_accuracy, val_classification_report
    

predictions,val_acc,val_classification_report = main(parameters.train_path,parameters.test_path,parameters.validation_size,parameters.classifier)