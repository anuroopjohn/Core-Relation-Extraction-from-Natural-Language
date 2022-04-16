from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,MultiLabelBinarizer

def read_data(path):
    '''
    Function to read the data from the given path
    args : path of the data
    results : user utterence and the realtions from train data or just user utterance if the file is test.
    '''
    data = pd.read_csv(path)
    utterences= data['utterances']
    if "Core Relations" in data.columns:
        multi_labels = [i.split() for i in data['Core Relations']]
        return utterences, multi_labels
    else:
        utterences= data['utterances']
        return utterences


def transform_data(utterences, labels):
    '''
    function to vectorize both the user utterance and realtions.
    args: utterences and labels in text format.
    results : vectorized utterances and realtions, length of utterances nad realtions, initialized tfidf vectorizer and multilabel binarizer.
    '''
    tfidfvectorizer = TfidfVectorizer(max_features=1500)
    x_tfidf = tfidfvectorizer.fit_transform(utterences).toarray()
    le = LabelEncoder()
    multilabel_binarizer = MultiLabelBinarizer()
    multilabel_binarizer.fit(labels)
    y_binarized = multilabel_binarizer.transform(labels)
    ip_dimension, out_dimension = len(x_tfidf[0]),len(y_binarized[0])
    return x_tfidf, y_binarized , ip_dimension, out_dimension , tfidfvectorizer, multilabel_binarizer

def split_data(x_tfidf,y_binarized,test_size=0.2):
    '''
    Function to split the data into train and validtion
    args : vectorized inputs and outputs
    results : train input, train output, validation input, validation output
    '''
    train_x,val_x,train_y,val_y = train_test_split(x_tfidf,y_binarized,test_size=0.2)
    return train_x,val_x,train_y,val_y