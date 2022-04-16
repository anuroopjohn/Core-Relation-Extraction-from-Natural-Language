import torch
import pandas as pd
from read_split_data import read_data
import numpy as np

def single_predict(model,x,tfidfvectorizer,multilabel_binarizer,device):
    '''
    function for model prediction for a a single input.
    args : trained model, vectorized utterance, loaded tfidf vectorizer and multilabel binarizer, device type(cpu/gpu)
    results : relation of a single test utterance
    '''
    x = tfidfvectorizer.transform([x]).toarray()
    x = torch.tensor(x, dtype=torch.float64).to(device)
    pred = model(x_in=x.float())
    y_1 = (pred).to('cpu').detach().numpy()
    y_1 = (np.array(y_1) >= 0.9)*1
    y_1 = multilabel_binarizer.inverse_transform(y_1)
    return y_1[0]

def prediction(test_path,tfidfvectorizer,multilabel_binarizer,device):
    '''
    function for model prediction for a entire input.
    args : test dataset path, loaded tfidf vectorizer and multilabel binarizer, device type(cpu/gpu)
    results : relation of a entire test utterance
    '''
    model = torch.load('slp-tfidf-multilabel-model')
    test_utterences = read_data(test_path)
    y_test_pred_li = []
    for utterence in test_utterences:
        test_pred = single_predict(model,utterence,tfidfvectorizer,multilabel_binarizer,device)
        if len(test_pred)>1 and 'none' in test_pred:
            test_pred=list(test_pred)
            test_pred.remove('none')
        y_test_pred_li.append(('_').join(sorted(test_pred)))
    id_list = list(range(len(y_test_pred_li)))
    prediction_df = pd.DataFrame(zip(id_list,y_test_pred_li), columns=['Id','Predicted'])
    prediction_df.to_csv('predicted.csv', index=None)
    return prediction_df


