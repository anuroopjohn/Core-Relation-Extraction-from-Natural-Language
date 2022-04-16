import torch
import torch.optim as optim
import math
import numpy as np
import pandas as pd
import torch.nn as nn
import sys

from read_split_data import read_data, transform_data, split_data
from dataloader import data_loader
from model import Classifier
from train import train, evaluate
from plot import plot_graph
from prediction import prediction, single_predict

def main(train_path, test_path, batch_size, val_size, num_epochs,learning_rate):
    '''
    Main function
    args : train , test dataset path, batch size for training, validation split size, number epochs for taining, learning rate,
    results : None
    '''
    utterences, multi_labels  = read_data(train_path) # reading the training data from disk
    x_tfidf, y_binarized, in_dimension, out_dimension, tfidfvectorizer, multilabel_binarizer = transform_data(utterences, multi_labels)
    train_x,val_x,train_y,val_y = split_data(x_tfidf,y_binarized,val_size) # splittng the data into train and validation data
    dataloader_train, dataloader_test, train_len, val_len = data_loader(train_x,val_x,train_y,val_y,batch_size)
    
    cuda = torch.cuda.is_available()
    print("Using CUDA: {}".format(cuda))
    device = torch.device("cuda" if cuda else "cpu") #selecting the device type(cpu/gpu)
    clf = Classifier(num_features=in_dimension,out_features=out_dimension) #initializing the model
    clf.to(device)
    loss_func = nn.CrossEntropyLoss()#initializing the loss function
    optimizer = optim.Adam(clf.parameters(), lr=learning_rate) #initializing the loss function
    n_iter=math.ceil(train_len/batch_size)
    losses = []
    epoch_loss_list, epoch_acc_list ,val_epoch_acc_list,val_epoch_loss_list = [],[],[],[]

    for epoch in range(num_epochs):
        #training the model
        epoch_loss , epoch_acc = train(clf, dataloader_train, optimizer, loss_func , train_len, device)
        epoch_loss_list.append(epoch_loss)
        epoch_acc_list.append(epoch_acc)
        #validating the model
        val_epoch_acc, val_epoch_loss = evaluate(clf, dataloader_test, loss_func, val_len, device)
        val_epoch_acc_list.append(val_epoch_acc)
        val_epoch_loss_list.append(val_epoch_loss)
        print('epoch : ' + str(epoch+1)+'/'+str(num_epochs))
        print("-"*40)
        print('loss : ' + str(epoch_loss)+ ' \t val loss : '+ str(val_epoch_loss)+ '\nacc :' + str(epoch_acc)+ ' \t val acc :' + str(val_epoch_acc))
        print("+"*40)  # -----------------------------------------
    #saving the trained model in disk
    torch.save(clf, 'slp-tfidf-multilabel-model')
    #ploting the train and validation accuracy and loss
    plot_graph("accuracy",epoch_acc_list, val_epoch_acc_list)
    plot_graph("loss",epoch_loss_list, val_epoch_loss_list)
    #prediction from test data
    predcted_df = prediction(test_path,tfidfvectorizer,multilabel_binarizer,device)
    return 
    
    

batch_size, val_size, num_epochs,learning_rate = 4, 0.2, 25, 0.001
main("../data/train_data.csv","../data/test_data.csv",batch_size, val_size, num_epochs,learning_rate)
