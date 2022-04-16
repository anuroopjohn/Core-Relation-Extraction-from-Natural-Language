import torch
import numpy as np

def train(clf, dataloader_train, optimizer, loss_func , train_len, device):
    '''
    Function to train the model
    args : initialized model, train dataloader, initialized optimizer, initilaized los function, train data lenth, device type(cpu/gpu)
    results : loss of one epoch, accuracy of one epoch
    '''
    epoch_loss,epoch_acc = 0,0
    for idx,(X,y) in enumerate(dataloader_train):
        # the training routine is these 5 steps:
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        # step 2. compute the output
        y_pred = clf(x_in=X.float())
        y_1 = (y_pred).to('cpu').detach().numpy()
        y_1=(np.array(y_1) >= 0)*1
        y_0=y.to('cpu').detach().numpy()
        acc = sum([(y_0[i]==y_1[i]).all()*1 for i in range(len(y_0))])
        epoch_acc+= acc
        # step 3. compute the loss
        loss = loss_func(y_pred, y.squeeze(1).float())
        epoch_loss+= loss.item()
        # step 4. use loss to produce gradients
        loss.backward()
        # step 5. use optimizer to take gradient step
        optimizer.step()
    epoch_loss ,  epoch_acc = round(epoch_loss/(idx+1),3), round(epoch_acc/train_len,3)
    return epoch_loss , epoch_acc

def evaluate(clf, dataloader_test, loss_func, val_len ,device):
    '''
    Function to validate the model
    args : initialized model, validation dataloader, initilaized los function, train data lenth, device type(cpu/gpu)
    results : loss of one epoch, accuracy of one epoch
    '''
    val_epoch_loss,val_epoch_acc  = 0,0
    for idx,(X,y) in enumerate(dataloader_test):
        X = X.to(device)
        y = y.to(device)
        y_pred = clf(x_in=X.float())
        y_1 = (y_pred).to('cpu').detach().numpy()
        y_1=(np.array(y_1) >= 0)*1
        y_0=y.to('cpu').detach().numpy()
        val_acc = sum([(y_0[i]==y_1[i]).all()*1 for i in range(len(y_0))])
        val_epoch_acc+=val_acc
        loss = loss_func(y_pred, y.squeeze(1).float())
        val_epoch_loss+= loss.item()
    val_epoch_acc=round(val_epoch_acc/val_len,3)
    val_epoch_loss = round(val_epoch_loss/(idx+1),3)
    return val_epoch_acc, val_epoch_loss
