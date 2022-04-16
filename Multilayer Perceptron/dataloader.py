import torch
from torch.utils.data import Dataset,DataLoader


class CRDataset(Dataset):
    '''class for dataset creation'''
    def __init__(self, X, y):
    # Convert arrays to torch tensors
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    #Must have
    def __len__(self):
        return len(self.y)

    #Must have
    def __getitem__(self,index):
        return self.X[index], self.y[index]



def data_loader(train_x,test_x,train_y,test_y,batch_size):
    '''
    Function create the train and validation dataloader
    args : train input, train output, validation input, validation output, batch size
    results : train dataloader, test dataloader, train data length, validation data length
    '''
    cr_dataset_train = CRDataset(X=train_x, y=train_y)
    cr_dataset_val = CRDataset(X=test_x, y=test_y)
    train_len, val_len = len(cr_dataset_train), len(cr_dataset_val)
    # define a Dataloader
    dataloader_train = DataLoader(dataset=cr_dataset_train,batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset=cr_dataset_val, batch_size=batch_size, shuffle=True)
    return dataloader_train, dataloader_val , train_len, val_len
