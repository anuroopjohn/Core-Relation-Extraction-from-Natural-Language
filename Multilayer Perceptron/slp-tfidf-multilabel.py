from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader
import torch
from sklearn.preprocessing import LabelEncoder,MultiLabelBinarizer
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np


data = pd.read_csv('train_data.csv')
test_data=pd.read_csv('test_data.csv')

multi_labels = [i.split() for i in data['Core Relations']]
single_labels = [i for i in data['Core Relations']]
single_labels_1 = [[i] for i in data['Core Relations']]
utterences= data['utterances']


learning_rate=0.001
num_epochs=50



tfidfvectorizer = TfidfVectorizer(max_features=1500)
x_tfidf = tfidfvectorizer.fit_transform(utterences).toarray()
#le = LabelEncoder()
#Y = le.fit_transform(single_labels)
#Y=[[i] for i in Y]
le = LabelEncoder()
Y_single = le.fit_transform(single_labels)

multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(multi_labels)
Y = multilabel_binarizer.transform(multi_labels)

n_op_features = len(Y[0])
train_x,test_x,train_y,test_y = train_test_split(x_tfidf,Y,test_size=0.2)
n_ip_features = len(train_x[0])



class CRDataset(Dataset):
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


# In[8]:


batch_size = 4

cr_dataset_train = CRDataset(X=train_x, y=train_y)
cr_dataset_test = CRDataset(X=test_x, y=test_y)
# define a Dataloader
dataloader_train = DataLoader(dataset=cr_dataset_train,batch_size=batch_size, shuffle=True)
dataloader_test = DataLoader(dataset=cr_dataset_test, batch_size=batch_size, shuffle=True)


ip,op=iter(dataloader_train).next()


# In[11]:


class Classifier(nn.Module):
    """ a multi-layered perceptron based classifier """
    def __init__(self, num_features,out_features):
        """
        Args:
            num_features (int): the size of the input feature vector
        """
        super(Classifier, self).__init__()
        # self.emb = nn.Embedding(num_embeddings=len(token2id),
        #                         embedding_dim=num_features)
        #self.rnn = nn.RNN(num_features,hidden_size=32)
        self.fc1 = nn.Linear(in_features=num_features, 
                             out_features=64)
       # self.fc2 = nn.Linear(in_features=64, 
        #                     out_features=32)
       # self.fc3 = nn.Linear(in_features=32, 
         #                    out_features=16)
        self.fc2 = nn.Linear(in_features=64,
                             out_features=out_features)

    def forward(self, x_in, apply_softmax=False):
        """The forward pass of the classifier
        
        Args:
            x_in (torch.Tensor): an input data tensor. 
                x_in.shape should be (batch, num_features)
            apply_softmax (bool): a flag for the sigmoid activation
                should be false if used with the Cross Entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch,)
        """
        y_out = torch.relu(self.fc1(x_in))
#         y_out = torch.relu(self.fc2(y_out))
#         y_out = torch.relu(self.fc3(y_out))
        y_out = self.fc2(y_out)#.squeeze(0)
        return y_out





cuda = torch.cuda.is_available()
print("Using CUDA: {}".format(cuda))

device = torch.device("cuda" if cuda else "cpu")


# In[13]:



epoch_loss_list=[]
epoch_acc_list=[]
val_epoch_acc_list=[]
val_epoch_loss_list=[]

clf = Classifier(num_features=n_ip_features,out_features=n_op_features)
clf.to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(clf.parameters(), lr=learning_rate)

n_iter=math.ceil(len(cr_dataset_train)/batch_size)
print(n_iter)

losses = []
from sklearn.metrics import accuracy_score
for epoch in range(num_epochs):
    epoch_loss = 0
    epoch_acc=0
    val_epoch_loss=0
    val_epoch_acc=0
    for k,(X,y) in enumerate(dataloader_train):
        # the training routine is these 5 steps:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        # step 2. compute the output
        y_pred = clf(x_in=X.float())
        #print(y_pred)
        #print(y_pred)
        #y_1 = torch.argmax(y_pred,dim=1).to('cpu')
        #y_1 = torch.flatten(y_pred)
        y_1 = (y_pred).to('cpu').detach().numpy()
        y_1=(np.array(y_1) >= 0)*1
        #print(y_1)
        
        #y_1 = np.where(y_1>0.5,1,0)
        
        #y_0=torch.flatten(y).to('cpu')
        #print(y_1.cpu().detach().numpy())
        #print(y_1)
        y_0=y.to('cpu').detach().numpy()
        #print(y_0)
        acc = sum([(y_0[i]==y_1[i]).all()*1 for i in range(len(y_0))])
        #print(acc)
        #acc= accuracy_score(y_1,y_0)
        #print(acc)
        epoch_acc+= acc
        # step 3. compute the loss
        loss = loss_func(y_pred, y.squeeze(1).float())
        epoch_loss+= loss.item()
        #print(acc)
        # step 4. use loss to produce gradients
        loss.backward()
        # step 5. use optimizer to take gradient step
        optimizer.step()
        #break
    #break
    epoch_loss = round(epoch_loss/(k+1),3)
    epoch_loss_list.append(epoch_loss)
    epoch_acc = round(epoch_acc/len(cr_dataset_train),3)
    epoch_acc_list.append(epoch_acc)
    
    for k,(X,y) in enumerate(dataloader_test):
        X = X.to(device)
        y = y.to(device)
        y_pred = clf(x_in=X.float())
        y_1 = (y_pred).to('cpu').detach().numpy()
        y_1=(np.array(y_1) >= 0)*1
        #y_pred = clf(x_in=X.float())
        #y_1 = torch.argmax(y_pred,dim=1).to('cpu')
        #y_0=torch.flatten(y).to('cpu')
        #val_epoch_acc+=accuracy_score(y_1,y_0)
        y_0=y.to('cpu').detach().numpy()
        val_acc = sum([(y_0[i]==y_1[i]).all()*1 for i in range(len(y_0))])
        val_epoch_acc+=val_acc
        #print(acc)
        loss = loss_func(y_pred, y.squeeze(1).float())
        #print(loss.item())
        val_epoch_loss+= loss.item()
    val_epoch_acc=round(val_epoch_acc/len(cr_dataset_test),3)
    val_epoch_acc_list.append(val_epoch_acc)
    val_epoch_loss = round(val_epoch_loss/(k+1),3)
    val_epoch_loss_list.append(val_epoch_loss)
    print('epoch : ' + str(epoch+1)+'/'+str(num_epochs))
    print("-"*40)
    print('loss : ' + str(epoch_loss)+ ' \t val loss : '+ str(val_epoch_loss)+ '\nacc :' + str(epoch_acc)+ ' \t val acc :' + str(val_epoch_acc))
    print("+"*40)  # -----------------------------------------
    losses.append(epoch_loss)


# In[14]:


torch.save(clf, 'slp-tfidf-multilabel-model')



model = torch.load('slp-tfidf-multilabel-model')
model1 = torch.load('slp-tfidf-singlelabel-model')

def multilabel_predict(x):
    #print(x)
    x = tfidfvectorizer.transform([x]).toarray()
    x = torch.tensor(x, dtype=torch.float64).cuda()
    pred = model(x_in=x.float())
    y_1 = (pred).to('cpu').detach().numpy()
    y_1=(np.array(y_1) >= 0.9)*1
    #print(y_1)
    y_1 = multilabel_binarizer.inverse_transform(y_1)
    #print(y_1[0])
    return y_1[0]

def singlelabel_predict(x):
    #print(x)
    x = tfidfvectorizer.transform([x]).toarray()
    x = torch.tensor(x, dtype=torch.float64).cuda()
    pred = model1(x_in=x.float())
    y_1 = torch.argmax(pred,dim=1).to('cpu')
    y_1 = le.inverse_transform(y_1)
    return y_1[0]
test_data = pd.read_csv('test_data.csv')
test_utterences= test_data['utterances']
y_test_pred_li=[]
j=0
for utterence in test_utterences:
    test_pred=multilabel_predict(utterence)
    if len(test_pred)>0:
        if len(test_pred)>1 and 'none' in test_pred:
            test_pred=list(test_pred)
            test_pred.remove('none')
        y_test_pred_li.append(('_').join(sorted(test_pred)))
    else:
        test_pred = singlelabel_predict(utterence)
        y_test_pred_li.append(test_pred)
        j+=1


# In[32]:


y_test_pred_li


# In[33]:


len(y_test_pred_li)


# In[34]:


id_list = list(range(len(y_test_pred_li)))
sub_dt = pd.DataFrame(zip(id_list,y_test_pred_li), columns=['Id','Predicted'])
sub_dt.to_csv('submission-ajohnabr12.csv', index=None)

