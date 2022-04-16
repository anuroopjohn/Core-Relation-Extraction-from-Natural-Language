import torch
import torch.nn as nn


class Classifier(nn.Module):
    """ a multi-layered perceptron based classifier """
    def __init__(self, num_features,out_features):
        """
        Args:
            num_features (int): the size of the input feature vector
        """
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(in_features=num_features, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=out_features)

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
        y_out = self.fc2(y_out)#.squeeze(0)
        return y_out