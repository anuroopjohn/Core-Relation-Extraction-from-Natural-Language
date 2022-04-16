import matplotlib.pyplot as plt
import numpy as np

def plot_graph(plot_var,train_plot_list,val_plot_list):
    '''
    Function to plot the train and validation accuracy and loss
    args : plotting variable(accuracy/loss), list of training points to plot, list of validation points to plot.
    results : None
    '''
    epochs = len(train_plot_list)
    fig = plt.figure(figsize=(8,6))
    if plot_var=="accuracy": plt.title("Train/Validation Accuracy")
    elif plot_var =="loss" : plt.title("Train/Validation Loss")
    plt.plot(list(np.arange(epochs) + 1) , train_plot_list, label='train')
    plt.plot(list(np.arange(epochs) + 1), val_plot_list, label='validation')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.legend(loc='best')
    if plot_var=="accuracy": plt.savefig('Train_Val_accuracy.png')
    elif plot_var =="loss" : plt.savefig("Train_Val_loss.png")
    return