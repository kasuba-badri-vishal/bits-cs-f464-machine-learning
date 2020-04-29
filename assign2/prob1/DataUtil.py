import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class DataUtil:

    def load_dataset(dataset, split_ratio=0.8):
        dataset = dataset.sample(frac=1)
        size = int(split_ratio*len(dataset))
        train_set = dataset[:size]
        test_set = dataset[size:]
        x_train = train_set.drop(columns=['t'])
        y_train = train_set['t']
        x_test = test_set.drop(columns=['t'])
        y_test = test_set['t']
        return x_train, y_train, x_test, y_test

    def initialize_weights(dist,size):
        if(dist == 'normal'):
            return np.random.normal(size=size)
        elif(dist == 'uniform'):
            return np.random.uniform(size=size)
        elif(dist == 'stdnormal'):
            return np.random.standard_normal(size=size)
        elif(dist == 'beta'):
            return np.random.beta(size=size)
        else:
            return np.random.random(size=size)

    def accuracy(y_test, y_pred):
        true_positive = np.sum(np.logical_and((y_pred == y_test), (y_pred == 1)))
        true_negative = np.sum(np.logical_and((y_pred == y_test), (y_pred == 0)))
        false_positive = np.sum(np.logical_and((y_pred != y_test), (y_pred == 1)))
        false_negative = np.sum(np.logical_and((y_pred != y_test), (y_pred == 0)))
        print("True Positive :\t",true_positive,"\nTrue Negative :\t", true_negative)
        print("False Positive :\t",false_positive,"\nFalse Negative :\t",false_negative)
        precision_1 = true_positive/(true_positive+false_positive)
        precision_0 = true_negative/(true_negative+false_negative)
        recall_1 = true_positive/(true_positive+false_negative)
        recall_0 = true_negative/(true_negative+false_positive)
        f1_score_1 = 2*(precision_1*recall_1)/(precision_1+recall_1)
        f1_score_0 = 2*(precision_0*recall_0)/(precision_0+recall_0)
        accuracy = (true_positive+true_negative)/y_test.size
        return accuracy, f1_score_1, recall_1, precision_1, recall_0, precision_0, f1_score_0

    def preprocess(scale,dataset):
        if(scale=='min-max'):
            min = np.min(dataset)
            max = np.max(dataset)
            dataset = (dataset-min)/(max-min)
            
        elif scale=='std':
            mean = np.mean(dataset)
            std = np.std(dataset)
            dataset = (dataset-mean)/std
        elif scale=='mean-norm':
            mean = np.mean(dataset)
            min = np.min(dataset)
            max = np.max(dataset)
            dataset = (dataset-mean)/(max-min)
        elif scale=='None':
            pass
        return dataset

    def plotData(error,iterations):
        plt.plot(iterations, error, label = "line 1")
        plt.xlabel('Iterations') 
        plt.ylabel('Error') 
        plt.title('Plot of Error of training set vs no.of Iterations') 
        plt.legend() 
        plt.show() 
        
    def plotData1(y_pred,y_test):
        plt.xlabel('prob') 
        plt.ylabel('Error')
        ax=plt.subplot(111)
        ax.set_xlim(1, 200)
#         dim=np.arange(1,200,1);
        ax.plot(y_pred,'ro',color='y',linewidth=2.0,alpha=0.6, label="Graph2")
        ax.plot(y_test,'ro',color='b',linewidth=2.0, label="Graph1")
#         plt.xticks(dim)
        plt.grid()   
        plt.show()    
        plt.close()