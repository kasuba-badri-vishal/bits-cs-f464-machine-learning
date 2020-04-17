import pandas as pd
import numpy as np


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

    def initialize_weights(type='normal', size=4):
        if(type == 'normal'):
            return np.random.normal(size=size)
        elif(type == 'uniform'):
            return np.random.uniform(size=size)
        else:
            return np.random.random(size=size)

    def accuracy(y_test, y_pred):
        true_positive = np.sum(np.logical_and(
            (y_pred == y_test), (y_pred == 1)))
        true_negative = np.sum(np.logical_and(
            (y_pred == y_test), (y_pred == 0)))
        false_positive = np.sum(np.logical_and(
            (y_pred != y_test), (y_pred == 1)))
        false_negative = np.sum(np.logical_and(
            (y_pred != y_test), (y_pred == 0)))
        print(true_positive, true_negative, false_positive, false_negative)
        precision_1 = true_positive/(true_positive+false_positive)
        precision_0 = true_negative/(true_negative+false_negative)
        recall_1 = true_positive/(true_positive+false_negative)
        recall_0 = true_negative/(true_negative+false_positive)
        f1_score_1 = 2*(precision_1*recall_1)/(precision_1+recall_1)
        f1_score_0 = 2*(precision_0*recall_0)/(precision_0+recall_0)
        accuracy = (true_positive+true_negative)/y_test.size
        return accuracy, f1_score_1, recall_1, precision_1, recall_0, precision_0, f1_score_0

    def preprocess(dataset):
        min = np.min(dataset)
        print(min)
        max = np.max(dataset)
        print(max)
        dataset = (dataset-min)/(max-min)
        print(dataset)
        return dataset
