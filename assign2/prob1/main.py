import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from LogisticRegression import Logistic


if __name__ == "__main__":
    dataset = pd.read_csv('data_banknote_authentication.txt', names=[
                          'p1', 'p2', 'p3', 'p4', 't'])
    train, test = train_test_split(dataset, test_size=0.2)
    # w = np.random.normal(size=4)
    w = np.array([3, 4, 3, 4])
    # print(train)
    print(w)
    value = train.drop(columns=['t']).multiply(w).sum(axis=1)
    # print(value)
    value = Logistic.sigmoid_fun(value)
    # print(value)
    error = Logistic.error(value, train['t'])
    print(error)
    Logistic.grad_desc(value, train['t'], w, train.drop(columns=['t']))
