import numpy as np
import pandas as pd


class LogisticRegression:

    def __init__(self):
        pass

    def sigmoid_fun(self, z):
        return (1/(1+np.exp(-z)))

    def error(self, y_train, a):
        return -(y_train.dot(np.log(a))+(1-y_train).dot(np.log(1-a)))

    def regularization(self, w, type, beta):
        if(type == 'wR'):
            return 0
        elif(type == 'L1'):
            return beta*(np.sum(np.abs(w)))
        elif(type == 'L2'):
            return beta*(np.sum(np.square(w)))

    def grad_desc(self, x_train, y_train, w, beta, alpha, loss, type):
        loss1 = 0
        while(loss1 != loss):
            z = (x_train.dot(w))  # z = wT*x
            a = self.sigmoid_fun(z)
            loss1 = loss
            reg_factor = self.regularization(w, type, beta)
            loss = reg_factor + self.error(y_train, a)
            w = w - alpha*((a-y_train).dot(x_train))
        return w

    def predict_data(self, x_test, x_train, w, y_train, type='L1', alpha=1e-4, beta=1e-3):
        z = (x_train.dot(w))  # z = wT*x
        a = self.sigmoid_fun(z)
        reg_factor = self.regularization(w, type, beta)
        loss = reg_factor + self.error(y_train, a)
        w = w - alpha*((a-y_train).dot(x_train))
        w = self.grad_desc(x_train, y_train, w, beta, alpha, loss, type)
        z = (x_test.dot(w))
        a = self.sigmoid_fun(z)
        return np.where(a > 0.5, 1, 0)
