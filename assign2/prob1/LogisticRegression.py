import numpy as np


class Logistic:

    def sigmoid_fun(y):
        return (1/(1+np.exp(-y)))

    def error(y, t):
        return -sum(t.multiply(np.log(y))+(1-t).multiply((1-np.log(y))))

    def grad_desc(y, t, w, x):

        err = Logistic.error(y, t)
        alpha = 0.00000000002
        beta = 10
        # print((t-y))
        # print(x.multiply((t-y), axis=0))
        print(x.multiply((t-y), axis=0).sum(axis=0))
        while(err > beta):
            w = w - alpha*(x.multiply((t-y), axis=0).sum(axis=0))
            value = x.multiply(w).sum(axis=1)
            value = Logistic.sigmoid_fun(value)
            err = Logistic.error(value, t)
            print(err)
