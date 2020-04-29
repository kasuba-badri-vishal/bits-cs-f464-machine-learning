import numpy as np
import pandas as pd
import argparse
import time
import matplotlib.pyplot as plt
from DataUtil import DataUtil
from LogisticRegression import LogisticRegression

def start_regression(dist,reg,alpha,scale):
    dataset1 = pd.read_csv('data_banknote_authentication.txt', names=['x1', 'x2', 'x3', 'x4', 't'])
    dataset = DataUtil.preprocess(scale,dataset1.drop(columns=['t']))
    dataset['x0'] = 1
    dataset['t'] = dataset1['t']
#     print(dataset)
    x_train, y_train, x_test, y_test = DataUtil.load_dataset(dataset, split_ratio=0.8)
    
    w = DataUtil.initialize_weights(dist, size=5) # dist = random,normal,uniform,stdnormal
    LR = LogisticRegression()
    y_pred,error,iterations = LR.predict_data(x_test, x_train, w, y_train,reg, alpha=0.0005, beta=1e-2)
    """
    wR - without Regularization
    L1 - L1 Regularization
    L2 - L2 Regularization
    """
    DataUtil.plotData(error,iterations)
    DataUtil.plotData1(y_pred,y_test)
    accuracy,f1_score_1,recall_1,precision_1,recall_0,precision_0,f1_score_0 = DataUtil.accuracy(y_test, y_pred)
    print("Accuracy is : {:.2f}".format(accuracy))
    print("Results   :\tPositive\tNegative")
    print("f1_score  :\t{:.2f}\t\t{:.2f}".format(f1_score_1,f1_score_0))
    print("recall    :\t{:.2f}\t\t{:.2f}".format(recall_1,recall_0))
    print("precision :\t{:.2f}\t\t{:.2f}".format(precision_1,precision_0))
    
    
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Logistic Regression Model for detection of Forged Bank Notes")
    parser.add_argument('dist',default='stdnormal',help='Type of weight Initialization')
    parser.add_argument('reg',default='wR',help='Regularization to be used')
    parser.add_argument('alpha',default=0.0001,help='Value of Alpha')
    parser.add_argument('scale',default='None',help='Type of Preprocessing')
    np.seterr(divide = 'ignore') 
    args = parser.parse_args()
    start_time = time.time()
    start_regression(args.dist,args.reg,args.alpha,args.scale)
    end_time = time.time()
    print("\nTotal Time Taken",(end_time-start_time))
    

    
