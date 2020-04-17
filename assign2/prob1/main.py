import numpy as np
import pandas as pd
from DataUtil import DataUtil
from LogisticRegression import LogisticRegression

if __name__ == "__main__":
    dataset = pd.read_csv('data_banknote_authentication.txt', names=[
                          'x1', 'x2', 'x3', 'x4', 't'])

    # dataset1 = DataUtil.preprocess(dataset.drop(columns=['t']))
    # dataset1['t'] = dataset['t']
    # dataset = dataset1
    x_train, y_train, x_test, y_test = DataUtil.load_dataset(
        dataset, split_ratio=0.8)

    # type = random,normal,uniform
    w = DataUtil.initialize_weights(type='normal', size=4)
    LR = LogisticRegression()
    y_pred = LR.predict_data(x_test, x_train, w, y_train,
                             type='L2', alpha=1e-3, beta=1e-2)
    """
    wR - without Regularization
    L1 - L1 Regularization
    L2 - L2 Regularization
    """

    accuracy, f1_score_1, recall_1, precision_1, recall_0, precision_0, f1_score_0 = DataUtil.accuracy(
        y_test, y_pred)
    print(accuracy)
    print(f1_score_1, f1_score_0)
    print(recall_1, recall_0)
    print(precision_1, precision_0)
