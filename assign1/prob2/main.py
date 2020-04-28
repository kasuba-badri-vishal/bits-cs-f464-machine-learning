import pandas as pd
import numpy as np
import re

if __name__ == "__main__":
    dataset = pd.io.parsers.read_csv('a1_d3.txt', sep='\t',names=['data','value'])
    dataset['data'] = dataset['data'].str.lower()
    print(dataset.head)