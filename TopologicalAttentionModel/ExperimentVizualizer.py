#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 12:07:40 2023

@author: joseaguilar
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd
import numpy as np

sns.set_theme()


def main_viz():
    data_file = './_experiment_Jan-19-2023_14-45-58_nodes_10.pkl'
    
    with open(data_file, 'rb') as jar:
        data = pickle.load(jar)
    print(data)
    performance = {
        'Performance': data['performance'],
    }
    #sns.set(rc={"figure.figsize":(10, 3)})
    plt.figure()
    #performance_pd = pd.DataFrame(performance)
    sns.lineplot(data=performance).set(\
                                       title=f'Performance Per Epoch on {data["cli"].nodes}-node Graphs', 
                                       xticks=range(len(data['performance'])),
                                       xlabel='Epochs',
                                       ylabel='Cost')
    plt.xticks(rotation=65)
    data_per_epoch = 2560

    plt.figure()
    sns.lineplot(data=performance).set(\
                                       title=f'Performance Per Epoch on {data["cli"].nodes}-node Graphs',
                                       xticks=range(len(data['performance'])),
                                       xticklabels=[i*data_per_epoch for i in range(len(data['performance']))],
                                       xlabel='Training Data',
                                       ylabel='Cost')
    #training_time = data['training_time']
    #import pdb; pdb.set_trace()

if __name__ == '__main__':
    main_viz()