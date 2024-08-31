import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
from helpers import load_data
from pandas import DataFrame, value_counts, unique
import sklearn.metrics
plt.rcParams.update({'font.size': 14})

CLASSES = [
            'barley',
            'wheat',
            'rapeseed',
            'corn',
            'sunflower',
            'orchards',
            'nuts',
            'permanent meadows',
            'temporary meadows',
            ]

def evaluation_rs(y_pred, y_true, ids, model_id):
    n_temp_observations = np.array(np.cumsum(value_counts(ids)[unique(ids)])[:-1])
    labels = np.vsplit(y_pred.reshape(-1,1), n_temp_observations)
    labels = np.vstack(labels)

    silhouette_s = sklearn.metrics.silhouette_score(y_pred.reshape(-1,1), y_true.reshape(-1,1), metric='euclidean')
    calinski_hs = sklearn.metrics.calinski_harabasz_score(y_pred.reshape(-1,1), y_true.reshape(-1,1))
    davies_bs = sklearn.metrics.davies_bouldin_score(y_pred.reshape(-1,1), y_true.reshape(-1,1))

#    print('distance metrics done!')

    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    kappa = sklearn.metrics.cohen_kappa_score(y_true,y_pred)
    mcc = sklearn.metrics.matthews_corrcoef(y_true,y_pred)

#    print('performance evaluation metrics done!')

    evaluation = dict(
        model_id=model_id,
        accuracy=accuracy,
        kappa=kappa,
        mcc=mcc,
        silhouette_score=silhouette_s,
        calinski_harabasz_score=calinski_hs,
        davies_bouldin_score=davies_bs,
    )
 #   print('computed metrics!')
    return pd.DataFrame([evaluation])

def process(data_path, prediction_filename, output_name, model_id):
    print('process...')
    test_data_path = data_path + 'test/'
    _, labels, ids = load_data(test_data_path)
    blocks_limits = list(np.cumsum(value_counts(ids)[unique(ids)])[:-1])
    blocks_limits = [0] + blocks_limits
    labels = labels[blocks_limits]
    prediction = np.load(prediction_filename)
#    print(labels.shape)
#    print(prediction.shape)

    df = evaluation_rs(prediction, labels, ids, model_id)
    #output_name = 'results/evaluation_random_search.csv'
    if os.path.exists(output_name):
        df.to_csv(output_name, mode='a', index=False, header=False)
    else:
        df.to_csv(output_name, index=False)

    print('Evaluation succesful!')
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    parser.add_argument('prediction_filename', type=str)
    parser.add_argument('output_name', type=str)
    parser.add_argument('--model_id', type=str)
    args = parser.parse_args()

    process(**vars(args))
