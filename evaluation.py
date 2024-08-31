import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
from helpers import load_data
from pandas import DataFrame, value_counts, unique
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
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

def plot_confusion_matrix(labels, prediction, to_file):
    fig, ax = plt.subplots(figsize=(10,10))
    cm = confusion_matrix(labels, prediction)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    disp.plot(cmap='Blues', values_format='', xticks_rotation=45, ax=ax)
    plt.savefig('results/' + to_file, bbox_inches='tight', pad_inches=0)
    return

def distance_metrics(x, labels, ids):
    n_temp_observations = np.array(np.cumsum(value_counts(ids)[unique(ids)])[:-1])
    labels = np.vsplit(labels.reshape(-1,1), n_temp_observations)
    labels = np.vstack(labels)
    x1 = x.reshape(-1,1)
    silhouette_s = sklearn.metrics.silhouette_score(x1, labels, metric='euclidean')
    calinski_hs = sklearn.metrics.calinski_harabasz_score(x1, labels)
    davies_bs = sklearn.metrics.davies_bouldin_score(x1, labels)
    distance_metric = dict(
        silhouette_score=silhouette_s,
        calinski_harabasz_score=calinski_hs,
        davies_bouldin_score = davies_bs,
    )
    return pd.DataFrame([distance_metric])

def metrics(y_true, y_pred):
        accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
        kappa = sklearn.metrics.cohen_kappa_score(y_true, y_pred)
        f1_micro = sklearn.metrics.f1_score(y_true, y_pred, average="micro")
        f1_macro = sklearn.metrics.f1_score(y_true, y_pred, average="macro")
        f1_weighted = sklearn.metrics.f1_score(y_true, y_pred, average="weighted")
        recall_micro = sklearn.metrics.recall_score(y_true, y_pred, average="micro")
        recall_macro = sklearn.metrics.recall_score(y_true, y_pred, average="macro")
        recall_weighted = sklearn.metrics.recall_score(y_true, y_pred, average="weighted")
        precision_micro = sklearn.metrics.precision_score(y_true, y_pred, average="micro")
        precision_macro = sklearn.metrics.precision_score(y_true, y_pred, average="macro")
        precision_weighted = sklearn.metrics.precision_score(y_true, y_pred, average="weighted")
        metrics_dict = dict(
            accuracy=accuracy,
            kappa=kappa,
            f1_micro=f1_micro,
            f1_macro=f1_macro,
            f1_weighted=f1_weighted,
            recall_micro=recall_micro,
            recall_macro=recall_macro,
            recall_weighted=recall_weighted,
            precision_micro=precision_micro,
            precision_macro=precision_macro,
            precision_weighted=precision_weighted,
        )
        return pd.DataFrame([metrics_dict])

def process(data_path, prediction_filename, output_name):
    test_data_path = data_path + 'test/'
    _, labels, ids = load_data(test_data_path)
    blocks_limits = list(np.cumsum(value_counts(ids)[unique(ids)])[:-1])
    blocks_limits = [0] + blocks_limits
    labels = labels[blocks_limits]
    prediction = np.load(prediction_filename)
    print(prediction.shape)
    print(labels.shape)

    report = classification_report(prediction, labels, output_dict=True)
    df = DataFrame(report).transpose()
    df.to_csv('results/classification_report_' + output_name + '.csv')
    plot_confusion_matrix(labels, prediction, to_file='confusion_matrix_' + output_name + '.pdf')
    metrics_df = metrics(labels, prediction)
    metrics_df.to_csv('results/metrics_' + output_name + '.csv')
    distance_metrics_df = distance_metrics(prediction, labels, ids)
    distance_metrics_df.to_csv('results/distance_metrics_'+ output_name + '.csv')
    print('Evaluation succesful!')
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('prediction_filename')
    parser.add_argument('--output_name', type=str, default=None)
    args = parser.parse_args()

    process(**vars(args))
