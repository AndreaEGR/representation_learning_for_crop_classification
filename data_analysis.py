import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import entropy
from helpers import load_data, add_spectral_indices

# GLOBAL_VARS
BANDS = [
            'B02',
            'B03',
            'B04',
            'B05',
            'B06',
            'B07',
            'B08',
            'B8A',
            'B11',
            'B12',
            ]
ADDITIONAL_FEATURES = [
            'sin',
            'cos',
            'ndvi',
            'ndwi',
            'ndti',
            'ndsvi',
            'evi',
            ]

FEATURES = BANDS + ADDITIONAL_FEATURES
NUM_FEATURES = len(FEATURES)
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

# functions

def boxplot_of_each_class(pixels, labels, output_path):
    classes_idxs = [CLASSES.index(c) for c in CLASSES]
    num_classes = len(CLASSES)
    for idx in classes_idxs:
        fig, ax = plt.subplots(figsize=(10,3))
        ax.boxplot(pixels[(labels == idx)])
        ax.set(
            title=CLASSES[idx],
            xticks = range(1,NUM_FEATURES+1),
            xticklabels=FEATURES,
            )
        plt.savefig(output_path + 'Box_and_whiskers_' + CLASSES[idx] + '.png', bbox_inches = 'tight', pad_inches = 0)
        plt.close()
    return

def multi_temporal_entropy(pixels,output_path):
    num_bins = 2**14
    entropy_each_feature = []
    for band in pixels.T:
        x = band - np.min(band) / (num_bins)
        counts, bins = np.histogram(x, bins=num_bins, range=(0,1))
        bins = bins[:-1] + (bins[1] - bins[0]) / 2
        p = counts / float(counts.sum())
        entropy_each_feature.append(entropy(p))
    y_pos = np.arange(1, len(entropy_each_feature)+1)
    fig, ax = plt.subplots(figsize=(10,3))
    ax.bar(y_pos, entropy_each_feature)
    ax.set(title="Entropy multitemporal",xticks = y_pos,xticklabels= FEATURES,)
    plt.savefig(output_path + 'Entropy_multitemporal.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    return

# main
def process(data_path, output_path, scaler):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    pixels, labels, ids = load_data(data_path)
    pixels = add_spectral_indices(pixels)
    if scaler != None:
        if  scaler == 'minmax':
            scale = MinMaxScaler()
            pixels = scale.fit_transform(pixels)
        else:
            scale = StandardScaler()
            pixels = scale.fit_transform(pixels)
    #boxplot_of_each_class(pixels,labels,output_path)
    multi_temporal_entropy(pixels,output_path)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='Location of dataset')
    parser.add_argument('output_path', type=str)
    parser.add_argument('--scaler', choices=['minmax','standar'], help='No normalization by default')
    args = parser.parse_args()
    process(**vars(args))
