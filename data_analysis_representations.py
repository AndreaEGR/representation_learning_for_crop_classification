import numpy as np
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from pandas import value_counts, unique
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

def boxplot_of_each_class(pixels, labels):
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
        plt.savefig('results/box_and_whiskers_' + CLASSES[idx] + '.png', bbox_inches = 'tight', pad_inches = 0)
        plt.close()
    return

def entropy_representations(representations, option):
    num_bins = 2**12
    r = representations.reshape(-1, len(CLASSES), NUM_FEATURES)
    entropy_each_feature = []
    entropy_dict = {}
    for f in range(NUM_FEATURES):
        x = (r[:,:,f] - np.min(r[:,:,f])) / (num_bins)
        counts, bins = np.histogram(x, bins=num_bins)
        bins = bins[:-1] + (bins[1] - bins[0]) / 2
        p = counts / float(counts.sum())
        entropy_score = entropy(p)
        entropy_each_feature.append(entropy_score)
        entropy_dict[FEATURES[f]] = entropy_score
    y_pos = np.arange(1,len(FEATURES)+1)
    fig, ax = plt.subplots(figsize = (8,4))
    ax.bar(y_pos, entropy_each_feature, align='center', alpha=0.5)
    ax.set(xticks=y_pos, xticklabels=FEATURES)
    ax.set_xticklabels(FEATURES, rotation=45)
    plt.savefig('results/entropy_' + option + '_representations.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()
    return entropy_dict

def plot_representations(representations, ids, labels, option):
    aes_names = ['ae_' + c for c in CLASSES]
    n_temp_observations = np.array(np.cumsum(value_counts(ids)[unique(ids)])[:-1])
    labels = np.vsplit(labels.reshape(-1,1), n_temp_observations)
    labels = [int(label[0]) for label in labels]
    representations = [representations[labels == l] for l in np.unique(labels)]
    for i, c in enumerate(CLASSES):
        fig, ax = plt.subplots(figsize=(10,5))
        scaler = StandardScaler()
        im = ax.imshow(scaler.fit_transform(np.float32(np.mean(np.abs(representations[i]), axis=0).reshape(len(CLASSES), NUM_FEATURES))),
                                                        cmap='gray',
                                                        vmin = -1, vmax=1)
        ax.set(title=c, yticks=range(len(CLASSES)), yticklabels=aes_names, xticks=range(NUM_FEATURES), xticklabels=FEATURES)
        ax.set_xticklabels(FEATURES, rotation=45)
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        plt.savefig('results/representations_' + option  + '_class_' + c +'.pdf', bbox_inches='tight',pad_inches=0)
        plt.close()
    return

# main
def process(data_path, options, scaler):
#    if not os.path.exists(output_path):
#        os.makedirs(output_path)
    entropies = []
    for option in options:
        representations_path = data_path + 'representations/' + option + '_representations.npy'
        representations = np.load(representations_path)
        option_data_path = data_path + option + '/'
        _, labels, ids = load_data(option_data_path)
        if  scaler == 'minmax':
            scale = MinMaxScaler()
            representations = scale.fit_transform(representations)
        elif scaler == 'standard':
            scale = StandardScaler()
            representations = scale.fit_transform(representations)

        entropy_score = entropy_representations(representations, option)
        entropies.append(entropy_score)
        plot_representations(representations, ids, labels, option)
    entropies = pd.DataFrame(entropies)
    entropies.to_csv(data_path + 'entropies.csv')
    print('Done!')
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
#    parser.add_argument('representations_path', type=str)
    parser.add_argument('data_path', type=str, help='Path to ids and labels')
#    parser.add_argument('output_path', type=str)
    parser.add_argument('--options', action='store', type=str, nargs='+', default=['train','validation','test'])
    parser.add_argument('--scaler', choices=['minmax','standard'], help='No normalization by default')
    args = parser.parse_args()
    process(**vars(args))
