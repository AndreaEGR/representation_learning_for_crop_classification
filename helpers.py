import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from pandas import value_counts, unique

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

N_FEATURES = len(FEATURES)

CLASSES = [
                    'barley',
                    'wheat',
                    'rapeseed',
                    'corn',
                    'sunflower',
                    'orchards',
                    'nuts',
                    'permanent_meadows',
                    'temporary_meadows',
                    ]

CLASSES_IDXS = [CLASSES.index(c) for c in CLASSES]
NUM_CLASSES = len(CLASSES)

def load_data(data_path):
    pixels = np.load(data_path + 'pixels.npy')
    labels = np.load(data_path + 'labels.npy')
    ids = np.load(data_path + 'ids.npy')

    return pixels, labels, ids

def add_spectral_indices(data):
    ndvi = (data[:,6] - data[:,2]) / (data[:,6] + data[:,2])
    ndwi = (data[:,1] - data[:,6]) / (data[:,1] + data[:,6])
    ndti = (data[:,8] - data[:,9]) / (data[:,8] + data[:,9])
    ndsvi = (data[:,8] - data[:,2]) / (data[:,8] + data[:,2])
    evi = 2.5 * ((data[:,6] - data[:,2]) / (data[:,0] + 6 * data[:,2] + 7.5 + data[:,0] + 1))
    data = np.c_[
                data,
                ndvi.reshape(-1,1),
                ndwi.reshape(-1, 1),
                ndti.reshape(-1, 1),
                ndsvi.reshape(-1, 1),
                evi.reshape(-1, 1),
                ]

    return data.astype('float32')

def preprocess(data_dir, no_ind, split=True):
    pixels, labels, ids = load_data(data_dir)
    # scaling to reflectance values [0,1] dividing by 10000
    pixels = (pixels/10000).astype('float32')
    # compute ndvi, ndwi, ndti, ndsvi and evi and add as
    # colums/features
    if no_ind == False:
        pixels = add_spectral_indices(pixels)
    # data standardization
    scaler = StandardScaler()
    pixels = scaler.fit_transform(pixels)
    if split:
        pixels = [pixels[labels == l] for l in np.unique(labels)]

    return pixels, ids

def plot_metric(history, keys, to_file, xlabel, ylabel, title=''):
    fig, ax = plt.subplots()
    ax.plot(history[keys[0]], linewidth=2, label = 'Training')
    ax.plot(history[keys[1]], '--', linewidth= 2, label = 'Validation')
    ax.set(
    title=title,
    ylabel=ylabel,
    xlabel=xlabel,
    )
    plt.legend()
    plt.savefig(to_file, bbox_inches='tight', pad_inches=0)
    plt.close()
    return

def load_representations_labels(data_path, option='train', features=FEATURES, scaler='standard', max_samples=None):
    n_features = len(features)
    representations = np.load(data_path + 'representations/' + option + '_representations.npy')
    representations = representations.reshape(-1, N_FEATURES)
    features_ids = [FEATURES.index(i) for i in features]
    representations = representations[:,features_ids].reshape(-1, NUM_CLASSES*n_features)
    if scaler == 'standard':
        scale = StandardScaler()
        representations = scale.fit_transform(representations)

    _, labels, ids = load_data(data_path + option + '/')

    blocks_limits = list(np.cumsum(value_counts(ids)[unique(ids)])[:-1])
    blocks_limits = [0] + blocks_limits
    labels = labels[blocks_limits]

    if max_samples != None:
        balance = value_counts(labels)
        print(balance)
        balance[balance > max_samples] = max_samples
        print(balance)
        rus = RandomUnderSampler(random_state=42)
        print(representations.shape)
        representations, labels = rus.fit_resample(representations, labels)
        print(representations.shape)
    labels = to_categorical(labels)

    return representations, labels
