import argparse
import os
import numpy as np
from breizhcrops import BreizhCrops
from tqdm import tqdm
from pandas import to_datetime

BANDS = [
        'B2',
        'B3',
        'B4',
        'B5',
        'B6',
        'B7',
        'B8',
        'B8A',
        'B11',
        'B12',
        ]

CLASSES_OF_INTEREST = [
                'Clear',
                'Water',
                'Shadow',
                'Cirrus',
                'Cloud',
                'Snow',
            ]

N_CLASSES = len(CLASSES_OF_INTEREST)
N_FEATURES = len(BANDS) + 1

def raw_transform(input_timeseries):
    return input_timeseries

def read_data(regions, n_samples, **kwargs):
    pixels = []
    labels = []
    ids = []

    for region in regions:
        data_L2A = BreizhCrops(region=region, level='L2A', transform=raw_transform)
#        selected_band_idxs = [data_L2A.bands.index(b) for b in BANDS]
        selected_band_idxs = list(range(1,11))
        if n_samples == None:
            n_samples_tmp = len(data_L2A)
        else:
            n_samples_tmp = n_samples
        for idx in tqdm(range(n_samples_tmp)):
            x, y, r_id = data_L2A[idx]
    #        if x.shape[0] == 101:
            x_bands = x[:, selected_band_idxs].astype('int16')
            if 'add_doy' in kwargs:
                doy = to_datetime(x[:,0])
                doy = doy.day_of_year
                sin = (((np.sin(doy*2*np.pi/365)+1)/2)*1e4).astype('int16')
                cos = (((np.cos(doy*2*np.pi/365)+1)/2)*1e4).astype('int16')
                x_bands = np.c_[x_bands, sin, cos]
            if 'add_ndvi' in kwargs:
                bands_ndvi = [data_L2A.bands.index(b) for b in ['B4','B8']]
                bands_ndvi = x[:, bands_ndvi]
                ndvi = (((x[:,1]-x[:,0]) / (x[:,1]+x[:,0]))*1e4).astype('int16')
                x_bands = np.c_[x_bands, ndvi]
            # pytorch tensor to numpy array
            label = y*np.ones_like(x_bands[:,0])
            p_id = r_id*np.ones_like(x_bands[:,0])
            pixels.append(x_bands)
            labels.append(label)
            ids.append(p_id)
    n_pixels = len(pixels)
    pixels = np.concatenate(pixels, axis=0)
    labels = np.concatenate(labels, axis=0)
    p_ids = np.concatenate(ids, axis=0)
    no_zero_idx = ~(np.sum(pixels[:, :10] != 0, axis=1) == 0)
    pixels = pixels[no_zero_idx].astype('int16')
    labels = labels[no_zero_idx].astype('int16')
    p_ids = p_ids[no_zero_idx].astype('int32')
    pixels = pixels+1
#    pixels = np.array(pixels).astype('int16')
#    labels = np.array(labels).astype('int16')
    return pixels, labels, p_ids

def create_model(n_features):
    from tensorflow.keras.layers import Input, Dense, Subtract, concatenate
    from tensorflow.keras.models import Model

    input_data = Input(shape=n_features)
    encoders = [Dense(3, activation='relu')(input_data) for _ in range(N_CLASSES)]
    decoders = [Dense(n_features, activation='sigmoid')(encoder) for encoder in encoders]
    differences = [Subtract()([input_data, decoder]) for d, decoder in enumerate(decoders)]

    errors = concatenate(differences)
    fcn = Dense(64, activation='relu')(errors)
    fcn = Dense(64, activation='relu')(fcn)
    output = Dense(N_CLASSES, activation='softmax')(fcn)

    return Model(inputs=input_data, outputs=output)

def load_weights(path):
    from tensorflow.keras.models import load_model

    autoencoders = [load_model(path + 'trained_AE_' + class_n) for class_n in CLASSES_OF_INTEREST]
    fcn = load_model(path + 'fc_trained_2_layers')
    encoder_weights = []
    decoder_weights = []
    for autoencoder in autoencoders:
        autoencoder_weights = autoencoder.get_weights()
        encoder_weights = encoder_weights + autoencoder_weights[:2]
        decoder_weights = decoder_weights + autoencoder_weights[2:]
    weights = encoder_weights + decoder_weights + fcn.get_weights()

    return weights

def load_trained_model():
    path = 'trained_models/'
    model = create_model(n_features=N_FEATURES)
    weights = load_weights(path)
    model.set_weights(weights)

    return model

def get_cloud_mask(data):
    if len(data.shape) != 2:
        data = data.reshape(-1, N_FEATURES)
    model = load_trained_model()
    mask = np.argmax(model.predict(data), axis=1)
    mask = np.isin(mask, [3,4,5]) * 1
    return mask

def process(regions, n_samples, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
#    data_L1C = BreizhCrops(region=region, level='L1C', transform=raw_transform)
#    data_L1C = read_data(data_L1C, n_samples, add_ndvi=True)
#    mask = get_cloud_mask(data_L1C[0])
#    del data_L1C
    #    print(data_L2A.shape, labels.shape, mask.shape)
    pixels, labels, p_ids = read_data(regions, n_samples, add_doy=True, with_labels=True)
    np.save(output_path + 'pixels.npy', pixels)
    np.save(output_path + 'labels.npy', labels)
    np.save(output_path + 'ids.npy', p_ids)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--regions', action='store',
                        type=str,
                        nargs='+',
                        default=['frh01'],
                        help='Select the region/s of interest to preprocess (frh01, frh02, frh03, frh04). Benchmark is frh01 and frh02 for training, frh03 validation and frh04 testing.')
#    parser.add_argument('--region', default='belle-ile', choices=['frh01', 'frh02', 'frh03', 'frh04', 'belle-ile'])
    parser.add_argument('--n_samples', type=int, help='Number of spatial samples to preprocess. By default all available in region')
#    parser.add_argument('--scaled', type=bool, default=True, help='whether scaling or not')
#    parser.add_argument('level', help='processing level', choices=['L1C', 'L2A'])
    parser.add_argument('output_path', type=str)
    args = parser.parse_args()

    process(**vars(args))

