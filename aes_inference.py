import argparse
import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
from pandas import value_counts, unique
from helpers import load_data, add_spectral_indices, preprocess

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
COLORS = [
            'b',
            'orange',
            'g',
            'r',
            'violet',
            'brown',
            'pink',
            'magenta',
            'yellow',
            ]

def load_trained_aes(models_path):
    autoencoders = [tf.keras.models.load_model(models_path + class_name + '.keras') for class_name in CLASSES]

    return autoencoders

def abs_error(y_true, y_pred):
    return np.abs(y_true - y_pred)

def create_model(n_features, code_dim):
    from tensorflow.keras.layers import Input, Dense, Subtract, concatenate
    from tensorflow.keras.models import Model

    input_data = Input(shape=n_features)
    encoders = [Dense(code_dim, activation='relu')(input_data) for _ in range(NUM_CLASSES)]
    decoders = [Dense(n_features, activation='sigmoid')(encoder) for encoder in encoders]
    differences = [Subtract()([input_data, decoder]) for d, decoder in enumerate(decoders)]

    errors = concatenate(differences)

    return Model(inputs=input_data, outputs=errors)

def load_weights(models_path):
    from tensorflow.keras.models import load_model

    autoencoders = [load_model(models_path + class_name + '.keras') for class_name in CLASSES]

    encoder_weights = []
    decoder_weights = []
    for autoencoder in autoencoders:
        autoencoder_weights = autoencoder.get_weights()
        encoder_weights = encoder_weights + autoencoder_weights[:2]
        decoder_weights = decoder_weights + autoencoder_weights[2:]
    weights = encoder_weights + decoder_weights

    return weights

def load_trained_model(models_path, n_features, units):
    model = create_model(n_features=n_features, code_dim=units)
    weights = load_weights(models_path)
    model.set_weights(weights)

    return model

def generate_error_vectors(pixels, ids, autoencoders, option, output_path, hard_mode):
    output_file = output_path + option + '_representations.npy'
    n_temp_observations = np.array(np.cumsum(value_counts(ids)[unique(ids)])[:-1])
    if hard_mode == False:
        print("Running easy mode to avoid memory overload.")
        pixels = np.vsplit(pixels, n_temp_observations)
        errors = []
        for pixel in tqdm(pixels):
            errors.append(np.median(autoencoders(pixel), axis=0))
    else:
        print("Running hard mode.")
        errors = np.array(autoencoders(pixels))
        errors = np.vsplit(errors, n_temp_observations)
        errors = [np.median(err, axis=0) for err in errors]
    errors = np.array(errors)
    np.save(output_file, errors)
    print(f'Saved representations with dtype {errors.dtype}')
    return

def process(data_path, output_path, models_path, units, options, hard_mode, no_ind):
    options = [option for option in options.split(",")]
    print(options)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # Get n_features from training file
    tmp = np.load(data_path + options[0] + '/pixels.npy')
    n_features = tmp.shape[1] if no_ind else tmp.shape[1] + 5
    print(f"Input features {n_features}")
    del tmp

    autoencoders = load_trained_model(models_path, n_features, units)
    for option in options:
        pixels, ids = preprocess(data_path + option + '/', no_ind, split=False)
        generate_error_vectors(pixels, ids, autoencoders, option, output_path, hard_mode)
        print(f'{option} representations saved!')
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('output_path')
    parser.add_argument('models_path')
    parser.add_argument('--options', default='train,validation,test')
    parser.add_argument('--units', type=int)
    parser.add_argument('--hard_mode', action='store_true', help='Provide in case memory resources are sufficient (> 8GB)')
    parser.add_argument('--no_ind', action='store_true', help='If given, spectral indices are not included. Use same selection chosen for aes_train.py.')
    args = parser.parse_args()

    process(**vars(args))

