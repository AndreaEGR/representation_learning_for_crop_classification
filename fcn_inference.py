import tensorflow as tf
import tensorflow_addons as tfa
import os
import numpy as np
import argparse

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


def predict(data_dir, model, features, option='train'):
    n_features = len(features)
    representations = np.load(data_dir)
    total_samples= representations.shape[0]
    representations = representations.reshape(total_samples * NUM_CLASSES, n_features + 2)
    features_ids = [FEATURES.index(i) for i in features]
    representations = representations[:,features_ids].reshape(total_samples, NUM_CLASSES*n_features)
    prediction = model(representations)
    prediction = np.argmax(prediction, axis=1)
    return prediction

def process(data_path, model_path, options, no_ind):
    if no_ind == False:
        features = BANDS + ['ndvi', 'ndwi', 'ndti', 'ndsvi', 'evi']
    else:
        features = BANDS
    model = tf.keras.models.load_model(model_path + 'trained_fcn.keras')
    predictions_dir = data_path + 'predictions/'
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)
    for option in options:
        data_dir = data_path + 'representations/' + option + '_representations.npy'
        prediction = predict(data_dir, model, features, option)
        np.save(predictions_dir + option + '_prediction.npy', prediction)
    print('Done!')
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('model_path')
    parser.add_argument('--options', action='store', type=str, nargs='+', default=['train','validation','test'])
    parser.add_argument('--no_ind', action="store_true", help="if given, spectral indices are not included")
    args = parser.parse_args()

    process(**vars(args))
