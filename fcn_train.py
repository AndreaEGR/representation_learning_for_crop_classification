import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from helpers import load_data, plot_metric
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

def fcn(n_classes, n_features, neurons_per_layer=[128,64,32]):
    input_data = tf.keras.layers.Input(shape=n_features)
    fc = tf.keras.layers.Dense(neurons_per_layer[0], activation='relu')(input_data)
    for n in neurons_per_layer[1:]:
        fc = tf.keras.layers.Dense(n, activation='relu')(fc)
    fc = tf.keras.layers.Dense(n_classes, activation='softmax')(fc)

    return tf.keras.models.Model(inputs=input_data, outputs=fc)

def train_model(train_representations, train_labels, validation_representations, validation_labels, neurons, lr, epochs, batch_size):
    results_dir = 'results/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    n_features = train_representations.shape[1]
    model = fcn(NUM_CLASSES, n_features, neurons)
    early_stop = tf.keras.callbacks.EarlyStopping(
                                                monitor='val_loss',
                                                patience=50,
                                                min_delta=1e-6,
                                                mode='min',
                                                restore_best_weights=True)

    mcc = tfa.metrics.MatthewsCorrelationCoefficient(num_classes=NUM_CLASSES)
    model.compile(loss='categorical_crossentropy',
                                metrics=['accuracy', mcc],
                                optimizer=tf.keras.optimizers.Adam(learning_rate=lr))

    history = model.fit(
                    x=train_representations,
                    y=train_labels,
                    # validation_split=0.1,
                    validation_data=[validation_representations, validation_labels],
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stop],
                    ).history
    plot_metric(history, keys=['loss', 'val_loss'],
                to_file='results/fcn_loss.pdf', xlabel='epochs', ylabel='loss')
    plot_metric(history, keys=['accuracy', 'val_accuracy'],
                to_file='results/fcn_accuracy.pdf', xlabel='epochs', ylabel='OA')
    plot_metric(history, keys=['MatthewsCorrelationCoefficient', 'val_MatthewsCorrelationCoefficient'],
                to_file='results/fcn_mcc.pdf', xlabel='epochs', ylabel='MCC')

    return model

def load_representations_labels(rep_path, labels_path, features, option='train', scaler='standard', max_samples=None):
    n_features = len(features)
    representations = np.load(rep_path + 'representations/' + option + '_representations.npy')
    total_samples = representations.shape[0]
    representations = representations.reshape(total_samples * NUM_CLASSES, n_features + 2)
    features_ids = [FEATURES.index(i) for i in features]
    representations = representations[:,features_ids].reshape(total_samples, NUM_CLASSES*n_features)
    if scaler == 'standard':
        scale = StandardScaler()
        representations = scale.fit_transform(representations)

    _, labels, ids = load_data(labels_path + option + '/')

    blocks_limits = list(np.cumsum(value_counts(ids)[unique(ids)])[:-1])
    blocks_limits = [0] + blocks_limits
    labels = labels[blocks_limits]

    if max_samples != None:
        balance = value_counts(labels)
        print(balance)
        balance[balance > max_samples] = max_samples
        print(balance)
        random_samples = []
        for c in unique(labels):
            samples_c = list(np.where(labels == c)[0])
            if len(samples_c) > max_samples:
                samples_c = list(np.random.choice(samples_c, max_samples, replace=False))
            random_samples = random_samples + samples_c
        random_samples = sorted(random_samples)
        total_samples = len(representations)
        representations = representations[random_samples]
        print(f"Total samples {total_samples} limited to {len(representations)} for dataset balance with max samples {max_samples}")
        labels = labels[random_samples]
    labels = tf.keras.utils.to_categorical(labels)
    return representations, labels

def process(rep_path, labels_path, output_path, neurons, lr, epochs, batch_size, max_samples, no_ind):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if no_ind == False:
        features = BANDS + ['ndvi', 'ndwi', 'ndti', 'ndsvi', 'evi']
    else:
        features = BANDS
    train_representations, train_labels = load_representations_labels(
                                                                    rep_path,
                                                                    labels_path,
                                                                    features=features,
                                                                    scaler=None,
                                                                    max_samples=max_samples,
                                                                    )
    validation_representations, validation_labels = load_representations_labels(
                                                                    rep_path,
                                                                    labels_path,
                                                                    option='validation',
                                                                    features=features,
                                                                    scaler=None,
                                                                    )
#    print(validation_labels)
    model = train_model(train_representations, train_labels,
                        validation_representations, validation_labels,
                        neurons, lr, epochs, batch_size)
    model.save(output_path + 'trained_fcn.keras')
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('rep_path')
    parser.add_argument('labels_path')
    parser.add_argument('output_path')
    parser.add_argument('--neurons', '-n', action='store', type=str, nargs='+', default=[128,64,32,16], help='List with the number of neurons in each hidden layer')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning_rate')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--max_samples', type=int, default=None, help='Max number of samples per class to balance the data')
    parser.add_argument('--no_ind', action="store_true", help="If given, spectral indices are not included")
    args = parser.parse_args()

    process(**vars(args))
