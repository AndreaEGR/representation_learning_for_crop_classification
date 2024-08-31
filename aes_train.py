import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import multiprocessing
from math import ceil
from sklearn.preprocessing import StandardScaler
from helpers import load_data, add_spectral_indices, preprocess
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from concurrent.futures import ThreadPoolExecutor
plt.switch_backend('agg')

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

def simple_autoencoder(n_features, code_dim):
    input_layer = tf.keras.layers.Input(shape=n_features)
    encoder = tf.keras.layers.Dense(code_dim, activation='relu')(input_layer)
    decoder = tf.keras.layers.Dense(n_features, activation='sigmoid')(encoder)
    model = tf.keras.Model(input_layer, decoder)

    return model

#def create_model(n_features, units):
#    input_layers = [tf.keras.layers.Input(shape=n_features)
#                    for _ in range(NUM_CLASSES)]
#    encoders = [tf.keras.layers.Dense(units, activation='relu')(input_layer)
#                for input_layer in input_layers]
#    decoders = [tf.keras.layers.Dense(n_features, activation='sigmoid')(encoder)
#                for encoder in encoders]
#    model = tf.keras.Model(input_layers, decoders)
#
#    return model

def plot_metric(history, to_file, xlabel, ylabel, title=''):
    fig, ax = plt.subplots()
    ax.plot(history['loss'], linewidth=2, label = 'Training')
    ax.plot(history['val_loss'], '--', linewidth= 2, label = 'Validation')
    ax.set(
        title=title,
        ylabel=ylabel,
        xlabel=xlabel,
        )
    plt.legend()
    plt.savefig(to_file, bbox_inches='tight', pad_inches=0)
    plt.close()
    return

def return_optimizer(opt, learning_rate):
    if opt == 'Adam':
        return Adam(learning_rate=learning_rate)
    elif opt == 'SGD':
        return SDG(learning_rate=learning_rate)
    elif opt == 'RMSprop':
        return RMSprop(learning_rate=learning_rate)


def train_ae(train_data, valid_data, class_name, n_features, epochs, units, batch_size_rate, learning_rate, loss, output_path):
    print(f"Training AE {class_name}")
    code_dim = units
    ae = simple_autoencoder(n_features, code_dim)

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                            min_delta=1e-5,
                                            patience=10,
                                            verbose=1,
                                            mode='min',
                                            restore_best_weights=True,
                                            )
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    ae.compile(
            optimizer=optimizer,
            loss=loss,
            )
    batch_size = ceil(batch_size_rate*train_data.shape[0])
    history = ae.fit(
        train_data, train_data,
        epochs=epochs,
        batch_size=batch_size,
        validation_data = [valid_data, valid_data],
        verbose=0,
        callbacks=[early_stop]).history
    mae = history['loss'][-1]
    plot_metric(history, to_file='results/loss_' + class_name + '.pdf', xlabel='epochs', ylabel= loss, title=f'{class_name}')
#    plot_metric(history['accuracy'], to_file=output_path + 'acc_' + class_name + '.pdf', xlabel='epochs', ylabel='acc', title=f'{class_name}')
#    plot_metric(history['MatthewsCorrelationCoefficient'], to_file=output_path + 'mcc_' + class_name + '.pdf', xlabel='epochs', ylabel='mcc', title=f'{class_name}')
    ae.save(output_path + class_name + '.keras')
    print(f'Saved AE {class_name}')
    return

def train_aes(train_pixels, validation_pixels, n_features, epochs, units, batch_size_rate, learning_rate, loss, output_path, workers):
#    from concurrent.futures import ProcessPoolExecutor
#    print(len(CLASSES), len(train_pixels), len(validation_pixels))
#    optimizer=return_optimizer(opt=str(optimizer_rs), learning_rate=learning_rate)
    args = ((
            x_train,
            x_valid,
             class_name,
             n_features,
             epochs,
             units,
             batch_size_rate,
             learning_rate,
             loss,
             output_path) for x_train, x_valid, class_name in zip(train_pixels, validation_pixels, CLASSES))
#    print(len(list(args)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = executor.map(train_ae, *zip(*args))
    results = list(results)
#    results = []
#    for arg in args:
#        results.append(train_ae(*arg))

    return results

def process(data_path, output_path, epochs, units, batch_size_rate, learning_rate, loss, no_ind, workers, test):
    total_cores = multiprocessing.cpu_count()
    tf.config.threading.set_intra_op_parallelism_threads(total_cores // workers)
    tf.config.threading.set_inter_op_parallelism_threads(total_cores // workers)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if no_ind:
        print("No indices included")
    if test:
        return

    train_pixels, _ = preprocess(data_path + 'train/', no_ind=no_ind)
    validation_pixels, _ = preprocess(data_path + 'validation/', no_ind=no_ind)
#    print(len(validation_pixels))
    n_features = train_pixels[0].shape[1]
    print(f"Starting aes training with {n_features} input/output features")
    train_aes(train_pixels, validation_pixels, n_features, epochs, units, batch_size_rate, learning_rate, loss, output_path, workers)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='Location of the training dataset')
    parser.add_argument('output_path', type=str)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--units', type=int, default=5)
    parser.add_argument('--batch_size_rate', type=float, default=0.05)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
#    parser.add_argument('--optimizer_rs', type=str, default='Adam')
    parser.add_argument('--loss', type=str, default='mae')
    parser.add_argument('--no_ind', action="store_true", help="If given, spectral indices are not included")
    parser.add_argument('--workers', type=int, default=4, help='num of parallel workers')
    parser.add_argument('--test', action="store_true")

    args = parser.parse_args()

    process(**vars(args))
