"""
Model Generation for Cifera2

Roadmap

> gen command?

> ADD MORE DATA
"""

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation, Reshape, Flatten, Conv1D, MaxPooling1D, AveragePooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D, LSTM, GRU, SimpleRNN, Bidirectional, LayerNormalization, Embedding, Normalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping

import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder

import numpy as np

import json
import kagglehub
import pandas as pd
import joblib

import os

def model_formation(X_train, y_onehot, layers, optimizer, loss):
    """
    Creates the Sequential model and adds its layers.

    Parameters:
        X_train (array-like): Training data

    Returns:
        model (tensorflow.keras.models.Sequential): The fully formed model.
        ver (str): Version number of the configuration.
    """
    # Creating the model
    model = Sequential()

    ver = None

    # Layer formation
    """
    # Cifera2/0 - First Iteration (PASS)
    ver = 0
    model.add(Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(y_onehot.shape[1], activation='softmax'))
    """
    """
    # Cifera2/1 - Second Iteration (FAIL)
    ver = 1
    model.add(Dense(1024, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(y_onehot.shape[1], activation='softmax'))
    """
    """
    # Cifera2/2 - Third Iteration (PASS)
    ver = 2
    model.add(Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(y_onehot.shape[1], activation='softmax'))
    """
    """
    # Cifera2/3 - Fourth Iteration (FAIL)
    ver = 3
    model.add(Dense(units=4000, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=2000, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=500, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=250, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(y_onehot.shape[1], activation='softmax'))
    """

    # Create model based on layers from config.json

    for layer in layers:
        model.add(eval(layer, globals(), locals()))

    # Compile model
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model, str(ver)

def model_gen(name, data, exclude_builtin_dataset=False):

    # Extract data from config file
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    config = config["model"]

    if name is None:
        name = config["name"]
    
    optimizer = config["optimizer"]
    loss = config["loss"]
    epochs = int(config["epochs"])
    batch_size = config["batch_size"]
    layers = list(config["layers"])

    # Make required directories if not already present
    os.makedirs(f'./models', exist_ok=True)
    os.makedirs(f'./artifacts', exist_ok=True)

    # Dataset: "Music features" by Insiyah Hajoori and MARSYAS.
    # File: data.csv
    # Contains 1000 audio tracks of 30-second length.
    # Contains 100 tracks of each genre, for a total of ten genres.
    # Genres:
        # "Blues"
        # "Classical"
        # "Country"
        # "Disco"
        # "Hiphop"
        # "Jazz"
        # "Metal"
        # "Pop"
        # "Reggae"
        # "Rock"
    # 22050Hz Mono 16-bit audio, .wav format
    # Features extracted using libROSA library
    if not exclude_builtin_dataset:
        file_path = kagglehub.dataset_download("insiyeah/musicfeatures", path="data.csv")
        data_music_features = pd.read_csv(file_path)
        if data is not None:
            data_music_features = pd.concat([data_music_features, pd.read_csv(data)], ignore_index=True)
    else:
        if data is not None:
            data_music_features = pd.read_csv(data)
        else:
            raise Exception("No data has been provided. Either provide a dataset or use the builtin dataset, or both.")


    # This dataset has 30 columns:
        # filename (str)
        # tempo (float)
        # beats (float)
        # chroma_stft (float)
        # rmse (float)
        # spectral_centroid (float)
        # spectral_bandwidth (float)
        # rolloff (float)
        # zero_crossing_rate (float)
        # mfcc1 (float)
        # ...
        # mfcc20 (float)
        # label (str): Actual label of each track.

    # data_music_features.head()

    # Split dataset into X (features) and y (labels)
    X = data_music_features.drop(['filename', 'label'], axis=1)
    y = data_music_features['label']

    # Apply One-Hot encoding to labels
    # y_onehot = pd.get_dummies(y)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    y_onehot = to_categorical(y_enc)

    # y_onehot.head()

    # Normalize features (I have no idea... thanks ChatGPT)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # The following code is only usable if StratifiedKFold is disabled and train_test_split is used instead.
    # model, ver = model_formation(X_train)

    # Print out a summary of the model
    # model.summary()

    # Train model, with StratifiedKFold
    skf = StratifiedKFold(
        n_splits = 5, # Number of folds to train over
        shuffle = True,
        random_state = 42
    )

    fold = 0

    # Yeah, I have no idea what this does... thanks a lot, ChatGPT
    # for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y_enc), start=1):
    if True:
        print(f'Fold {fold + 1}')

        # Splits training and testing data, different for each fold
        # X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        # y_train, y_test = y_onehot[train_idx], y_onehot[test_idx]

        # Split into training and testing, different for each fold
        # Random state = 42 or None
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_onehot, test_size=0.2, random_state=42, shuffle=True, stratify=y_enc)

        # Configure Early Stopping to prevent overfitting
        early_stop = EarlyStopping(
            monitor = 'val_loss', # Alternatively, use 'val_accuracy'
            patience = 10, # Number of epochs to wait after lack of improvement before terminating
            restore_best_weights = True # Model returns to best weights
        )

        # Form model and its layers
        # To configure layer formation, see the function
        model, ver = model_formation(X_train, y_onehot, layers, optimizer, loss)

        # Train the model
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stop], shuffle=True)

        # Evaluation for each fold
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f"Fold {fold} - Test Accuracy: {accuracy:.2f}")

    # Test accuracy (again)
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Total Test Accuracy: {accuracy:.2f}")

    ver = name

    # Save the fitted scaler
    joblib.dump(scaler, f'artifacts\Cifera2-{str(ver)}-scaler.pkl')

    # Save the label encoder
    joblib.dump(le, f'artifacts\Cifera2-{str(ver)}-label_encoder.pkl')

    model.save(f"models\Cifera2-{str(ver)}.keras")