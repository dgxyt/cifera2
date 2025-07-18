import librosa
import tensorflow as tf

import os
import json
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import joblib

import csv

def plot_value_array(predictions_array, class_labels):
    """
    Plots an inputted prediction array using matplotlib.

    Parameters:
        predictions_array (numpy.ndarray): The confidence array to be graphed.
        class_labels (numpy.ndarray): Human-readable array of labels.
    """
    predictions_array = np.squeeze(predictions_array)
    plt.figure(figsize=(10,4))
    thisplot = plt.bar(class_labels, predictions_array, color="#777777")
    plt.ylabel("Confidence")
    plt.xticks(class_labels, rotation=45)
    plt.title("Prediction confidence")
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    #plt.tight_layout()
    plt.show()

def audio_format(path, genre='unknown'):
    """
    Uses libROSA to format the audio into a readable format for a model to evaluate. Credit: Insayaa on GitHub
    
    Parameters:
        path (str): Path to the (singular) unformatted audio file.
        genre (str): The genre of the file.
    """

    # Header for the CSV file
    header = 'filename tempo beats chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()

    path = path.strip()
    genre = genre.strip()

    # Create a .csv file with the same filename as the inputted audio file.
    songname = '.'.join(list(path.split('\\')[-1].split('.')[:-1]))
    parent_folder = path.split(songname)[0]
    if parent_folder.strip() == '':
        parent_folder = '.'
    if parent_folder[-1] == '\\':
        parent_folder = parent_folder[:-1]
    os.makedirs(f'{parent_folder}\\Cifera2_data', exist_ok=True)
    file = open(f'{parent_folder}\\Cifera2_data\\{songname}.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)

    # Extract song data (Courtesy of Insayaa on GitHub)
    y, sr = librosa.load(path, mono=True, duration=30)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    tempo = tempo[0]
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    to_append = f'{songname} {tempo} {beats.shape[0]} {np.mean(chroma_stft)} {np.mean(rms)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
    for e in mfcc:
        to_append += f' {np.mean(e)}'
    to_append += f' {genre.lower()}'
    file = open(f'{parent_folder}\\Cifera2_data\\{songname}.csv', 'a', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(to_append.split())

def audio_batch_format(path, seperate, genre='unknown'):
    """
    Uses libROSA to format the folder of audio into files of a readable format for a model to evaluate. Credit: Insayaa on GitHub
    
    Parameters:
        path (str): Path to the folder of unformatted audio files.
        genre (str): The genre of the files.
    """
    # Header for the CSV file
    header = 'filename tempo beats chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()

    path = path.strip()
    if path == '':
        path = '.'
    genre = genre.strip()
    if path[-1] == '\\':
        path = path[:-1]

    os.makedirs(f'{path}\\Cifera2_data', exist_ok=True)

    if seperate:
        for filename in os.listdir(path):
            full_path = os.path.join(path, filename)
            if os.path.isfile(full_path):
                y, sr = librosa.load(full_path, mono=True, duration=30)
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
                tempo = tempo[0]
                chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
                rms = librosa.feature.rms(y=y)
                spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
                spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                zcr = librosa.feature.zero_crossing_rate(y)
                mfcc = librosa.feature.mfcc(y=y, sr=sr)
                to_append = f'{filename} {tempo} {beats.shape[0]} {np.mean(chroma_stft)} {np.mean(rms)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
                for e in mfcc:
                    to_append += f' {np.mean(e)}'
                to_append += f' {genre}'

                file = open(f"{path}\\Cifera2_data\\{'.'.join(list(filename.split('.')[:-1]))}.csv", 'w', newline='')
                with file:
                    writer = csv.writer(file)
                    writer.writerow(header)
                    writer.writerow(to_append.split())
    else:
        file = open(f'{path}\\Cifera2_data\\batch_data.csv', 'w', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(header)
        for filename in os.listdir(path):
            full_path = os.path.join(path, filename)
            if os.path.isfile(full_path):
                y, sr = librosa.load(full_path, mono=True, duration=30)
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
                tempo = tempo[0]
                chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
                rms = librosa.feature.rms(y=y)
                spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
                spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                zcr = librosa.feature.zero_crossing_rate(y)
                mfcc = librosa.feature.mfcc(y=y, sr=sr)
                to_append = f'{filename} {tempo} {beats.shape[0]} {np.mean(chroma_stft)} {np.mean(rms)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
                for e in mfcc:
                    to_append += f' {np.mean(e)}'
                to_append += f' {genre}'
                file = open(f'{path}\\Cifera2_data\\batch_data.csv', 'a', newline='')
                with file:
                    writer = csv.writer(file)
                    writer.writerow(to_append.split())


def audio_evaluate(path):
    """
    Uses the model, specified in config.json, to predict the genre of inputted music.
    
    Parameters:
        path (str): Path to the audio in formatted form, as a .csv file.
    
    Returns:
        predictions (numpy.ndarray): Confidence array as outputted by the model.
        labels (numpy.ndarray): Human-readable genre labels.
        predicted_label (str): The final predicted label.
    """

    # Grab requested version from config.json
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    MODEL_VERSION = config['commands']['evaluate']['model_version']
    
    # Make a list of all versions of currently availible models.
    ACTIVE_VERSIONS = list(map(lambda model: model.replace('Cifera2-', '').replace('.keras', ''), os.listdir(".\models")))
    
    # Except if the model version is invalid
    if MODEL_VERSION not in ACTIVE_VERSIONS:
        raise Exception("An invalid version of the model has been entered. Check the models folder for missing models, or change the requested version in config.json.")
    
    # Load model
    model = tf.keras.models.load_model(f'.\models\Cifera2-{str(MODEL_VERSION)}.keras')

    # Read converted file, drop filename and label
    data = pd.read_csv(path).drop(['filename', 'label'], axis=1, errors='ignore')

    try:
        # Load the fitted StandardScaler
        scaler = joblib.load(f'.\\artifacts\\Cifera2-{str(MODEL_VERSION)}-scaler.pkl')
    except FileNotFoundError:
        # If file not found, raise Exception (with additional context)
        raise Exception("The scaler file was not found. Check the artifacts folder for a missing file, or generate a new model.\nName format of scaler file: \"Cifera2-[MODEL_VERSION]-scaler.pkl\"")

    try:
        # Load the label encoder to decode the file
        le = joblib.load(f'.\\artifacts\\Cifera2-{str(MODEL_VERSION)}-label_encoder.pkl')
    except FileNotFoundError:
        # If file not found, raise Exception (with additional context)
        raise Exception("The encoder file was not found. Check the artifacts folder for a missing file, or generate a new model.\nName format of encoder file: \"Cifera2-[MODEL_VERSION]-label_encoder.pkl\"")

    # Normalize
    # Do not use scaler.fit_transform whenever predicting - only when training.
    data_scaled = scaler.transform(data)

    # Make prediction
    predictions = model.predict(data_scaled, verbose=1)
    predicted_class = predictions.argmax(axis=1)
    predicted_label = le.inverse_transform(predicted_class)

    # Fetch label array
    labels = le.classes_

    return predictions, labels, predicted_label
