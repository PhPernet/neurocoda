import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU-only to avoid GPU CUDA issues

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def build_encoder(input_shape=(None, 1)):
    return keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(filters=16, kernel_size=4, strides=2, padding='same'),
        layers.ReLU(),
        layers.Conv1D(filters=32, kernel_size=4, strides=2, padding='same'),
        layers.ReLU(),
        layers.Conv1D(filters=64, kernel_size=4, strides=2, padding='same'),
        layers.ReLU(),
    ], name='encoder')


def build_decoder(input_shape=(None, 64)):
    return keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1DTranspose(filters=32, kernel_size=4, strides=2, padding='same'),
        layers.ReLU(),
        layers.Conv1DTranspose(filters=16, kernel_size=4, strides=2, padding='same'),
        layers.ReLU(),
        layers.Conv1DTranspose(filters=1, kernel_size=4, strides=2, padding='same'),
        layers.Activation('tanh'),
    ], name='decoder')


def build_autoencoder(frame_size=1024):
    encoder = build_encoder(input_shape=(frame_size, 1))
    decoder = build_decoder(input_shape=(frame_size // 8, 64))

    inputs = keras.Input(shape=(frame_size, 1))
    latent = encoder(inputs)
    reconstructed = decoder(latent)

    autoencoder = keras.Model(inputs=inputs, outputs=reconstructed, name='autoencoder')
    return autoencoder, encoder, decoder


# Numpy equivalent of the PyTorch audio_to_frames utility
def audio_to_frames(audio, audio_length, frame_size=1024, hop_length=512):
    remainder = (audio_length - frame_size) % hop_length
    padding = (hop_length - remainder) % hop_length
    padded = np.pad(audio, (0, padding))

    num_frames = (len(padded) - frame_size) // hop_length + 1
    frames = np.stack([padded[i * hop_length: i * hop_length + frame_size]
                       for i in range(num_frames)])
    return frames


def train_model(audio, frame_size=1024, hop_length=512):
    autoencoder, encoder, decoder = build_autoencoder(frame_size)

    # Prepare frames: shape (num_frames, frame_size) -> (num_frames, frame_size, 1)
    frames = audio_to_frames(audio, len(audio), frame_size, hop_length)
    frames = frames[..., np.newaxis].astype(np.float32)

    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss='mae',  # L1 loss equivalent
    )

    autoencoder.summary()

    print("Beginning Training...")
    autoencoder.fit(
        frames, frames,  # input = target (autoencoder)
        batch_size=32,
        epochs=25,
        shuffle=True,
    )

    encoder.save_weights("model_state/encoder_state.weights.h5")
    decoder.save_weights("model_state/decoder_state.weights.h5")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog='code',
        description='Trains Encoder/Decoder model on a sample of audio')

    parser.add_argument('filename')  # positional argument
    parser.add_argument('-t', '--train',
                        action='store_true')  # on/off flag

    args = parser.parse_args()
    print(args.filename, args.train)

    if args.train and args.filename:
        from librosa import load
        audio, _ = load(args.filename, sr=None)
        audio = audio.astype(np.float32)
        train_model(audio)
