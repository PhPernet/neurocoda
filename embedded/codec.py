import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU-only to avoid GPU CUDA issues

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def mel_spectrogram(
    waveforms: tf.Tensor,
    *,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 80,
    power: float = 1.0,
) -> tf.Tensor:
    """Compute a mel-spectrogram for 1D waveforms (batch or single).

    This mirrors the PyTorch notebook setup:
    - STFT magnitude (power=1.0)
    - mel filterbank projection
    """
    waveforms = tf.convert_to_tensor(waveforms, dtype=tf.float32)
    if waveforms.shape.rank == 3 and waveforms.shape[-1] == 1:
        waveforms = tf.squeeze(waveforms, axis=-1)

    stft = tf.signal.stft(
        waveforms,
        frame_length=n_fft,
        frame_step=hop_length,
        fft_length=n_fft,
        window_fn=tf.signal.hann_window,
        pad_end=False,
    )
    magnitude = tf.abs(stft)
    if power != 1.0:
        magnitude = tf.pow(magnitude, power)

    num_spectrogram_bins = n_fft // 2 + 1
    mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=sample_rate,
        lower_edge_hertz=0.0,
        upper_edge_hertz=float(sample_rate) / 2.0,
    )

    mel = tf.tensordot(magnitude, mel_weight_matrix, axes=1)
    mel.set_shape(magnitude.shape[:-1].concatenate([n_mels]))
    # Output shape: [B, time, n_mels] (or [time, n_mels] for single waveform)
    return mel


def make_mel_loss(
    *,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 80,
    power: float = 1.0,
):
    """Return a Keras-compatible loss: L1(log1p(MEL(pred)), log1p(MEL(true)))."""

    @tf.function
    def mel_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        mel_true = mel_spectrogram(
            y_true,
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=power,
        )
        mel_pred = mel_spectrogram(
            y_pred,
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=power,
        )
        mel_true = tf.math.log1p(mel_true)
        mel_pred = tf.math.log1p(mel_pred)
        return tf.reduce_mean(tf.abs(mel_pred - mel_true))

    return mel_loss


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


def train_model(
    audio,
    frame_size: int = 1024,
    hop_length: int = 512,
    *,
    sample_rate: int = 44100,
    n_mels: int = 80,
):
    autoencoder, encoder, decoder = build_autoencoder(frame_size)

    # Prepare frames: shape (num_frames, frame_size) -> (num_frames, frame_size, 1)
    frames = audio_to_frames(audio, len(audio), frame_size, hop_length)
    frames = frames[..., np.newaxis].astype(np.float32)

    mel_loss = make_mel_loss(
        sample_rate=sample_rate,
        n_fft=frame_size,
        hop_length=hop_length,
        n_mels=n_mels,
        power=1.0,
    )

    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss=mel_loss,
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
        audio, sr = load(args.filename, sr=None)
        audio = audio.astype(np.float32)
        train_model(audio, sample_rate=int(sr))
