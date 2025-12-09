import torch
import numpy as np
from scipy.io.wavfile import write

# Reconstruct audio given a tensor of frames
def frames_to_audio(audio_frames, original_length, frame_size=1024, hop_length=512):
    num_frames = audio_frames.shape[0]
    total_length = hop_length * (num_frames - 1) + frame_size
    reconstructed = torch.zeros(total_length)
    weight = torch.zeros(total_length)

    for i, frame in enumerate(audio_frames.squeeze(1)):
        start = i * hop_length
        end = start + frame_size
        reconstructed[start:end] += frame
        weight[start:end] += 1

    reconstructed = reconstructed / torch.clamp(weight, min=1)

    return reconstructed[:original_length]

# Create batches of frames of a given length n (frame_size) every k (hop_length) samples in the audio
def audio_to_frames(audio_tensor: torch.Tensor, audio_length: int, frame_size:int=1024, hop_length:int=512):
    # If, for iteration i, i * hop_length + frame_size > audio_length, torch.unfold will drop any data at the end of the 1D tensor
    # So here we calculate the number of 0s we need to pad to the end of the 1D tensor
    remainder = (audio_length - frame_size) % hop_length
    padding = (hop_length - remainder) % hop_length
    padded_audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding))

    frames = padded_audio_tensor.unfold(0, frame_size, hop_length)
    return frames

def audio_to_file(file_name, audio, sample_rate):
    wav_data = np.clip(audio, -1.0, 1.0)
    wav_data = (wav_data * 32767).astype(np.int16)
    write(filename=file_name, rate=sample_rate, data=wav_data)