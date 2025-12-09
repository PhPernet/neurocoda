import sounddevice as sd
import numpy as np
import torch
import threading
import time

from utils.ringbuffer import RingBuffer
from codec import Encoder, Decoder
from utils.utils import frames_to_audio

class AudioPipeline:
    def __init__(self,
                 sample_rate=16000, # Common for speech models
                 callback_block_size=1024, # Frames per chunk
                 max_input_seconds=2.0, # Capacity of input buffer (seconds)
                 max_output_seconds=2.0): # Capacity of output buffer (seconds)

        self.sample_rate = sample_rate
        self.callback_block_size = callback_block_size
        self.output_buffer = RingBuffer(capacity=int(max_output_seconds * self.sample_rate))
        self.input_buffer = RingBuffer(capacity=int(max_input_seconds * self.sample_rate))

        self.input_stream = sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.callback_block_size,
            callback=self.input_stream_callback
        )
        self.output_stream = sd.OutputStream(
            channels=1,
            samplerate=self.sample_rate,
            callback=self.output_stream_callback
        )

        self.encoder_model = Encoder()
        self.encoder_model.load_state_dict(torch.load("model_state/encoder_state", weights_only=True))
        self.decoder_model = Decoder()
        self.decoder_model.load_state_dict(torch.load("model_state/decoder_state", weights_only=True))

        self.encoder_model.eval()
        self.decoder_model.eval()

    def input_stream_callback(self, indata, frames, time, status):
        if status:
            print(status)
        mono = np.mean(indata, axis=1) # convert audio to mono
        self.input_buffer.write(mono)

    def output_stream_callback(self, outdata, frames, time, status):
        chunk = self.output_buffer.peek_read(frames)
        if len(chunk) < frames:
            chunk = np.pad(chunk, (0, frames - len(chunk)))
        self.output_buffer.consume(frames)
        outdata[:] = chunk.reshape(-1,1)

    def poll_buffer(self, rate_seconds=0.01):
        while self.input_stream.active:
            if len(self.input_buffer) > self.callback_block_size:
                data = self.input_buffer.read(self.callback_block_size)
                self.process_audio(data)
            time.sleep(rate_seconds)

    def process_audio(self, audio):
        audio_tensor = torch.from_numpy(audio).type(torch.float32)
        # frames = self.audio_to_frames(audio_tensor, audio.shape[0])
        with torch.inference_mode():
            encoded_audio_frames = self.encoder_model(audio_tensor.unsqueeze(0).unsqueeze(0))
            decoded_audio_frames = self.decoder_model(encoded_audio_frames)
            decoded_audio = frames_to_audio(decoded_audio_frames, audio.shape[0])

        self.output_buffer.write(decoded_audio)

    def start_listening(self):
        print("Starting Listening...")
        self.input_stream.start()
        # input()

    def stop_listening(self):
        print("Stopping Listening...")
        self.input_stream.stop()

    def start_playback(self):
        print("Playing back audio...")
        self.output_stream.start()

    def stop_playback(self):
        print("Stooping audio playback...")
        self.output_stream.stop()

if __name__ == "__main__":
    asc = AudioPipeline(sample_rate=16000)
    polling_thread = threading.Thread(target=asc.poll_buffer)

    asc.start_listening()
    asc.start_playback()
    polling_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        asc.stop_playback()
        asc.stop_listening()

