import sounddevice as sd
import numpy as np
import torch
import threading
import time
import sys
import queue
import json

from utils.ringbuffer import RingBuffer
from model import VQAutoEncoder
from utils.utils import frames_to_audio
import os

class AudioPipeline:
    def __init__(self,
                 sample_rate=16000, # Common for speech models
                 callback_block_size=1024, # Frames per chunk
                 max_input_seconds=2.0): # Capacity of input buffer (seconds)

        self.sample_rate = sample_rate
        self.callback_block_size = callback_block_size

        self.network_out_queue = queue.Queue(maxsize=50)

        self.input_buffer = RingBuffer(capacity=int(max_input_seconds * self.sample_rate))
        self.playback_buffer = RingBuffer(capacity=int(max_input_seconds * self.sample_rate))

        self.data_ready = threading.Event()

        self.input_stream = sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.callback_block_size,
            callback=self.input_stream_callback
        )
        self.output_stream = sd.OutputStream(
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.callback_block_size,
            callback=self.output_stream_callback
        )

        # Si jamais on veut utiliser le client en tant qu'executable
        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            dirname = sys._MEIPASS
        else:
            dirname = os.path.dirname(__file__)

        model_path = os.path.join(dirname, "model_state/vq_model_weights.pth")

        self.model = VQAutoEncoder()
        self.model.load_state_dict(torch.load(model_path, weights_only=True))

        self.model.eval()

        self.compression_active = True

    def input_stream_callback(self, indata, frames, time, status):
        if status:
            print(status)
        mono = np.mean(indata, axis=1) # convert audio to mono
        self.input_buffer.write(mono)
        self.data_ready.set()

    def output_stream_callback(self, outdata, frames, time, status):
        chunk = self.playback_buffer.peek_read(frames)
        if len(chunk) < frames:
            chunk = np.pad(chunk, (0, frames - len(chunk)))
        self.playback_buffer.consume(frames)
        outdata[:] = chunk.reshape(-1,1)

    def poll_buffer(self):
        while self.input_stream.active:
            # Blocks until the data_ready flag is set
            if self.data_ready.wait(timeout=1):
                self.data_ready.clear()

                # Process the data as long as there is enough in the input buffer
                while len(self.input_buffer) >= self.callback_block_size:
                    data = self.input_buffer.read(self.callback_block_size)
                    self.process_audio(data)

    def process_audio(self, audio):
        if self.compression_active:
            audio_tensor = torch.from_numpy(audio).type(torch.float32)
            with torch.inference_mode():
                quantized_frames = self.model.encode_to_indices(audio_tensor.unsqueeze(0).unsqueeze(0))

            tensor_uint8 = quantized_frames.detach().cpu().to(torch.uint8).contiguous()
            raw_bytes = tensor_uint8.numpy().tobytes()
            header = json.dumps({"shape": tensor_uint8.shape}).encode()

            packet = b'C' + len(header).to_bytes(4, "little") + header  + raw_bytes
        else:
            packet = b'R' + audio.tobytes()

        try:
            self.network_out_queue.put_nowait(packet)
        except queue.Full:
            pass

    def process_incoming_packet(self, packet: bytes):
        if not packet:
            return

        packet_type = packet[0:1]
        payload = packet[1:]

        if packet_type == b'C':
            header_len = int.from_bytes(payload[:4], "little")
            header_b = payload[4:4+header_len]

            header = json.loads(header_b)
            shape = header["shape"]

            body = payload[4+header_len:]

            # Reconstruct the tensor en précisant le type uint8
            buff = np.frombuffer(body, dtype=np.uint8)

            # Passage en torch.long obligatoire pour la couche nn.Embedding du décodeur
            encoded_tensor = torch.from_numpy(buff).reshape(shape).to(torch.long)

            # Run the decoding step ONLY on the receiver
            with torch.inference_mode():
                decoded_audio_frames = self.model.decode_from_indices(encoded_tensor)
                decoded_audio = frames_to_audio(decoded_audio_frames, self.callback_block_size)

            self.playback_buffer.write(decoded_audio)
        else:
            self.write_playback_bytes(payload)

    def toggle_compression(self):
        self.set_compression_active(not self.compression_active)

    def set_compression_active(self, flag: bool):
        self.compression_active = flag

    @property
    def is_listening(self):
        return self.input_stream.active

    def start_listening(self):
        print("Starting Listening...")
        self.input_stream.start()

        polling_thread = threading.Thread(target=self.poll_buffer)
        polling_thread.start()

    def stop_listening(self):
        print("Stopping Listening...")
        self.input_stream.stop()

    def start_playback(self):
        print("Playing back audio...")
        self.output_stream.start()

    def stop_playback(self):
        print("Stopping audio playback...")
        self.output_stream.stop()

    def write_playback_bytes(self, buffer: bytes):
        data = np.frombuffer(buffer, dtype=np.float32)
        self.playback_buffer.write(data)

# ATTENTION : N'exécute pas sans écouteurs
if __name__ == "__main__":
    asc = AudioPipeline(sample_rate=16000)

    asc.start_listening()
    asc.start_playback()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        asc.stop_playback()
        asc.stop_listening()

