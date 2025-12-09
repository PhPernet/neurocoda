import numpy as np
import threading

class RingBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = np.zeros(self.capacity, dtype=np.float32)

        self.size = 0

        self.write_pos = 0
        self.read_pos = 0

        self.lock = threading.Lock()

    def __len__(self):
        return self.size

    def write(self, data):
        with self.lock:
            n = len(data)
            if n >= self.capacity:
                self.buffer[:] = data[-self.capacity:]
                self.write_pos = 0
                self.read_pos = 0
                self.size = self.capacity
                return

            free_space = self.capacity - self.write_pos
            if n <= free_space:
                self.buffer[self.write_pos:self.write_pos+n] = data
                self.write_pos = (self.write_pos + n) % self.capacity
            else:
                self.buffer[self.write_pos:] = data[:free_space]
                self.buffer[:n-free_space] = data[free_space:]
                self.write_pos = n-free_space

            overflow = (self.size + n) - self.capacity
            if overflow > 0:
                self.read_pos = (self.read_pos + overflow) % self.capacity # read_pos stays at 0 until we've filled the buffer and start overlapping
                self.size = self.capacity
            else:
                self.size += n

    def peek_read(self, num_bytes):
        with self.lock:
            bytes_available = min(num_bytes, self.size)
            if bytes_available == 0:
                return np.zeros(num_bytes, dtype=np.float32)
            # If the peek doesn't require wrapping around to the beginning of array, return n bytes starting from read_pos
            if self.read_pos + bytes_available <= self.capacity:
                return self.buffer[self.read_pos : self.read_pos + bytes_available].copy()
            # Otherwise read n bytes starting from read_pos up to end of array
            part1 = self.buffer[self.read_pos:].copy()
            # Read the remaining bytes starting from the beginning of array
            part2 = self.buffer[:bytes_available - len(part1)].copy()
            return np.concatenate([part1, part2])

    def consume(self, num_bytes):
        with self.lock:
            n_consume = min(num_bytes, self.size)
            self.read_pos = (self.read_pos + n_consume) % self.capacity
            self.size -= n_consume

    def read(self, num_bytes):
        data = self.peek_read(num_bytes)
        self.consume(num_bytes)
        return data