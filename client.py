import time
from websockets.sync.client import connect, ClientConnection
import json
import socket
from audio_pipeline import AudioPipeline
import random
import threading
import queue
import inspect

random.seed(time.time())
stop_event = threading.Event()

DISCOVERY_PORT = 50000
SERVER_PORT = 8080

def debug_print(*args):
    # Get the previous frame in the stack (where debug_print was called)
    frame = inspect.currentframe().f_back
    line_number = frame.f_lineno
    filename = frame.f_code.co_filename

    print(f"[{filename}:{line_number}]", *args)

def discover_server(timeout=2):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.settimeout(timeout)

    s.sendto(b"DISCOVER_SERVER", ("255.255.255.255", DISCOVERY_PORT))
    data, addr = s.recvfrom(1024)

    return addr[0]  # server IP


class Client:
    asc = AudioPipeline(sample_rate=16000)

    def __init__(self):
        self.tcp_websocket: ClientConnection = None

        self.udp_socket: socket.socket = socket.socket(socket.AF_INET,  # Internet
                               socket.SOCK_DGRAM)  # UDP
        self.udp_socket.settimeout(1.0)

        port = random.randint(8000, 9000)
        client_socket_address = ("0.0.0.0", port)
        self.udp_socket.bind(client_socket_address)

        print(f"Client udp socket bound to {client_socket_address}")

    def leave_current_room(self):
        try:
            if self.tcp_websocket:
                data = {"action": "leave_room", "payload": {}}
                self.tcp_websocket.send(json.dumps(data))
                res = json.loads(self.tcp_websocket.recv(timeout=10))
                if res["status_code"] != 200:
                    print("uh oh")
        except TimeoutError:
            print("timeout error")
            pass
        except Exception:
            print("other exception")
            return

    def receive(self):
        self.asc.start_playback()
        self.udp_socket.settimeout(1.0)
        while not stop_event.is_set():
            try:
                data, addr = self.udp_socket.recvfrom(8192)
                if data:
                    self.asc.process_incoming_packet(data)
            except socket.timeout:
                pass
            except Exception as e:
                debug_print(e)
                stop_event.set()
                self.leave_current_room()
                break

        self.asc.stop_playback()

    def send(self, destination: str):
        self.asc.start_listening()

        # bytes_to_send = 1024 # this is what the server expects
        time.sleep(0.1)
        # Clear out any stale packets left in the network queue
        while not self.asc.network_out_queue.empty():
            try:
                self.asc.network_out_queue.get_nowait()
            except queue.Empty:
                break

        while not stop_event.is_set():
            try:
                # Blocks for up to 0.1 seconds waiting for an exact, whole neural network packet
                data = self.asc.network_out_queue.get(timeout=0.1)
                self.udp_socket.sendto(data, destination)
            except queue.Empty:
                pass  # Expected timeout if no audio is processed
            except Exception as e:
                debug_print(e)
                stop_event.set()
                self.leave_current_room()
                break

            time.sleep(0.01)

        self.asc.stop_listening()


    def chat_handler(self, client_id: str, room_address: tuple):
        stop_event.clear() # Reset the stop flag

        print(f"Starting UDP handshake with {room_address}...")

        while not stop_event.is_set():
            self.udp_socket.sendto(f"REG:{client_id}".encode(), room_address)
            try:
                data, addr = self.udp_socket.recvfrom(1024)
                if data == b"ACK_REG":
                    print("UDP Handshake acknowledged")
                    break
            except socket.timeout:
                print("Retrying UDP registration...")

        # Start threads
        listening_thread = threading.Thread(target=self.receive)
        listening_thread.start()
        sending_thread = threading.Thread(target=self.send, args=[room_address])
        sending_thread.start()

        print("Press 'q' to leave room.")
        print("Press 'm' to toggle microphone.")
        print("Press 't' to toggle compression model.")
        try:
            while not stop_event.is_set():
                inp = input().strip().lower()
                if inp == "q":
                    print("Quitting...")
                    stop_event.set()

                    sending_thread.join()
                    listening_thread.join()

                    self.leave_current_room()

                elif inp == "m":
                    if self.asc.is_listening:
                        self.asc.stop_listening()
                    else:
                        self.asc.start_listening()
                elif inp == "t":
                    if self.asc.compression_active:
                        print("Disabling audio compression.")
                        self.asc.set_compression_active(False)
                    else:
                        print("Enabling audio compression.")
                        self.asc.set_compression_active(True)

                time.sleep(0.1)
        except KeyboardInterrupt:
            stop_event.set()

            sending_thread.join()
            listening_thread.join()

            raise KeyboardInterrupt

    def main(self):
        try:
            server_addr = discover_server()
        except TimeoutError:
            print("Failed to receive response from server, exiting...")
            return
        uri = f"ws://{server_addr}:{SERVER_PORT}"
        with connect(uri) as websocket:
            self.tcp_websocket = websocket
            connected = False
            while not connected:
                res = json.loads(websocket.recv())
                if res["status_code"] == 200:
                    client_id = res["body"]["client_id"]
                    connected = True
                else:
                    time.sleep(1)

            while True:
                print("1. Create room\n2. Join room\n3. Quit\n")
                try:
                    choice = int(input())
                    # os.system('cls' if os.name == 'nt' else 'clear')

                    if choice not in [1, 2, 3]:
                        raise Exception()

                    if choice == 1:
                        print(f"Creating room {client_id}")
                        data = {"action": "create_room", "payload": {"room_id": client_id}}
                        websocket.send(json.dumps(data))
                        res = json.loads(websocket.recv())
                        if res["status_code"] == 200:
                            print(f"Successfully joined room {client_id}")
                            address = (server_addr, res["body"]["room_address"][1])
                            self.chat_handler(client_id, address)
                        else:
                            print(f"Failed to join room {room_id}")

                    elif choice == 2:
                        data = {"action": "get_available_rooms", "payload": {}}
                        websocket.send(json.dumps(data))
                        res = json.loads(websocket.recv())
                        available_rooms = res["body"]
                        if not available_rooms:
                            print("No available rooms found.\n")
                        else:
                            while True:
                                num_rooms = len(available_rooms)
                                res = """Available rooms:\n"""
                                for i, room in enumerate(available_rooms):
                                    res += f"{i + 1}. {room}\n"
                                res += f"{num_rooms + 1}. Return"
                                print(res)
                                try:
                                    choice = int(input())
                                    if choice < 1 or choice > num_rooms + 1:
                                        raise Exception()
                                    elif choice == num_rooms+1:
                                        break

                                    room_id = available_rooms[choice - 1]
                                    data = {"action": "join_room", "payload": {"room_id": room_id}}
                                    websocket.send(json.dumps(data))
                                    res = json.loads(websocket.recv())

                                    if res["status_code"] == 200:
                                        print(f"Successfully joined room {room_id}")
                                        address = (server_addr, res["body"]["room_address"][1])
                                        self.chat_handler(client_id, address)
                                        break
                                    else:
                                        print(f"Unable to join room {room_id}")
                                        pass

                                except Exception as e:
                                    debug_print(e)
                                    print("Please select a valid option.")
                    elif choice == 3:
                        websocket.close()
                        break
                except KeyboardInterrupt:
                    stop_event.set()
                    websocket.close()
                    break
                except Exception as e:
                    debug_print(e)
                    print("Please select a valid option.")

        self.udp_socket.close()


if __name__ == "__main__":
    client = Client()
    client.main()


