import asyncio
import itertools
import json
import random
import socket
import threading
import time
from collections import defaultdict, deque

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from websockets.asyncio.server import serve
from websockets.asyncio.server import ServerConnection

DISCOVERY_PORT = 50000
SERVER_PORT = 8080

BYTES_PER_MB = 125_000

exit_event = threading.Event()

class TrafficMonitor:
    def __init__(self, window_size: int = 60):
        self.window_size = window_size

        # {client_id: num_bytes}
        self.data_store = defaultdict(int)

        self.last_check = time.time()

        # {client_id: deque[speed1, speed2, ...]}
        self.history: dict[str, deque[float]] = {}

    def update(self, client_id: str , byte_count: float):
        self.data_store[client_id] += byte_count

    def calculate_speeds(self):
        time_diff = time.time() - self.last_check
        self.last_check = time.time()

        speeds = defaultdict(float)
        if time_diff < 1:
            return speeds

        for client, num_bytes in self.data_store.items():
            speeds[client] = num_bytes / time_diff

        self.data_store.clear()

        # Update Histories(Rolling Window)
        # Identify all clients (active now OR previously active)
        all_clients = set(self.history.keys()).union(speeds.keys())

        for client in all_clients:
            if not client in self.history:
                self.history[client] = deque([0.0 for _ in range(self.window_size)], maxlen=self.window_size)

            self.history[client].append(speeds[client])

        return self.history

monitor = TrafficMonitor(window_size=60)

class DiscoveryProtocol(asyncio.DatagramProtocol):
    def __init__(self):
        super().__init__()
        self.transport = None

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        if data == b"DISCOVER_SERVER":
            self.transport.sendto(b"SERVER_HERE", addr)

class RoomException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class RoomJoinException(RoomException):
    def __init__(self, message = ""):
        self.message = message
        super().__init__(self.message)

class RoomLeaveException(RoomException):
    def __init__(self, message = ""):
        self.message = message
        super().__init__(self.message)

class RoomCreateException(RoomException):
    def __init__(self, message = ""):
        self.message = message
        super().__init__(self.message)


class Room:
    max_size = 2

    def __init__(self, room_id: str, sock: socket.socket):
        self.participants: dict[str, ServerConnection] = {}
        self.udp_registry: dict[str, tuple] = {}

        self.id: str = room_id
        self.udp_socket: socket.socket = sock

        self.listening_thread = threading.Thread(target=self.listen)
        self.listening_thread.start()

    @property
    def is_full(self):
        return len(self.participants) >= self.max_size

    def join(self, websocket: ServerConnection):
        participant_id = websocket.id.__str__()
        if self.is_full:
            raise RoomJoinException("Room full")

        if participant_id not in self.participants:
            print(f"Participant {participant_id} joined room {self.id}")

            self.participants[participant_id] = websocket
        else:
            raise RoomJoinException(f"Participant {self.id} already in room")

    def listen(self):
        self.udp_socket.settimeout(1.0)
        while not exit_event.is_set() and self.udp_socket.fileno() != -1:
            try:
                data, addr = self.udp_socket.recvfrom(800000)

                sender_id = None

                # Check if this address is already registered to a client ID
                for cid, info in self.udp_registry.items():
                    if info[0] == addr:
                        sender_id = cid
                        break

                if sender_id:
                    # Update the global monitor
                    monitor.update(sender_id, len(data))

                if data.startswith(b'REG:'):
                    client_id = data.decode().split(":")[1]
                    if client_id not in self.udp_registry:
                        self.udp_registry[client_id] = (addr, time.time())
                        self.udp_socket.sendto(b"ACK_REG", addr)
                elif len(data) > 0:
                    self.broadcast(addr,data)
            except ConnectionResetError:
                self.cleanup()
            except socket.timeout:
                pass
            except Exception as e:
                raise e

    def broadcast(self, sending_address, data):
        invalid_participants = []
        for client_id, info in self.udp_registry.items():
            target_address, last_seen = info
            if sending_address != target_address:
                try:
                    self.udp_socket.sendto(data, target_address)
                except Exception as e:
                    print(e)
                    invalid_participants.append(client_id)
            else:
                self.udp_registry[client_id] = (sending_address, time.time())

        for p in invalid_participants:
            self.leave(p)

    def cleanup(self):
        invalid_participants = []
        timeout = 10
        for client_id, info in self.udp_registry.items():
            target_address, last_seen = info
            if time.time() - last_seen >= timeout:
                invalid_participants.append(client_id)

        for p in invalid_participants:
            self.udp_registry.__delitem__(p)
            self.leave(p)

    def leave(self, participant_id):
        print(participant_id)
        print(self.participants)
        if participant_id in self.participants:
            print(f"Participant {participant_id} left room {self.id}")
            self.participants.__delitem__(participant_id)
        else:
            print("participant not found")

    def stop(self):
        self.udp_socket.close()
        if self.listening_thread.is_alive():
            self.listening_thread.join(timeout=2)

class RoomManager:
    def __init__(self):
        self.rooms: dict[str, Room] = {}

    def create_room(self, room_id: str) -> Room:
        if room_id not in self.rooms:
            print(f"Creating room {room_id}")

            server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            port = random.randint(8000,9000)
            server.bind(("0.0.0.0", port))
            print(f"UDP socket bound to port {port}")

            room = Room(room_id, server)
            self.rooms[room_id] = room
            return room
        else:
            raise RoomCreateException("Room already exists.")

    def delete_room(self, room_id: str):
        if room_id in self.rooms:
            self.rooms.__delitem__(room_id)

    @property
    def available_rooms(self) -> list:
        return list(self.rooms.keys())

    def get_room(self, room_id):
        if room_id in self.rooms:
            return self.rooms[room_id]

        return None

    def shutdown(self):
        print("Manager shutting down rooms...")
        for room in list(self.rooms.values()):
            room.stop()
        self.rooms.clear()


manager = RoomManager()

async def handler(websocket: ServerConnection):
    # remote_address returns a tuple: (ip, port)
    client_id = websocket.id.__str__()
    client_address = websocket.remote_address
    client_ip = client_address[0]
    client_port = client_address[1]
    current_room = None

    print(f"New client connection {websocket.id} from IP: {client_ip}, Port: {client_port}")

    await websocket.send(json.dumps({"status_code": 200, "body":{"client_id": client_id}}))

    try:
        async for message in websocket:
            print(message)
            data = json.loads(message)
            if "action" in data and "payload" in data:
                if data["action"] == "create_room":
                    try:
                        room_id = data["payload"]["room_id"]
                        current_room = manager.create_room(room_id)
                        current_room.join(websocket)
                        res = json.dumps({"status_code": 200, "body":{"room_address": current_room.udp_socket.getsockname()}})
                        await websocket.send(res)
                    except Exception as e:
                        res = json.dumps({"status_code": 400})
                        await websocket.send(res)
                elif data["action"] == "join_room":
                    try:
                        print(data)
                        room_id = data["payload"]["room_id"]
                        current_room = manager.get_room(room_id)
                        current_room.join(websocket)
                        res = json.dumps({"status_code": 200, "body":{"room_address": current_room.udp_socket.getsockname()}})
                        await websocket.send(res)
                    except Exception as e:
                        res = json.dumps({"status_code": 400})
                        await websocket.send(res)
                elif data["action"] == "leave_room":
                    try:
                        if not current_room:
                            raise RoomLeaveException("User is not in a room.")

                        current_room.leave(client_id)
                        if len(current_room.participants) == 0:
                            print(f"Deleting room {current_room.id}")
                            manager.delete_room(current_room.id)
                        res = json.dumps({"status_code": 200, "body":{}})
                        await websocket.send(res)
                    except Exception as e:
                        res = json.dumps({"status_code": 400, "reason": str(e)})
                        await websocket.send(res)
                elif data["action"] == "get_available_rooms":
                    try:
                        available_rooms = manager.available_rooms
                        res = json.dumps({"status_code": 200, "body": available_rooms})
                        await websocket.send(res)
                    except Exception as e:
                        print(e)
                        res = json.dumps({"status_code": 400, "reason": e})
                        await websocket.send(res)
                else:
                    msg = f'Action {data["action"]} not implemented.'
                    print(msg)
                    res = json.dumps({"status_code": 400, "reason": msg})
                    await websocket.send(res)

            else:
                msg = "Action/Payload not found in message"
                print(msg)
                res = json.dumps({"status_code": 400, "reason": msg})
                await websocket.send(res)
    except Exception as e:
        print(e)
    finally:
        print("Connection closed.")
        if current_room:
            current_room.leave(client_id)
            if len(current_room.participants) == 0:
                print(f"Deleting room {current_room.id}")
                manager.delete_room(current_room.id)

async def main():
    print("Starting server...")

    loop = asyncio.get_running_loop()

    transport, protocol = await loop.create_datagram_endpoint(
        lambda: DiscoveryProtocol(),
        local_addr=("0.0.0.0", DISCOVERY_PORT)
    )

    async with serve(handler, "0.0.0.0", SERVER_PORT) as server:
        print(f"Listening on port {SERVER_PORT}...")

        # --- REPLACEMENT FOR serve_forever() ---
        # instead of blocking forever, we block until the event is set.
        while not exit_event.is_set():
            # Check every 1 second. This keeps the loop active so
            # websockets (handler) can still process messages concurrently.
            await asyncio.sleep(1)

    transport.close()

def run_server():
    asyncio.run(main())

if __name__ == "__main__":
    server_thread = threading.Thread(target=run_server)
    server_thread.start()

    # Setup Matplotlib
    fig, ax = plt.subplots()
    ax.set_title("Real-time Network Traffic")
    ax.set_ylabel("Bytes / Sec")
    ax.set_ylim(0, 1000)
    ax.grid(True)

    lines = {}  # Store line objects {client_id: Line2D}
    colors = itertools.cycle(["cyan", "magenta", "orange", "lime", "red", "blue"])


    def update(frame):
        # Stop animation if server died
        if not server_thread.is_alive():
            plt.close(fig)
            return

        # Fetch data
        history_map = monitor.calculate_speeds()

        # Draw lines
        max_y = 100
        for client_id, data_deque in history_map.items():
            if not data_deque: continue

            # Create line if new client
            if client_id not in lines:
                line, = ax.plot([], [], label=f"Client {client_id[-4:]}", color=next(colors))
                lines[client_id] = line
                ax.legend(loc="upper left")

            # Update Line Data
            x_data = list(range(len(data_deque)))
            y_data = list(data_deque)
            lines[client_id].set_data(x_data, y_data)

            max_y = max(max(y_data), max_y)

        # Rescale axes
        ax.set_xlim(0, 60)
        ax.set_ylim(0, max_y * 1.2)

        return (ax,)

    try:
        # 3. Start Animation
        ani = FuncAnimation(fig, func=update, interval=1000)  # Update every 1000ms
        plt.show()  # This blocks the main thread!
    except KeyboardInterrupt:
        pass

    exit_event.set()
    server_thread.join(timeout=10)
    manager.shutdown()





