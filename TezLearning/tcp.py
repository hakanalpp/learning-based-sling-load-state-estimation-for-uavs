import socket
import struct
import threading
import time
from typing import Callable, List

import numpy as np


class TCPHandler:
    def __init__(
        self,
        image_host: str = "0.0.0.0",
        image_port: int = 10001,
        data_host: str = "127.0.0.1",
        data_port: int = 9997,
        image_dimensions: tuple = (224, 224),
    ):
        self.image_host = image_host
        self.image_port = image_port
        self.data_host = data_host
        self.data_port = data_port
        self.image_width, self.image_height = image_dimensions
        self.img_size = self.image_width * self.image_height * 3
        self.stop_thread = threading.Event()

        self.receiver_thread = None

    def _receiver_thread(self, callback: Callable) -> None:
        print(f"Starting image receiver on {self.image_host}:{self.image_port}")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.image_host, self.image_port))
            s.listen()
            s.settimeout(1.0)
            print(f"Image receiver listening on {self.image_host}:{self.image_port}")
            while not self.stop_thread.is_set():
                try:
                    conn, addr = s.accept()
                    print(f"Image client connected from {addr}")
                    try:
                        self._handle_client(conn, callback)
                    except Exception as e:
                        print(f"Error handling image client: {e}")
                    finally:
                        conn.close()
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"Error in image receiver: {e}")
                    time.sleep(1)
        print("Image receiver thread exiting")

    def _receive_data(self, sock: socket.socket, size: int) -> bytearray:
        buf = bytearray()
        while len(buf) < size:
            try:
                pkt = sock.recv(min(4096, size - len(buf)))
                if not pkt:
                    raise ConnectionError("Connection closed while receiving data")
                buf.extend(pkt)
            except Exception as e:
                print(f"Error receiving data: {e}")
                raise
        return buf

    def _handle_client(self, conn: socket.socket, callback: Callable) -> None:
        while not self.stop_thread.is_set():
            try:
                img_data = self._receive_data(conn, self.img_size)
            except (ConnectionError, OSError) as e:
                print(f"Error receiving image data: {e}")
                break

            try:
                pose_data = self._receive_data(conn, 28)
                position_data = pose_data[:12]
                x, y, z = struct.unpack("<3f", position_data)
                rotation_data = pose_data[12:]
                qx, qy, qz, qw = struct.unpack("<4f", rotation_data)
            except Exception as e:
                print(f"Error receiving drone pose data: {e}")
                break

            img = np.frombuffer(img_data, np.uint8).reshape(
                self.image_height, self.image_width, 3
            )
            callback(img, [x, y, z], [qx, qy, qz, qw])

    def start_receiver(self, callback: Callable) -> None:
        self.receiver_thread = threading.Thread(
            target=self._receiver_thread, args=(callback,)
        )
        self.receiver_thread.daemon = True
        self.receiver_thread.start()

    def send_floats(
        self, values: List[float], position: List[float], rotation: List[float]
    ) -> None:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.data_host, self.data_port))
            all_values = values.copy()
            all_values.extend(position)
            all_values.extend(rotation)
            packed_data = struct.pack("<15f", *all_values)
            sock.sendall(packed_data)
            sock.close()
        except Exception as e:
            print(f"Error sending data to Unity: {e}")

    def stop(self) -> None:
        self.stop_thread.set()
        if self.receiver_thread:
            self.receiver_thread.join(timeout=2)
