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

    def _receive_image(self, sock: socket.socket) -> np.ndarray:
        try:
            img_data = self._receive_data(sock, self.img_size)
        except (ConnectionError, OSError) as e:
            print(f"Error receiving image data: {e}")
            return None

        img = np.frombuffer(img_data, np.uint8).reshape(
            self.image_height, self.image_width, 3
        )
        return img

    def _receive_label(self, sock: socket.socket) -> tuple[List[float], List[float], List[float]]:
        try:
            label_data = self._receive_data(sock, 32)
        except Exception as e:
            print(f"Error receiving pose and label data: {e}")
            return None

        label = list(struct.unpack("<8f", label_data))
    
        return label
    
    def _handle_client(self, conn: socket.socket, callback: Callable) -> None:
        while not self.stop_thread.is_set():
            image = self._receive_image(conn)
            label = self._receive_label(conn)

            callback(image, label)

    def start_receiver(self, callback: Callable) -> None:
        self.receiver_thread = threading.Thread(
            target=self._receiver_thread, args=(callback,)
        )
        self.receiver_thread.daemon = True
        self.receiver_thread.start()

    def send_floats(self, values: List[float]) -> None:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.data_host, self.data_port))
            all_values = values.copy()
            packed_data = struct.pack("<8f", *all_values)
            sock.sendall(packed_data)
            sock.close()
        except Exception as e:
            print(f"Error sending data to Unity: {e}")

    def stop(self) -> None:
        self.stop_thread.set()
        if self.receiver_thread:
            self.receiver_thread.join(timeout=2)
