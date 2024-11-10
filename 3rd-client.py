import socket
import cv2
from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput

# Initialize the camera
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 360)}))

class ClientOutput(FileOutput):
    def __init__(self, server_address):
        super().__init__()
        self.server_address = server_address
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(self.server_address)

    def write(self, buf):
        self.sock.sendall(buf)

    def close(self):
        self.sock.close()

server_address = ('<server_ip>', 8000)  # Replace <server_ip> with the server's IP address
client_output = ClientOutput(server_address)

picam2.start_recording(MJPEGEncoder(), client_output)

try:
    while True:
        pass  # Keep the client running

except KeyboardInterrupt:
    print("Stopping client...")

finally:
    picam2.stop_recording()
    client_output.close()
