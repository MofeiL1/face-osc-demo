import cv2
import mediapipe as mp
from pythonosc.udp_client import SimpleUDPClient
import argparse
from mediapipe.python.solutions import face_mesh
import math  # Add this import
import matplotlib.pyplot as plt
from collections import deque
import numpy as np

def dist(p1, p2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# ← OSC target (Wekinator default)
OSC_IP   = "127.0.0.1"
OSC_PORT = 6448

# parse a device index
parser = argparse.ArgumentParser()
parser.add_argument(
    "--device", type=int, default=0,
    help="camera device index (try 0,1,2,… until you get your built-in webcam)"
)
args = parser.parse_args()

# Initialize OSC client
client = SimpleUDPClient(OSC_IP, OSC_PORT)

# MediaPipe setup
mp_face = mp.solutions.face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    refine_landmarks=True
)

# Initialize camera with higher resolution
cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Add error checking for camera initialization
if not cap.isOpened():
    print("Error: Could not open camera device 1")
    exit()

# Initialize plotting
plt.ion()  # Enable interactive plotting
fig, ax = plt.subplots(figsize=(6, 2))
max_points = 50  # Reduced for overlay clarity

# Create deques for storing data (removed blink_data)
time_data = deque(maxlen=max_points)
mouth_data = deque(maxlen=max_points)
brow_data = deque(maxlen=max_points)

# Initialize lines
lines = []
for name, color in [("Mouth", 'r'), ("Brow", 'b')]:
    line, = ax.plot([], [], label=name, color=color)
    lines.append(line)

ax.set_ylim(0, 1.2)
ax.set_xlabel("Time")
ax.set_ylabel("Value")
ax.legend()
plt.grid(True)
ax.set_facecolor((0, 0, 0, 0.5))  # Semi-transparent background
fig.patch.set_alpha(0.5)  # Semi-transparent figure

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert and process
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face.process(rgb)

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            lm   = face.landmark

            # draw mesh lines using MediaPipe's face mesh connections
            h, w, _ = frame.shape
            points = []
            for p in lm:
                x, y = int(p.x * w), int(p.y * h)
                points.append((x, y))
            
            # Draw face mesh connections using MediaPipe's FACEMESH_TESSELATION
            for connection in face_mesh.FACEMESH_TESSELATION:
                start_idx = connection[0]
                end_idx = connection[1]
                
                start_point = points[start_idx]
                end_point = points[end_idx]
                
                cv2.line(frame, start_point, end_point, (255, 255, 255), 1)

            # compute mouth open
            top    = lm[13]; bottom = lm[14]
            mouth_open = bottom.y - top.y  # Removed max(0.0) and multiplication by 10

            # Improved eyebrow tracking without restrictions
            brow_points = [276, 282, 283]  # outer eyebrow points
            eye_points = [386, 374, 373]   # upper eye points
            
            brow_y = sum(lm[i].y for i in brow_points) / len(brow_points)
            eye_y = sum(lm[i].y for i in eye_points) / len(eye_points)
            
            # Calculate raw distance without offset or scaling
            brow_raise = eye_y - brow_y  # Removed offset and scaling

            # send over OSC
            client.send_message("/wek/inputs", [brow_raise, mouth_open])

            # overlay text
            cv2.putText(frame, f"Mouth: {mouth_open:.8f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, f"Brow: {brow_raise:.8f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            # Update plot data
            time_data.append(len(time_data))
            mouth_data.append(mouth_open)
            brow_data.append(brow_raise)

            # Update line data
            for line, data in zip(lines, [mouth_data, brow_data]):
                line.set_xdata(np.arange(len(data)))
                line.set_ydata(data)

            # Adjust plot limits
            ax.set_xlim(max(0, len(time_data) - max_points), len(time_data))
            
            # Convert plot to image for overlay - fixed for Retina displays
            fig.canvas.draw()
            
            # Get buffer and calculate actual dimensions from buffer size
            buf = fig.canvas.tostring_argb()
            buf_size = len(buf)
            
            # Calculate actual dimensions for Retina display (2x scale factor)
            w, h = fig.canvas.get_width_height()
            scale = int(np.sqrt(buf_size / (w * h * 4)))
            actual_w, actual_h = w * scale, h * scale
            
            # Reshape using actual dimensions
            plot_img = np.frombuffer(buf, dtype=np.uint8)
            plot_img = plot_img.reshape(actual_h, actual_w, 4)
            
            # Post-process and resize
            plot_img = plot_img[:, :, 1:]  # Remove alpha channel
            plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)
            plot_img = cv2.resize(plot_img, (320, 120))  # Resize for overlay
            
            # Overlay plot on frame
            h, w = plot_img.shape[:2]
            frame[20:20+h, frame.shape[1]-340:frame.shape[1]-20] = plot_img

        # Display (optional)
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
