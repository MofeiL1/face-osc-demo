import cv2
import mediapipe as mp
from pythonosc.udp_client import SimpleUDPClient
import argparse
from mediapipe.python.solutions import face_mesh
import math  # Add this import
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # More stable for real-time plotting
import time

last_plot_update = time.time()

def dist(p1, p2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# OSC target (Wekinator default)
OSC_IP   = "127.0.0.1"
OSC_PORT = 6448

# Parse a device index
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
cap = cv2.VideoCapture(args.device, cv2.CAP_AVFOUNDATION)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Add error checking for camera initialization
if not cap.isOpened():
    print("Error: Could not open camera device 1")
    exit()

# Initialize plotting with larger size
plt.ion()  # Enable interactive plotting
fig, ax = plt.subplots(figsize=(10, 4), dpi=80)  # Lower DPI to reduce memory
max_points = 50  # History length

# Create deques for all facial parameters
time_data = deque(maxlen=max_points)
feature_data = {
    "Mouth Open": deque(maxlen=max_points),
    "Mouth Width": deque(maxlen=max_points),
    "L Brow": deque(maxlen=max_points),
    "R Brow": deque(maxlen=max_points),
    "L Eye": deque(maxlen=max_points),
    "R Eye": deque(maxlen=max_points),
    "Brow Contract": deque(maxlen=max_points),
    "Cheek Raise": deque(maxlen=max_points),
    "Mouth Corner": deque(maxlen=max_points),
    "Head Tilt": deque(maxlen=max_points),
    "L Corner H": deque(maxlen=max_points),
    "R Corner H": deque(maxlen=max_points),
    "Mouth Asym": deque(maxlen=max_points)
}

# Initialize lines with color palette for better distinction - all solid lines
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', 
          '#a65628', '#f781bf', '#999999', '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']

lines = {}
for i, (name, color) in enumerate(zip(feature_data.keys(), colors)):
    # All lines are solid now
    line, = ax.plot([], [], label=name, color=color, linestyle='-')
    lines[name] = line

# Setup the plot appearance - with fixed y-axis range
ax.set_ylim(-0.1, 0.6)  # Fixed range with maximum at 0.6
ax.set_xlabel("Time")
ax.set_ylabel("Value")
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5, fontsize=8)
plt.grid(True)
ax.set_facecolor((1, 1, 1, 1))  # Solid white background for separate window
fig.patch.set_alpha(1.0)  # Solid figure background
plt.tight_layout()

try:
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera")
                break

            # Convert and process
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_face.process(rgb)

            if results.multi_face_landmarks:
                face = results.multi_face_landmarks[0]
                lm   = face.landmark

                # Draw mesh lines using MediaPipe's face mesh connections
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

                # Calculate face size for normalization
                # Using vertical distance between top of face and chin
                top_face = lm[10]     # Top of forehead
                bottom_face = lm[152]  # Bottom of chin
                face_height = bottom_face.y - top_face.y
                
                # Calculate face width for additional normalization reference
                left_face = lm[234]   # Left cheek
                right_face = lm[454]  # Right cheek
                face_width = dist(left_face, right_face)
                
                # Feature 1: Mouth openness (vertical)
                top_lip = lm[13]
                bottom_lip = lm[14]
                mouth_open = dist(top_lip, bottom_lip) / face_height
                
                # Feature 2: Mouth width (smile detection)
                left_mouth = lm[61]
                right_mouth = lm[291]
                mouth_width = dist(left_mouth, right_mouth) / face_width
                
                # Feature 3: Left eyebrow raise
                left_brow_points = [336, 296, 334, 293, 300]  # inner to outer
                left_eye_points = [362, 398, 384, 386, 387]   # corresponding eye points
                
                left_brow_y = sum(lm[i].y for i in left_brow_points) / len(left_brow_points)
                left_eye_y = sum(lm[i].y for i in left_eye_points) / len(left_eye_points)
                left_brow_raise = (left_eye_y - left_brow_y) / face_height
                
                # Feature 4: Right eyebrow raise
                right_brow_points = [107, 66, 105, 63, 70]  # inner to outer
                right_eye_points = [133, 173, 157, 159, 160]  # corresponding eye points
                
                right_brow_y = sum(lm[i].y for i in right_brow_points) / len(right_brow_points)
                right_eye_y = sum(lm[i].y for i in right_eye_points) / len(right_eye_points)
                right_brow_raise = (right_eye_y - right_brow_y) / face_height
                
                # Feature 5: Left eye openness
                left_eye_top = lm[386]  # upper eyelid
                left_eye_bottom = lm[374]  # lower eyelid
                left_eye_open = dist(left_eye_top, left_eye_bottom) / face_height
                
                # Feature 6: Right eye openness
                right_eye_top = lm[159]  # upper eyelid
                right_eye_bottom = lm[145]  # lower eyelid
                right_eye_open = dist(right_eye_top, right_eye_bottom) / face_height
                
                # Feature 7: Frown/concentration (eyebrows moving inward/down)
                left_inner_brow = lm[336]
                right_inner_brow = lm[107]
                brow_contraction = dist(left_inner_brow, right_inner_brow) / face_width
                
                # Feature 8: Cheek raise (genuine smile indicator)
                left_cheek = lm[117]
                right_cheek = lm[346]
                cheek_raise = ((left_cheek.y + right_cheek.y) / 2 - top_face.y) / face_height
                
                # Feature 9: Mouth corner raise (smile/frown)
                mouth_corner_left = lm[61]
                mouth_corner_right = lm[291]
                mouth_center_y = (top_lip.y + bottom_lip.y) / 2
                mouth_corner_raise = ((mouth_corner_left.y + mouth_corner_right.y) / 2 - mouth_center_y) / face_height
                
                # Feature 11: Left mouth corner horizontal position
                mouth_center_x = (left_mouth.x + right_mouth.x) / 2
                left_corner_horizontal = (mouth_corner_left.x - mouth_center_x) / face_width
                
                # Feature 12: Right mouth corner horizontal position
                right_corner_horizontal = (mouth_corner_right.x - mouth_center_x) / face_width
                
                # Feature 13: Mouth asymmetry (for smirk detection)
                mouth_asymmetry = (mouth_corner_left.y - mouth_corner_right.y) / face_height
                
                # Feature 10: Head tilt (basic pose estimation)
                left_eye_center = [(lm[33].x + lm[133].x) / 2, (lm[33].y + lm[133].y) / 2]
                right_eye_center = [(lm[263].x + lm[362].x) / 2, (lm[263].y + lm[362].y) / 2]
                head_tilt = math.atan2(right_eye_center[1] - left_eye_center[1], 
                                    right_eye_center[0] - left_eye_center[0]) / math.pi
                
                # Send all normalized facial features over OSC (13 features total)
                facial_features = [
                    mouth_open,             # 1. Mouth openness
                    mouth_width,            # 2. Mouth width (smile)
                    left_brow_raise,        # 3. Left eyebrow raise  
                    right_brow_raise,       # 4. Right eyebrow raise
                    left_eye_open,          # 5. Left eye openness
                    right_eye_open,         # 6. Right eye openness
                    brow_contraction,       # 7. Brow contraction (concentration)
                    cheek_raise,            # 8. Cheek raise (genuine smile)
                    mouth_corner_raise,     # 9. Mouth corner position (smile/frown)
                    head_tilt,              # 10. Head tilt (basic pose)
                    left_corner_horizontal, # 11. Left mouth corner horizontal
                    right_corner_horizontal,# 12. Right mouth corner horizontal
                    mouth_asymmetry         # 13. Mouth asymmetry (smirk)
                ]
                
                client.send_message("/wek/inputs", facial_features)

                # Display ALL metrics on screen (organized in two columns)
                metrics = [
                    # Basic facial features
                    ["Mouth Open", mouth_open],
                    ["Mouth Width", mouth_width],
                    ["L Brow", left_brow_raise],
                    ["R Brow", right_brow_raise], 
                    ["L Eye", left_eye_open],
                    ["R Eye", right_eye_open],
                    ["Head Tilt", head_tilt],
                    
                    # Advanced expression features
                    ["Brow Contract", brow_contraction],
                    ["Cheek Raise", cheek_raise],
                    ["Mouth Corner", mouth_corner_raise],
                    
                    # Mouth corner features
                    ["L Corner H", left_corner_horizontal],
                    ["R Corner H", right_corner_horizontal],
                    ["Mouth Asym", mouth_asymmetry]
                ]
                
                # Display metrics in two columns to reduce screen clutter
                y_offset = 30
                x_left = 10
                x_right = 200
                
                for i, (metric, value) in enumerate(metrics):
                    x_pos = x_right if i >= 7 else x_left  # First 7 in left column, rest in right
                    y_pos = 30 + (i % 7) * 20              # Maximum 7 metrics per column
                    cv2.putText(frame, f"{metric}: {value:.2f}",
                                (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                # Update plot data with all metrics
                time_data.append(len(time_data))
                
                # Update all feature data
                feature_values = {
                    "Mouth Open": mouth_open,
                    "Mouth Width": mouth_width,
                    "L Brow": left_brow_raise,
                    "R Brow": right_brow_raise,
                    "L Eye": left_eye_open,
                    "R Eye": right_eye_open,
                    "Brow Contract": brow_contraction,
                    "Cheek Raise": cheek_raise,
                    "Mouth Corner": mouth_corner_raise,
                    "Head Tilt": head_tilt,
                    "L Corner H": left_corner_horizontal,
                    "R Corner H": right_corner_horizontal,
                    "Mouth Asym": mouth_asymmetry
                }
                
                # Throttle plot updates to reduce CPU/GPU load (update at max 20fps)
                current_time = time.time()
                if current_time - last_plot_update > 0.05:  # 20fps = update every 0.05s
                    # Update all lines
                    for name, value in feature_values.items():
                        feature_data[name].append(value)
                        lines[name].set_xdata(np.arange(len(feature_data[name])))
                        lines[name].set_ydata(feature_data[name])
                    
                    # Keep fixed y-axis range
                    ax.set_ylim(-0.1, 0.6)
                    
                    # Adjust plot limits for x-axis
                    ax.set_xlim(max(0, len(time_data) - max_points), len(time_data))
                    
                    # Update the plot as a separate window
                    plt.draw()
                    plt.pause(0.001)  # Small pause to allow plot to update
                    
                    last_plot_update = current_time
                else:
                    # Still collect data even when not updating plot
                    for name, value in feature_values.items():
                        feature_data[name].append(value)

                # Add explicit garbage collection periodically
                if len(time_data) % 100 == 0:  # Every 100 frames
                    import gc
                    gc.collect()  # Force garbage collection

                # Remove all the plot overlay code - not needed anymore
                # No need for: fig.canvas.draw(), buffer conversion, etc.

            # Display video feed without plot overlay
            cv2.imshow("Webcam", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break

        except Exception as e:
            print(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
            break  # Exit on error

finally:
    print("Cleaning up resources...")
    cap.release()
    cv2.destroyAllWindows()
    plt.close(fig)  # Make sure to close the matplotlib window too
    print("Cleanup complete")
