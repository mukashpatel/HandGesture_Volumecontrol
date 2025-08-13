import cv2
import numpy as np
import mediapipe as mp
import pyvirtualcam
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
from ctypes import cast, POINTER
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoEngine:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils
        # Initialize volume control
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = cast(interface, POINTER(IAudioEndpointVolume))
        self.volume_range = self.volume.GetVolumeRange()  # (min, max, increment) in dB
        self.min_vol, self.max_vol = self.volume_range[0], self.volume_range[1]
        # Initialize video capture and virtual camera
        self.cap = None
        self.cam = None
        self.streaming = False
        # Gesture detection parameters
        self.PINCH_THRESHOLD = 0.04  # Normalized distance for fist detection

    def is_open_hand(self, landmarks, img_width, img_height):
        """Determine if hand is open or closed based on thumb-index finger distance."""
        thumb_tip = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        dx = (thumb_tip.x - index_tip.x) * img_width
        dy = (thumb_tip.y - index_tip.y) * img_height
        distance = np.sqrt(dx**2 + dy**2) / max(img_width, img_height)
        return distance > self.PINCH_THRESHOLD  # Open hand if distance exceeds threshold

    def start_streaming(self):
        """Start video capture and processing."""
        if self.streaming:
            logger.error("Streaming already active")
            return False

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            logger.error("Could not open webcam")
            return False

        width, height = int(self.cap.get(3)), int(self.cap.get(4))
        backends = ["obs", None]  # Try OBS,millimeter, then default
        self.cam = None
        for backend in backends:
            try:
                self.cam = pyvirtualcam.Camera(width=width, height=height, fps=30, backend=backend)
                logger.info(f"Using backend: {backend if backend else 'default'}")
                break
            except RuntimeError as e:
                logger.error(f"Failed with backend {backend if backend else 'default'}: {e}")
                continue

        if not self.cam:
            logger.error("Could not initialize any virtual camera backend")
            self.cap.release()
            return False

        self.streaming = True
        return True

    def process_frame(self):
        """Process a single frame and return the processed frame."""
        if not self.streaming or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret:
            logger.error("Failed to capture frame")
            return None

        frame = cv2.flip(frame, 1)  # Flip for natural view
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        gesture = None
        current_volume = self.volume.GetMasterVolumeLevelScalar()
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                img_height, img_width = frame.shape[:2]
                if self.is_open_hand(hand_landmarks, img_width, img_height):
                    gesture = "Open Hand - Volume Up"
                    new_volume = min(1.0, current_volume + 0.05)
                    self.volume.SetMasterVolumeLevelScalar(new_volume, None)
                else:
                    gesture = "Fist - Volume Down"
                    new_volume = max(0.0, current_volume - 0.05)
                    self.volume.SetMasterVolumeLevelScalar(new_volume, None)

                # Draw bounding box around hand (approximated from landmarks)
                x_coords = [lm.x * img_width for lm in hand_landmarks.landmark]
                y_coords = [lm.y * img_height for lm in hand_landmarks.landmark]
                x1, x2 = int(min(x_coords)), int(max(x_coords))
                y1, y2 = int(min(y_coords)), int(max(y_coords))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if gesture:
            cv2.putText(frame, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame

    def send_to_virtual_camera(self, frame):
        """Send processed frame to virtual camera."""
        if frame is not None and self.cam:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.cam.send(frame_rgb)
            self.cam.sleep_until_next_frame()

    def stop_streaming(self):
        """Stop streaming and release resources."""
        self.streaming = False
        if self.cap:
            self.cap.release()
        if self.cam:
            self.cam.close()
        self.hands.close()
        logger.info("Streaming stopped")

    def run(self):
        """Main loop for video processing."""
        if not self.start_streaming():
            return

        try:
            while self.streaming:
                frame = self.process_frame()
                if frame is None:
                    break
                self.send_to_virtual_camera(frame)
        finally:
            self.stop_streaming()