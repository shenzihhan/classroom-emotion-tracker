import streamlit as st
import time
import os
import av
import cv2
import numpy as np
import matplotlib.pyplot as plt
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from collections import defaultdict

st.set_page_config(page_title="Facial Emotion Tracker", layout="centered")
st.title("Real-time Emotion Detection (Classroom Demo)")

# Parameters
RECORD_SECONDS = st.slider("Recording Duration (seconds)", 10, 60, 30)
FRAME_INTERVAL = st.slider("Analysis Interval (seconds)", 2, 10, 5)

SAVE_DIR = "demo_video_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

# Global state to store emotions
emotion_counter = defaultdict(int)
dominant_emotions = []
timestamps = []
confused_count = 0
start_time = None

class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_capture_time = 0
        self.capture_count = 0

    def recv(self, frame):
        global start_time, emotion_counter, dominant_emotions, timestamps, confused_count

        img = frame.to_ndarray(format="bgr24")
        current_time = time.time()

        if start_time is None:
            start_time = current_time

        elapsed = current_time - start_time

        if elapsed >= RECORD_SECONDS:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        if current_time - self.last_capture_time >= FRAME_INTERVAL:
            timestamp_sec = int(elapsed)
            img_path = os.path.join(SAVE_DIR, f"frame_{timestamp_sec}.jpg")
            cv2.imwrite(img_path, img)
            self.capture_count += 1

            try:
                results = DeepFace.analyze(
                    img_path=img_path,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='mtcnn'
                )
                if isinstance(results, list):
                    for res in results:
                        emotions = res['emotion']
                        dominant = res['dominant_emotion']
                        dominant_emotions.append(dominant)
                        timestamps.append(timestamp_sec)
                        emotion_counter[dominant] += 1

                        if emotions.get('surprise', 0) > 30 and emotions.get('neutral', 0) > 30:
                            confused_count += 1
                        elif emotions.get('sad', 0) > 20 and emotions.get('fear', 0) > 20:
                            confused_count += 1
            except Exception as e:
                st.warning(f"Emotion analysis failed at {timestamp_sec}s: {e}")

            self.last_capture_time = current_time

        return av.VideoFrame.from_ndarray(img, format="bgr24")

if st.button("Start Live Emotion Tracking"):
    webrtc_streamer(key="emotion-demo", video_processor_factory=EmotionProcessor)
    st.info("Recording in progress... Please wait until it reaches selected duration.")

    # Wait until recording is done
    while start_time is None:
        time.sleep(0.5)
    while time.time() - start_time < RECORD_SECONDS:
        time.sleep(1)

    st.success("Recording completed and analyzed.")

    # Pie Chart
    st.subheader("Emotion Distribution")
    if emotion_counter:
        fig1, ax1 = plt.subplots()
        ax1.pie(emotion_counter.values(), labels=emotion_counter.keys(), autopct='%1.1f%%')
        st.pyplot(fig1)

        # Bar Chart
        st.subheader("Emotion Frequency")
        fig2, ax2 = plt.subplots()
        ax2.bar(emotion_counter.keys(), emotion_counter.values(), color='skyblue')
        st.pyplot(fig2)

        # Trend Chart
        emotion_map = {'happy': 5, 'surprise': 4, 'neutral': 3, 'sad': 2, 'angry': 1}
        emotion_numeric = [emotion_map.get(e, 0) for e in dominant_emotions]
        fig3, ax3 = plt.subplots()
        ax3.plot(timestamps, emotion_numeric, marker='o')
        ax3.set_yticks(list(emotion_map.values()))
        ax3.set_yticklabels(list(emotion_map.keys()))
        ax3.set_xlabel("Time (sec)")
        ax3.set_ylabel("Dominant Emotion")
        ax3.set_title("Emotion Trend Over Time")
        ax3.grid(True)
        st.subheader("Emotion Trend")
        st.pyplot(fig3)

        # Suggestion for Teacher
        st.subheader("Suggestion For Teacher")
        if confused_count >= 2:
            st.warning(f"Detected signs of confusion {confused_count} times. Try slowing down or using more visuals.")
        if 'angry' in emotion_counter or 'fear' in emotion_counter:
            st.error("Negative emotions detected. Consider using stress-relief or check student feedback.")
        if 'happy' in emotion_counter and emotion_counter['happy'] >= 2:
            st.success("Students appear engaged. Current methods are working well!")
        if 'neutral' in emotion_counter and emotion_counter['neutral'] >= 3:
            st.info("Mostly neutral. Consider adding interaction or humor.")
    else:
        st.error("No emotions were detected. Please try again.")
        
