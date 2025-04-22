import streamlit as st
import time
import os
import av
import cv2
import numpy as np
import matplotlib.pyplot as plt
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from collections import defaultdict

st.set_page_config(page_title="Real-time Emotion Tracker", layout="wide")
st.title("Real-time Emotion Detection (Classroom Demo)")

# Parameters
RECORD_SECONDS = st.slider("Recording Duration (seconds)", 10, 60, 30)
FRAME_INTERVAL = st.slider("Analysis Interval (seconds)", 2, 10, 5)

SAVE_DIR = "demo_video_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

# Global state
emotion_counter = defaultdict(int)
dominant_emotions = []
timestamps = []
confused_count = 0
start_time = None

# Use empty container to display charts later
chart_container = st.empty()

# Define RTC configuration to avoid disconnection
RTC_CONF = RTCConfiguration({"iceServers": []})

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
                    detector_backend='opencv'
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
    webrtc_streamer(
        key="emotion-demo",
        video_processor_factory=EmotionProcessor,
        rtc_configuration=RTC_CONF,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    st.info("Recording in progress. Please wait until the selected duration ends.")

    # Wait for recording to complete
    while start_time is None:
        time.sleep(0.5)
    while time.time() - start_time < RECORD_SECONDS:
        time.sleep(1)

    st.success("Recording completed. Generating emotion analytics...")

    # Generate Charts
    if emotion_counter:
        st.subheader("Emotion Distribution")
        fig1, ax1 = plt.subplots()
        ax1.pie(emotion_counter.values(), labels=emotion_counter.keys(), autopct='%1.1f%%')
        chart_container.pyplot(fig1)

        st.subheader("Emotion Frequency")
        fig2, ax2 = plt.subplots()
        ax2.bar(emotion_counter.keys(), emotion_counter.values(), color='skyblue')
        chart_container.pyplot(fig2)

        st.subheader("Emotion Trend Over Time")
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
        chart_container.pyplot(fig3)

        st.subheader("Emotion Timeline")
        st.dataframe({"Timestamp (s)": timestamps, "Dominant Emotion": dominant_emotions})

        # Teaching suggestions
        st.subheader("Teaching Suggestion Based on Emotion Analytics")
        if confused_count >= 2:
            st.warning("Multiple signs of confusion detected. Try using more visuals or adjusting the pace.")
        if 'angry' in emotion_counter or 'fear' in emotion_counter:
            st.error("Negative emotions like anger or fear appeared. Consider pausing and engaging with students.")
        if 'happy' in emotion_counter and emotion_counter['happy'] >= 2:
            st.success("Positive engagement detected. Your current teaching approach is effective!")
        if 'neutral' in emotion_counter and emotion_counter['neutral'] >= 3:
            st.info("Many neutral responses. Consider introducing interactive elements or humor.")
    else:
        st.error("No emotions were detected. Please try again.")
