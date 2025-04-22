import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase
import av
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from deepface import DeepFace
from collections import Counter

st.set_page_config(page_title="Real-time Emotion Detection", layout="wide")
st.title("Real-time Emotion Detection (Classroom Demo)")

# Slider for recording duration and analysis interval
RECORD_SECONDS = st.slider("Recording Duration (seconds)", 10, 60, 30)
FRAME_INTERVAL = st.slider("Analysis Interval (seconds)", 2, 10, 5)
st.caption(f"Recording will automatically stop after {RECORD_SECONDS} seconds.")

# Configure RTC
RTC_CONF = RTCConfiguration({"iceServers": []})

# Frame buffer and timestamp
captured_frames = []
start_time = None

class EmotionProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.last_frame_time = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        global captured_frames, start_time
        img = frame.to_ndarray(format="bgr24")

        if start_time is None:
            start_time = time.time()

        elapsed = time.time() - start_time
        if elapsed <= RECORD_SECONDS and elapsed - self.last_frame_time >= FRAME_INTERVAL:
            captured_frames.append(img.copy())
            self.last_frame_time = elapsed

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Start detection
if st.button("Start Live Emotion Tracking"):
    webrtc_ctx = webrtc_streamer(
        key="emotion-demo",
        video_processor_factory=EmotionProcessor,
        rtc_configuration=RTC_CONF,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Countdown display
    status_text = st.empty()
    start_time = None

    while webrtc_ctx.state.playing:
        if start_time is None:
            start_time = time.time()
        elapsed = time.time() - start_time
        if elapsed >= RECORD_SECONDS:
            break
        status_text.info(f"Recording... {int(RECORD_SECONDS - elapsed)} seconds remaining")
        time.sleep(1)

    # Process results
    if captured_frames:
        status_text.success("Recording complete. Analyzing...")

        emotion_counts = []
        for frame in captured_frames:
            try:
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                if isinstance(result, list):
                    result = result[0]
                dominant_emotion = result['dominant_emotion']
                emotion_counts.append(dominant_emotion)
            except Exception as e:
                print(f"Emotion detection error: {e}")

        emotion_freq = Counter(emotion_counts)

        # Show result as bar chart
        fig, ax = plt.subplots()
        ax.bar(emotion_freq.keys(), emotion_freq.values(), color='coral')
        ax.set_title("Detected Emotion Distribution")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        # Teaching recommendation
        most_common = emotion_freq.most_common(1)
        if most_common:
            emo, freq = most_common[0]
            st.markdown(f"### Suggested Action for Instructors:")
            if emo in ["sad", "fear", "angry"]:
                st.info(f"Many students appeared **{emo}**. Consider slowing down or engaging with the class through a discussion or interactive activity.")
            elif emo == "happy":
                st.success("Great engagement! Keep up the positive energy.")
            else:
                st.info(f"Dominant emotion: **{emo}**. Consider checking in with students if needed.")
    else:
        st.warning("No frames captured. Please ensure your camera is active and try again.")
