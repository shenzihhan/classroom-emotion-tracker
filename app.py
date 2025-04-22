import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase
import av
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from deepface import DeepFace
from collections import Counter

st.set_page_config(page_title="Real-time Emotion Detection (Classroom Demo)", layout="wide")
st.title("Real-time Emotion Detection (Classroom Demo)")

RECORD_SECONDS = st.slider("Recording Duration (seconds)", 10, 60, 30)
FRAME_INTERVAL = st.slider("Analysis Interval (seconds)", 2, 10, 5)

RTC_CONF = RTCConfiguration({"iceServers": []})

start_time = None
recording_done = False
captured_frames = []

class EmotionProcessor(VideoProcessorBase):
    def recv(self, frame):
        global start_time, recording_done, captured_frames

        img = frame.to_ndarray(format="bgr24")

        if start_time is None:
            start_time = time.time()

        elapsed = time.time() - start_time
        if not recording_done and int(elapsed) % FRAME_INTERVAL == 0:
            captured_frames.append(img.copy())

        if elapsed >= RECORD_SECONDS:
            recording_done = True

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def analyze_and_display_results():
    st.subheader("Analysis Results")

    results = []
    for frame in captured_frames:
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            if isinstance(result, list):
                results.append(result[0]['dominant_emotion'])
            else:
                results.append(result['dominant_emotion'])
        except Exception as e:
            results.append("error")

    filtered_results = [r for r in results if r != "error"]
    if not filtered_results:
        st.warning("No valid emotions detected.")
        return

    emotion_counts = Counter(filtered_results)
    fig, ax = plt.subplots()
    ax.bar(emotion_counts.keys(), emotion_counts.values(), color="skyblue")
    ax.set_title("Detected Emotions During Recording")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    most_common = emotion_counts.most_common(1)[0][0]
    st.success(f"Most frequently detected emotion: {most_common}")

    st.subheader("Teaching Suggestions Based on Emotions")
    suggestions = {
        "happy": "Students seem engaged. You can keep up the current teaching pace and tone!",
        "neutral": "Consider asking a few questions to check understanding and maintain attention.",
        "sad": "Try using more visual aids, stories, or examples to lift the atmosphere.",
        "angry": "There might be frustration—check if something is unclear or stressful.",
        "surprise": "Surprised faces might indicate confusion or excitement—ask for reactions.",
        "fear": "Try slowing down or creating a more relaxed classroom environment.",
        "disgust": "Content may be unappealing or offensive—check and reframe your approach."
    }
    st.info(suggestions.get(most_common, "Use engagement techniques to adapt based on emotional feedback."))

if st.button("Start Live Emotion Tracking"):
    webrtc_streamer(
        key="emotion-demo",
        video_processor_factory=EmotionProcessor,
        rtc_configuration=RTC_CONF,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    progress_placeholder = st.empty()
    while not recording_done:
        elapsed = int(time.time() - start_time) if start_time else 0
        remaining = RECORD_SECONDS - elapsed
        if remaining >= 0:
            progress_placeholder.info(f"Recording... {remaining} seconds remaining")
        time.sleep(1)

    progress_placeholder.success("Recording complete. Analyzing...")
    analyze_and_display_results()
