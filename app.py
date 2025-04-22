import streamlit as st
import cv2
import time
import os
from deepface import DeepFace
import matplotlib.pyplot as plt
from collections import defaultdict

st.set_page_config(page_title="Facial Emotion Tracker", layout="centered")
st.title("Real-time Emotion Detection (Classroom Demo)")

# Parameters
RECORD_SECONDS = st.slider("Recording Duration (seconds)", 10, 60, 30)
FRAME_INTERVAL = st.slider("Analysis Interval (seconds)", 2, 10, 5)

if st.button("Start Recording"):
    SAVE_DIR = "demo_video_frames"
    os.makedirs(SAVE_DIR, exist_ok=True)
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('emotion_demo_video.mp4', fourcc, 10.0, (frame_width, frame_height))

    st.info("Recording started...")
    start_time = time.time()
    last_capture_time = start_time

    emotion_counter = defaultdict(int)
    dominant_emotions = []
    timestamps = []
    confused_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        current_time = time.time()
        elapsed = current_time - start_time

        if current_time - last_capture_time >= FRAME_INTERVAL:
            timestamp_sec = int(elapsed)
            img_path = os.path.join(SAVE_DIR, f"frame_{timestamp_sec}.jpg")
            cv2.imwrite(img_path, frame)

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

            last_capture_time = current_time

        if elapsed >= RECORD_SECONDS:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
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
