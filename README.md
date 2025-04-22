# Facial Emotion Tracker (Classroom Demo)

This Streamlit web application uses **DeepFace**, **OpenCV**, and **Streamlit WebRTC** to analyze student emotions in real-time using the webcam. It is designed for educational use cases such as classroom engagement tracking.

## Features
- Real-time webcam video capture via browser (WebRTC)
- Emotion detection using DeepFace's `mtcnn` face detector
- Visualizations: Pie chart, bar chart, and line chart for dominant emotions over time
- Intelligent recommendations for teachers based on student emotional states

## Deploying to Streamlit Cloud

### 1. **Prepare your repository structure:**

```bash
├── app.py
├── requirements.txt
├── .devcontainer/
│   └── devcontainer.json (optional)
├── README.md
```

### 2. **Main file path setting in Streamlit Cloud:**
- Repository: `your-username/your-repo-name`
- Branch: `main` (or `master`)
- Main file path: `app.py`

### 3. **requirements.txt**
Make sure this file exists with the following content:
```txt
streamlit==1.32.2
streamlit-webrtc
opencv-python-headless==4.8.0.76
deepface==0.0.93
tensorflow==2.10.0
keras==2.11.0
mtcnn==0.1.0
matplotlib==3.8.2
```

> ⚠Note: Streamlit Cloud must use `opencv-python-headless` instead of `opencv-python`

## How to Use
1. Click **Start Live Emotion Tracking**.
2. Allow browser access to webcam.
3. The system will capture frames and analyze emotion at defined intervals.
4. After the set duration, the app will show:
   - Pie chart: Emotion distribution
   - Bar chart: Frequency
   - Line chart: Emotion trend over time
   - Suggestion for teaching based on detected emotions

## Emotion Categories
The DeepFace model can detect the following:
- happy
- surprise
- neutral
- sad
- angry
- fear
- disgust

## Optional Local Usage
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then visit: `http://localhost:8501`

---
using [DeepFace](https://github.com/serengil/deepface), [Streamlit](https://streamlit.io/), and [WebRTC](https://github.com/whitphx/streamlit-webrtc).
