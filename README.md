# ✋ Air Zoom — Gesture-Based Webcam

A real-time **gesture-controlled webcam application** that lets you control **zoom and image filters using only your fingertips** — no mouse, keyboard, or physical buttons required.

The project uses **Python, OpenCV, and MediaPipe Tasks API** to detect hand landmarks and translate finger movements into webcam controls.

## ✨ Features

* 📷 Real-time webcam feed
* 🤏 **Pinch gesture for zoom control**
* 🖐️ **Finger gestures for changing filters**
* 🎨 Multiple real-time image filters
* ⚡ Smooth gesture-based interaction
* 🧠 Hand landmark detection using MediaPipe
* 🖥️ No additional hardware required

## 🎮 Gesture Controls

| Gesture                        | Action                   |
| ------------------------------ | ------------------------ |
| 🤏 Thumb + Index Finger Pinch  | Zoom out                 |
| 🫱 Thumb + Index Finger Spread | Zoom in                  |
| ☝️ Finger combinations         | Select different filters |
| 🤏 Left-hand pinch             | Adjust filter intensity  |

### Available Filters

1. **Normal** — Original webcam feed
2. **Grayscale** — Converts the image to black and white
3. **Brightness & Contrast** — Adjusts image appearance
4. **Halftone** — Creates a dot-based visual effect
5. **Pencil Sketch** — Gives the webcam a sketch effect
6. **Color Invert** — Inverts the colors

## 🛠️ Tech Stack

* **Python**
* **OpenCV** — Webcam processing and image manipulation
* **MediaPipe Tasks API** — Hand landmark detection
* **NumPy** — Numerical and image processing operations
* **Math** — Finger-distance and gesture calculations

## 🧠 How It Works

The application follows a simple pipeline:

```text
Webcam
   ↓
Capture Video Frame
   ↓
MediaPipe Hand Detection
   ↓
Detect Finger Landmarks
   ↓
Calculate Finger Distances
   ↓
Recognize Gesture
   ↓
Perform Action
   ↓
Apply Zoom / Filter
   ↓
Display Result
```

### 1. Hand Detection

MediaPipe detects the user's hand and identifies key landmarks such as:

* Thumb tip
* Index finger tip
* Middle finger tip
* Ring finger tip
* Little finger tip

### 2. Zoom Control

The distance between the **thumb tip and index-finger tip** is calculated.

```text
Thumb ● -------- ● Index
       ↑
   Finger distance
```

* Small distance → Zoom out
* Large distance → Zoom in

This allows the user to control zoom simply by **pinching and spreading their fingers in the air**.

### 3. Filter Selection

Different finger combinations are mapped to different filters.

For example:

```text
0 fingers → Normal
1 finger  → Grayscale
2 fingers → Brightness / Contrast
3 fingers → Halftone
4 fingers → Pencil Sketch
5 fingers → Color Invert
```

The exact gesture mapping can be modified in the code.

### 4. Filter Intensity

The pinch distance of the other hand can be used to control the **intensity of the selected filter**.

This makes the interaction more dynamic than simply turning a filter ON/OFF.

## 📂 Project Structure

```text
Air-Zoom/
│
├── main.py
├── requirements.txt
├── hand_landmarker.task
├── README.md
└── assets/
    └── demo.png
```

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/mani-sha12/Air-Zoom.git
cd Air-Zoom
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
opencv-python
mediapipe
numpy
```

### 4. Run the Application

```bash
python main.py
```

Allow the application to access your webcam.

## 💡 Key Concepts

This project demonstrates practical use of:

* Computer Vision
* Hand Gesture Recognition
* Hand Landmark Detection
* Real-time Image Processing
* Distance-based Gesture Recognition
* OpenCV
* MediaPipe Tasks API
* NumPy
* Human-Computer Interaction

## 🔍 Gesture Detection Logic

The application calculates the Euclidean distance between finger landmarks.

For two points:

```text
(x1, y1)
(x2, y2)
```

the distance is calculated using:

```python
distance = math.sqrt(
    (x2 - x1) ** 2 +
    (y2 - y1) ** 2
)
```

This distance is then converted into an appropriate action such as zooming or changing filter intensity.

## 🚀 Future Improvements

Possible future additions include:

* ✋ Custom user-defined gestures
* 🔄 Gesture-based camera rotation
* 💾 Capture photos using gestures
* 🎥 Start/stop video recording using gestures
* 🔊 Gesture-controlled volume
* 🎨 More advanced filters
* 👥 Multi-person hand tracking
* 📱 Gesture-controlled virtual camera
* 🖥️ Integration with video conferencing applications

## 🎯 Why This Project?

Traditional webcam applications require users to interact with buttons, sliders, or keyboard shortcuts.

**Air Zoom** explores a more natural interaction model where the user's **hands become the controller**.

Instead of clicking a zoom button:

> 🤏 Move your fingers → Zoom changes

Instead of selecting a filter from a menu:

> 🖐️ Show a finger gesture → Filter changes

## 👩‍💻 Author

**Manisha**

GitHub: [mani-sha12](https://github.com/mani-sha12)

---

⭐ If you found this project interesting, consider giving the repository a star!
