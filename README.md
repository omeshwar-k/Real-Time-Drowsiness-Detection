# 🚗 Real-Time Drowsiness Detection

## 📌 About
A real-time drowsiness detection system that uses a webcam to monitor eye closure
and plays an alert sound when drowsiness is detected.

## 🧠 What is used?

- 📷 Webcam – live video input
- 👁️ Eye Aspect Ratio (EAR) – detects eye closure
- 🤖 Pre-trained facial landmark model (dlib)
- 🔊 Alert sound (beep.wav)

## 🛠️ Technologies Used

- 🐍 Python
- 📷 OpenCV
- 🤖 dlib
- 🔢 NumPy
- 📐 SciPy
- 🔊 winsound (Windows)

## ⚙️ How it works

- 👀 Eyes open → no sound  
- 😴 Eyes closed for few seconds → beep sound ON  
- 🙂 Eyes open again → beep sound OFF
