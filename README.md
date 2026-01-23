# ID Card Attendance System using Computer Vision and OCR

An automated ID card–based attendance system that verifies student ID cards using computer vision, extracts roll numbers via OCR, and marks attendance in real time through a Streamlit web application.


## Features

- ID card detection using an Edge Impulse image classification model  
- Roll number extraction using Tesseract OCR with regex-based validation  
- Confidence thresholding and retry control to reduce false positives  
- Supports webcam capture and image upload  
- Attendance management dashboard:
  - Duplicate attendance prevention  
  - Session-wise archiving  
  - Date-based filtering  
  - CSV export and backup  
- Interactive Streamlit user interface  


## Workflow

1. User uploads or captures an ID card image  
2. Edge Impulse model verifies the presence of a valid ID card  
3. OCR pipeline extracts text and identifies the roll number  
4. Attendance is logged with date and timestamp  
5. Records are stored and managed using CSV files  


## Tech Stack

- Language: Python  
- Computer Vision: OpenCV  
- Machine Learning Model: Edge Impulse (Image Classification)  
- OCR: Tesseract (pytesseract)  
- Web Framework: Streamlit  
- Data Handling: Pandas, NumPy  
