import streamlit as st
import cv2
import re
import pandas as pd
import pytesseract
import numpy as np
from datetime import datetime
from edge_impulse_linux.image import ImageImpulseRunner
from PIL import Image
import os

# Configure page
st.set_page_config(
    page_title="ID Card Attendance System",
    page_icon="",
    layout="centered"
)

# Initialize session state
if 'attempt_count' not in st.session_state:
    st.session_state.attempt_count = 0
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'captured_image' not in st.session_state:
    st.session_state.captured_image = None
if 'capture_mode' not in st.session_state:
    st.session_state.capture_mode = "upload"

# Constants
MODEL_PATH = "/home/parth/id_card_project/id-card-attendance-system-linux-x86_64-v2.eim"
CONFIDENCE_THRESHOLD = 0.65
MAX_ATTEMPTS = 3
ATTENDANCE_FILE = "attendance.csv"

# Title and description
st.title(" ID Card Attendance System")
st.markdown("Upload or capture an image of your ID card to mark attendance")

# Sidebar for settings
st.sidebar.header(" Settings")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.5, 
    max_value=0.95, 
    value=CONFIDENCE_THRESHOLD, 
    step=0.05
)
st.sidebar.info(f"Current attempt: {st.session_state.attempt_count + 1}/{MAX_ATTEMPTS}")

# Sidebar - Attendance Management
st.sidebar.markdown("---")
st.sidebar.header(" Attendance Management")

# Show current stats
if os.path.exists(ATTENDANCE_FILE):
    df_check = pd.read_csv(ATTENDANCE_FILE)
    total_entries = len(df_check)
    today = str(datetime.now().date())
    today_entries = len(df_check[df_check["Date"] == today])

    st.sidebar.metric("Total Records", total_entries)
    st.sidebar.metric("Today's Attendance", today_entries)
else:
    st.sidebar.info("No attendance records yet")

# Clear/Reset options
st.sidebar.markdown("###  Clear Attendance")

col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button(" Clear Today", help="Clear only today's attendance"):
        if os.path.exists(ATTENDANCE_FILE):
            df = pd.read_csv(ATTENDANCE_FILE)
            today = str(datetime.now().date())
            df = df[df["Date"] != today]
            df.to_csv(ATTENDANCE_FILE, index=False)
            st.sidebar.success(" Today's attendance cleared!")
            st.rerun()
        else:
            st.sidebar.warning("No records to clear")

with col2:
    if st.button(" Clear All", help="Clear all attendance records"):
        if os.path.exists(ATTENDANCE_FILE):
            # Backup before clearing
            backup_name = f"attendance_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df = pd.read_csv(ATTENDANCE_FILE)
            df.to_csv(backup_name, index=False)

            # Clear the file
            pd.DataFrame(columns=["Roll_No", "Date", "Time"]).to_csv(ATTENDANCE_FILE, index=False)
            st.sidebar.success(f" All records cleared!\n Backup saved: {backup_name}")
            st.rerun()
        else:
            st.sidebar.warning("No records to clear")

# New Class button
if st.sidebar.button(" Start New Class Session", help="Archive current data and start fresh", type="primary"):
    if os.path.exists(ATTENDANCE_FILE):
        # Archive with timestamp
        archive_name = f"attendance_class_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df = pd.read_csv(ATTENDANCE_FILE)
        df.to_csv(archive_name, index=False)

        # Clear current file
        pd.DataFrame(columns=["Roll_No", "Date", "Time"]).to_csv(ATTENDANCE_FILE, index=False)
        st.sidebar.success(f" New session started!\n Previous data archived: {archive_name}")
        st.rerun()
    else:
        st.sidebar.info("No previous session to archive")

def initialize_model():
    """Initialize the Edge Impulse model"""
    try:
        runner = ImageImpulseRunner(MODEL_PATH)
        runner.init()
        return runner
    except Exception as e:
        st.error(f" Error loading model: {str(e)}")
        return None

def detect_id_card(img, runner, threshold):
    """Detect ID card in the image and return confidence"""
    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        features, _ = runner.get_features_from_image(img_rgb)
        res = runner.classify(features)

        boxes = res["result"].get("bounding_boxes", [])

        if len(boxes) == 0:
            return None, 0.0, res

        best_box = max(boxes, key=lambda x: x["value"])
        confidence = best_box["value"]

        return best_box, confidence, res
    except Exception as e:
        st.error(f" Detection error: {str(e)}")
        return None, 0.0, None

def extract_roll_number(img):
    """Extract roll number from ID card image using OCR"""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        text = pytesseract.image_to_string(gray)

        # Pattern: 4 digits + 3 letters + 4 digits (e.g., 2021CSE1234)
        match = re.search(r"\d{4}[A-Z]{3}\d{4}", text)

        if not match:
            return None, text

        roll_full = match.group()
        roll_no = roll_full[-4:]  # Last 4 digits

        return roll_no, text
    except Exception as e:
        st.error(f" OCR error: {str(e)}")
        return None, ""

def mark_attendance(roll_no):
    """Mark attendance in CSV file"""
    try:
        if os.path.exists(ATTENDANCE_FILE):
            df = pd.read_csv(ATTENDANCE_FILE)
        else:
            df = pd.DataFrame(columns=["Roll_No", "Date", "Time"])

        now = datetime.now()
        today = str(now.date())

        # Check if already marked
        already_marked = df[
            (df["Roll_No"] == roll_no) & 
            (df["Date"] == today)
        ]

        if already_marked.empty:
            new_entry = pd.DataFrame({
                "Roll_No": [roll_no],
                "Date": [today],
                "Time": [now.strftime("%H:%M:%S")]
            })
            df = pd.concat([df, new_entry], ignore_index=True)
            df.to_csv(ATTENDANCE_FILE, index=False)
            return True, " Attendance marked successfully!"
        else:
            return False, " Attendance already marked today"
    except Exception as e:
        return False, f" Error marking attendance: {str(e)}"

def process_image(img, runner, threshold):
    """Process image and return results"""
    # Display uploaded image
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Captured/Uploaded Image", use_container_width=True)

    # Detect ID card
    with st.spinner(" Detecting ID card..."):
        box, confidence, res = detect_id_card(img, runner, threshold)

    # Check confidence
    if box is None or confidence < threshold:
        st.session_state.attempt_count += 1

        st.error(f" ID card confidence too low: {confidence:.2%}")
        st.warning(f" Please retry with a clearer image")

        if st.session_state.attempt_count < MAX_ATTEMPTS:
            st.info(f" Attempts remaining: {MAX_ATTEMPTS - st.session_state.attempt_count}")
            return False
        else:
            st.error(" Maximum attempts reached. Please contact administrator.")
            st.session_state.attempt_count = 0
            return False

    # ID card verified
    st.success(f" ID Card verified with confidence: {confidence:.2%}")

    # Extract roll number
    with st.spinner(" Extracting roll number..."):
        roll_no, extracted_text = extract_roll_number(img)

    with col2:
        st.subheader("Extracted Text")
        st.text_area("OCR Output", extracted_text, height=200)

    if roll_no is None:
        st.error(" Roll number not found in the image")
        st.session_state.attempt_count += 1
        if st.session_state.attempt_count < MAX_ATTEMPTS:
            st.info(f" Attempts remaining: {MAX_ATTEMPTS - st.session_state.attempt_count}")
        return False

    # Display roll number
    st.success(f" Roll Number Detected: **{roll_no}**")

    # Mark attendance
    with st.spinner(" Marking attendance..."):
        success, message = mark_attendance(roll_no)

    if success:
        st.success(message)
        st.balloons()
    else:
        st.warning(message)

    # Reset attempt count on success
    st.session_state.attempt_count = 0
    return True

# Main app
def main():
    # Choose input mode
    st.markdown("---")
    input_mode = st.radio(
        " Choose Input Method:",
        ["Upload Image", "Capture from Webcam"],
        horizontal=True
    )

    img_to_process = None

    if input_mode == "Upload Image":
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload ID Card Image",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of your ID card"
        )

        if uploaded_file is not None:
            # Convert uploaded file to cv2 image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_to_process = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    else:  # Capture from Webcam
        st.markdown("###  Webcam Capture")

        # Camera input using st.camera_input
        camera_photo = st.camera_input("Take a photo of your ID card")

        if camera_photo is not None:
            # Convert to OpenCV format
            pil_image = Image.open(camera_photo)
            img_array = np.array(pil_image)
            img_to_process = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Process button
    if img_to_process is not None:
        if st.button(" Process Image", type="primary"):
            # Initialize model
            runner = initialize_model()

            if runner is None:
                st.error(" Failed to initialize model")
                return

            try:
                success = process_image(img_to_process, runner, confidence_threshold)
            finally:
                runner.stop()

        # Reset button
        if st.session_state.attempt_count > 0:
            if st.button(" Reset Attempts"):
                st.session_state.attempt_count = 0
                st.rerun()

    # Display attendance records
    st.markdown("---")
    st.subheader(" Attendance Records")

    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
        if not df.empty:
            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                filter_option = st.selectbox(
                    "Filter by:",
                    ["All Records", "Today Only", "Last 7 Days"]
                )

            # Apply filter
            if filter_option == "Today Only":
                today = str(datetime.now().date())
                df_display = df[df["Date"] == today]
            elif filter_option == "Last 7 Days":
                from datetime import timedelta
                week_ago = str((datetime.now() - timedelta(days=7)).date())
                df_display = df[df["Date"] >= week_ago]
            else:
                df_display = df

            # Show count
            st.info(f"Showing {len(df_display)} of {len(df)} total records")

            # Display table
            st.dataframe(df_display.tail(20), use_container_width=True)

            # Download button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=" Download Attendance CSV",
                data=csv,
                file_name=f"attendance_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No attendance records yet")
    else:
        st.info("No attendance records yet")

if __name__ == "__main__":
    main()
