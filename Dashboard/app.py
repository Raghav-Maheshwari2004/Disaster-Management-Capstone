import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np
import os
import pandas as pd

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Disaster Rescue AI (Analytics Mode)",
    page_icon="🚁",
    layout="wide"
)

# --- 2. SIDEBAR SETTINGS ---
st.sidebar.title("🔧 Control Panel")
confidence_score = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)
# model_type = st.sidebar.radio("Select Model", ["YOLOv11 Nano", "YOLOv11 Small"])

# Load Model
@st.cache_resource
def load_model():
    # Attempt to load custom best.pt, fallback to standard if missing
   
        return YOLO("best.pt") 
   

try:
    model = load_model()
    st.sidebar.success("✅ AI Engine Online")
except Exception as e:
    st.sidebar.error(f"❌ Error loading model: {e}")

# --- 3. ALGORITHMS & METRICS ---

def calculate_metrics(image):
    """
    Calculates 'Sharpness' using Laplacian Variance.
    Higher score = More details visible (less fog).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score

def standard_dehaze(image_bgr):
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    limg = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def sage_algorithm(image_bgr):
    # 1. Red Boost
    b, g, r = cv2.split(image_bgr)
    r_boosted = cv2.addWeighted(r, 1.2, np.zeros_like(r), 0, 0)
    img_red_biased = cv2.merge((b, g, r_boosted))
    
    # 2. Aggressive CLAHE
    lab = cv2.cvtColor(img_red_biased, cv2.COLOR_BGR2LAB)
    l, a, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    limg = cv2.merge((l_enhanced, a, b_channel))
    enhanced_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # 3. Sharpening
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    final_output = cv2.filter2D(enhanced_bgr, -1, kernel)
    
    return final_output

# --- 4. MAIN DASHBOARD ---
st.title("🚁 AI Rescue: Comparative Analytics")

app_mode = st.radio("Select Input Mode:", ["Image Analysis", "Live Video"])

if app_mode == "Image Analysis":
    uploaded_file = st.file_uploader("Upload Hazy/Foggy Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_raw = cv2.imdecode(file_bytes, 1)
        
        # PROCESS IMAGES
        img_standard = standard_dehaze(img_raw)
        img_sage = sage_algorithm(img_raw)
        
        # --- DISPLAY 3-PANEL COMPARISON ---
        st.divider()
        st.subheader("1. Visual Comparison")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB), caption="Raw Input", use_column_width=True)
        with col2:
            st.image(cv2.cvtColor(img_standard, cv2.COLOR_BGR2RGB), caption="Standard Dehaze", use_column_width=True)
        with col3:
            st.image(cv2.cvtColor(img_sage, cv2.COLOR_BGR2RGB), caption="SAGE (Ours)", use_column_width=True)

        # --- AI & ANALYTICS SECTION ---
        st.divider()
        
        # CHECKBOX for Comparison Mode
        show_raw_comparison = st.checkbox("🔍 Compare AI Performance (Raw vs. SAGE)", value=False)
        
        if st.button("🚀 Run Analysis & Detection"):
            
            # 1. Calculate Data for Graph
            raw_score = calculate_metrics(img_raw)
            std_score = calculate_metrics(img_standard)
            sage_score = calculate_metrics(img_sage)
            
            # Create Dataframe for Line Chart
            chart_data = pd.DataFrame({
                'Image Quality (Sharpness Score)': [raw_score, std_score, sage_score],
                'Algorithm': ['1. Raw', '2. Standard', '3. SAGE (Ours)']
            }).set_index('Algorithm')

            # 2. Run YOLO on SAGE (Always)
            results_sage = model.predict(img_sage, conf=confidence_score)
            res_plotted_sage = results_sage[0].plot()
            
            # 3. Conditional: Run YOLO on Raw (If checkbox is True)
            res_plotted_raw = None
            if show_raw_comparison:
                results_raw = model.predict(img_raw, conf=confidence_score)
                res_plotted_raw = results_raw[0].plot()

            # --- DISPLAY RESULTS (Dynamic Layout) ---
            st.subheader("2. Performance Metrics & Detection")
            
            if show_raw_comparison:
                # 3-Column Layout: [Graph] | [Raw Detect] | [SAGE Detect]
                res_col1, res_col2, res_col3 = st.columns([1, 1, 1])
                
                with res_col1:
                    st.write("**Visibility Curve**")
                    st.line_chart(chart_data)
                    st.info(f"📈 SAGE Improvement: **{((sage_score-raw_score)/raw_score)*100:.1f}%**")

                with res_col2:
                    st.write("**⚠️ Detection on Raw (Original)**")
                    st.image(cv2.cvtColor(res_plotted_raw, cv2.COLOR_BGR2RGB), caption="Without SAGE", use_column_width=True)
                    st.write(f"Objects Found: {len(results_raw[0].boxes)}")

                with res_col3:
                    st.write("**✅ Detection on SAGE (Ours)**")
                    st.image(cv2.cvtColor(res_plotted_sage, cv2.COLOR_BGR2RGB), caption="With SAGE", use_column_width=True)
                    st.write(f"Objects Found: {len(results_sage[0].boxes)}")
                    
            else:
                # 2-Column Layout (Cleaner, SAGE only)
                res_col1, res_col2 = st.columns([1, 1]) 
                
                with res_col1:
                    st.write("**Visibility Improvement Curve**")
                    st.line_chart(chart_data)
                    st.info(f"📈 SAGE improved sharpness by **{((sage_score-raw_score)/raw_score)*100:.1f}%**")

                with res_col2:
                    st.write("**Final AI Output (SAGE Only)**")
                    st.image(cv2.cvtColor(res_plotted_sage, cv2.COLOR_BGR2RGB), caption="Final Detection", use_column_width=True)
                    
                    boxes = results_sage[0].boxes
                    if len(boxes) > 0:
                        st.success(f"✅ Detected {len(boxes)} objects.")
                    else:
                        st.warning("No objects detected.")

elif app_mode == "Live Video":
    uploaded_video = st.file_uploader("Upload Drone Video", type=["mp4", "avi"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") 
        tfile.write(uploaded_video.getbuffer())
        tfile.close()
        
        vf = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        
        while vf.isOpened():
            ret, frame = vf.read()
            if not ret:
                break
            
            # Only process SAGE for video to keep speed high
            enhanced_frame = sage_algorithm(frame)
            results = model.predict(enhanced_frame, conf=confidence_score)
            res_plotted = results[0].plot()
            
            stframe.image(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB), caption="SAGE Real-Time Analysis", use_column_width=True)
            
        vf.release()
        os.remove(tfile.name)