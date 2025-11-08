# Install Streamlit and other dependencies
pip install streamlit ultralytics opencv-python pillow numpy
# streamlit_app.py
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# Set page configuration
st.set_page_config(
    page_title="Surgical Tools Detection",
    page_icon="🔪",
    layout="wide"
)

# Load model with caching
@st.cache_resource
def load_model():
    try:
        # Load your trained model
        model = YOLO('runs/detect/train/weights/best.pt')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    st.title("🔪 Surgical Tools Detection")
    st.write("Upload an image to detect surgical tools using YOLOv8")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("Could not load the model. Please check if the model file exists.")
        return
    
    # Sidebar for additional options
    st.sidebar.title("Settings")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.5,
        help="Adjust the minimum confidence level for detections"
    )
    
    # File upload section
    st.header("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload an image containing surgical tools"
    )
    
    if uploaded_file is not None:
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Display image info
            st.write(f"**Image Size:** {image.size}")
            st.write(f"**Image Mode:** {image.mode}")
        
        with col2:
            st.subheader("Detection Results")
            
            # Add a predict button
            if st.button("🔍 Detect Surgical Tools", type="primary"):
                with st.spinner("Processing image..."):
                    try:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                            image.save(tmp_file.name)
                            
                            # Run prediction with confidence threshold
                            results = model.predict(
                                tmp_file.name, 
                                conf=confidence_threshold,
                                save=False  # We'll handle saving ourselves
                            )
                        
                        # Process results
                        result_image = results[0].plot()
                        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                        
                        # Display result image
                        st.image(result_image_rgb, caption='Detection Results', use_column_width=True)
                        
                        # Show detection statistics
                        boxes = results[0].boxes
                        if boxes is not None and len(boxes) > 0:
                            st.success(f"✅ Detected {len(boxes)} surgical tool(s)")
                            
                            # Display detection details in an expandable section
                            with st.expander("📊 Detection Details"):
                                st.subheader("Detected Tools:")
                                
                                # Count tools by class
                                tool_counts = {}
                                detection_data = []
                                
                                for i, box in enumerate(boxes):
                                    class_id = int(box.cls)
                                    class_name = model.names[class_id]
                                    confidence = float(box.conf)
                                    
                                    # Update counts
                                    if class_name in tool_counts:
                                        tool_counts[class_name] += 1
                                    else:
                                        tool_counts[class_name] = 1
                                    
                                    detection_data.append({
                                        'id': i + 1,
                                        'tool': class_name,
                                        'confidence': confidence,
                                        'bbox': box.xyxy[0].tolist()
                                    })
                                
                                # Show summary
                                st.write("**Summary:**")
                                for tool, count in tool_counts.items():
                                    st.write(f"- {tool}: {count}")
                                
                                # Show detailed table
                                st.write("**Detailed Results:**")
                                for detection in detection_data:
                                    st.write(f"**{detection['id']}. {detection['tool']}**")
                                    st.write(f"   Confidence: {detection['confidence']:.2%}")
                                    st.write(f"   Bounding Box: {[round(x, 1) for x in detection['bbox']]}")
                                    st.write("---")
                            
                            # Download button for results
                            st.download_button(
                                label="📥 Download Result Image",
                                data=cv2.imencode('.jpg', result_image)[1].tobytes(),
                                file_name=f"detection_result_{uploaded_file.name}",
                                mime="image/jpeg"
                            )
                            
                        else:
                            st.warning("⚠️ No surgical tools detected. Try adjusting the confidence threshold.")
                            
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
                    
                    finally:
                        # Clean up temporary file
                        if 'tmp_file' in locals():
                            os.unlink(tmp_file.name)
    
    else:
        # Show instructions when no file is uploaded
        st.info("👆 Please upload an image file to get started")
        
        # Example section
        with st.expander("ℹ️ How to use this app"):
            st.write("""
            1. **Upload an image** containing surgical tools using the file uploader above
            2. **Adjust the confidence threshold** in the sidebar if needed
            3. **Click the 'Detect Surgical Tools' button** to run the detection
            4. **View the results** including:
               - Original and processed images
               - Detection counts and confidence scores
               - Bounding box coordinates
            5. **Download the result image** if desired
            
            **Supported surgical tools:**
            - Grasper
            - Hook  
            - Scissors
            - Clipper
            """)

if __name__ == '__main__':
    main()
    