import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import requests
from io import BytesIO

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
        # Try to load the model - you'll need to upload this file to Streamlit Cloud
        model = YOLO('best.pt')
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.info("💡 Please make sure 'best.pt' is in your GitHub repository")
        return None

def process_detection(model, image, confidence):
    """Process image and return detections"""
    try:
        # Convert PIL to OpenCV format
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Run detection
        results = model(img_array, conf=confidence)
        
        # Process results
        result_image = results[0].plot()
        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        
        detections = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                detections.append({
                    'class': model.names[int(box.cls)],
                    'confidence': float(box.conf),
                    'bbox': box.xyxy[0].tolist()
                })
        
        return result_image_rgb, detections, None
        
    except Exception as e:
        return None, [], f"Detection error: {str(e)}"

def main():
    st.title("🔪 Surgical Tools Detection")
    st.markdown("""
    **Detect surgical tools in images using AI-powered computer vision**
    
    This application uses YOLOv8 to identify:
    - **Graspers** 🗜️
    - **Hooks** 🪝  
    - **Scissors** ✂️
    - **Clippers** 🔗
    """)
    
    # Initialize model
    model = load_model()
    
    if model is None:
        st.warning("""
        ⚠️ **Model not loaded** - This is expected in the demo version.
        
        To use the full version:
        1. Clone this app from GitHub
        2. Add your trained `best.pt` model file
        3. Deploy on Streamlit Cloud
        
        **Try the demo with sample images below!**
        """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        confidence = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            help="Adjust detection sensitivity"
        )
        
        st.header("ℹ️ About")
        st.markdown("""
        This AI model was trained on surgical tool datasets 
        using YOLOv8 for real-time object detection.
        
        **Features:**
        - Real-time detection
        - Multiple tool recognition
        - Confidence scoring
        - Bounding box visualization
        """)
        
        st.header("📁 Sample Images")
        st.markdown("Try these sample surgical tool images:")
        
        # Sample image URLs
        sample_images = {
            "Surgical Tools 1": "https://images.unsplash.com/photo-1551601651-2a8555f1a136?w=400",
            "Surgical Tools 2": "https://images.unsplash.com/photo-1582750433449-648ed127bb54?w=400",
            "Medical Instruments": "https://images.unsplash.com/photo-1559757148-5c350d0d3c56?w=400"
        }
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📤 Upload Image", "🌐 Sample Images", "📚 Instructions"])
    
    with tab1:
        st.subheader("Upload Your Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image containing surgical tools"
        )
        
        if uploaded_file is not None:
            process_uploaded_file(uploaded_file, model, confidence)
    
    with tab2:
        st.subheader("Try with Sample Images")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🖼️ Load Sample 1", use_container_width=True):
                load_sample_image(sample_images["Surgical Tools 1"], model, confidence)
        
        with col2:
            if st.button("🖼️ Load Sample 2", use_container_width=True):
                load_sample_image(sample_images["Surgical Tools 2"], model, confidence)
        
        with col3:
            if st.button("🖼️ Load Sample 3", use_container_width=True):
                load_sample_image(sample_images["Medical Instruments"], model, confidence)
    
    with tab3:
        st.subheader("📖 User Guide")
        
        st.markdown("""
        ### How to Use This App:
        
        1. **Upload an Image** (Tab 1):
           - Click "Browse files" or drag & drop an image
           - Supported formats: JPG, JPEG, PNG, BMP
           - Click "Detect Tools" to analyze
        
        2. **Try Sample Images** (Tab 2):
           - Test the app with pre-loaded sample images
           - See how detection works with different tools
        
        3. **Adjust Settings** (Sidebar):
           - Confidence Threshold: Controls detection sensitivity
           - Higher = fewer detections but more accurate
           - Lower = more detections but may include false positives
        
        ### Expected Output:
        - Bounding boxes around detected tools
        - Tool classification (Grasper, Hook, etc.)
        - Confidence scores for each detection
        - Option to download results
        
        ### Tips for Best Results:
        - Use clear, well-lit images
        - Ensure tools are visible and not obscured
        - Adjust confidence threshold as needed
        - For multiple tools, ensure they're separated in the image
        """)

def process_uploaded_file(uploaded_file, model, confidence):
    """Process user-uploaded file"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Original Image")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
        st.write(f"**Image Details:** {image.size[0]} × {image.size[1]} pixels")
    
    with col2:
        st.subheader("🔍 Detection Results")
        
        if model is None:
            st.error("""
            🚧 **Demo Mode** 
            
            The full detection model is not available in this demo. 
            To enable detection:
            
            1. **Download the complete app** from GitHub
            2. **Add your trained model** (best.pt)
            3. **Deploy on Streamlit Cloud**
            
            [Get the complete code](https://github.com/)
            """)
            return
        
        if st.button("🚀 Detect Surgical Tools", type="primary", use_container_width=True):
            with st.spinner("🔄 Analyzing image for surgical tools..."):
                result_image, detections, error = process_detection(model, image, confidence)
            
            if error:
                st.error(f"❌ {error}")
            else:
                # Display results
                st.image(result_image, use_column_width=True)
                
                if detections:
                    show_detection_results(detections, result_image, uploaded_file.name)
                else:
                    st.warning("""
                    ⚠️ **No tools detected**
                    
                    Try:
                    - Lowering the confidence threshold
                    - Using a clearer image
                    - Ensuring tools are visible
                    """)

def load_sample_image(url, model, confidence):
    """Load and process sample image"""
    try:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        
        st.subheader("Sample Image Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Sample Image", use_column_width=True)
        
        with col2:
            if model is None:
                st.error("Model not available in demo")
                return
            
            with st.spinner("Processing sample image..."):
                result_image, detections, error = process_detection(model, image, confidence)
            
            if error:
                st.error(f"Error: {error}")
            else:
                st.image(result_image, caption="Detection Results", use_column_width=True)
                
                if detections:
                    st.success(f"✅ Found {len(detections)} tool(s)")
                    for det in detections:
                        st.write(f"- **{det['class']}** ({det['confidence']:.1%} confidence)")
                else:
                    st.info("No surgical tools detected in this sample image")
    
    except Exception as e:
        st.error(f"Failed to load sample image: {str(e)}")

def show_detection_results(detections, result_image, filename):
    """Display detection results and download options"""
    st.success(f"✅ **Detection Complete!** Found {len(detections)} surgical tool(s)")
    
    # Detection details
    with st.expander("📊 View Detailed Results", expanded=True):
        # Summary statistics
        tool_counts = {}
        for det in detections:
            tool_name = det['class']
            tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
        
        st.write("**Detection Summary:**")
        for tool, count in tool_counts.items():
            st.write(f"- {tool}: {count} detected")
        
        st.write("---")
        st.write("**Detailed Results:**")
        
        for i, det in enumerate(detections, 1):
            st.write(f"**{i}. {det['class']}**")
            st.write(f"   - Confidence: `{det['confidence']:.1%}`")
            st.write(f"   - Bounding Box: `{[round(coord, 1) for coord in det['bbox']]}`")
    
    # Download section
    st.download_button(
        label="📥 Download Result Image",
        data=cv2.imencode('.jpg', cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))[1].tobytes(),
        file_name=f"detected_{filename}",
        mime="image/jpeg",
        use_container_width=True
    )

if __name__ == '__main__':
    main()