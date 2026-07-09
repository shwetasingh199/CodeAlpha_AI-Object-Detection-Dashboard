import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px

from PIL import Image

from processor import VideoProcessor

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI Vision Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main{
    background-color:#0E1117;
}

.block-container{
    padding-top:2rem;
}

h1{
    color:#00E5FF;
}

.metric-card{
    background:#1E1E1E;
    padding:15px;
    border-radius:15px;
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("⚙️ Detection Settings")

input_type = st.sidebar.radio(
    "Choose Input",
    ["Image", "Video"]
)

confidence = st.sidebar.slider(
    "Confidence Threshold",
    0.10,
    1.00,
    0.25,
    0.05
)

iou = st.sidebar.slider(
    "IoU Threshold",
    0.10,
    1.00,
    0.45,
    0.05
)

st.sidebar.markdown("---")

st.sidebar.info(
"""
Supported Classes

• Person

• Car

• Dog

• Cat

• Bicycle

• Bus

• Truck

• etc...
"""
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

processor = VideoProcessor(confidence=confidence)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<h1 style='text-align:center'>
🎯 AI Object Detection Dashboard
</h1>
""", unsafe_allow_html=True)

st.markdown(
"""
<center>

Real-Time Object Detection using
YOLOv8 + OpenCV + Streamlit

</center>
""",
unsafe_allow_html=True
)

st.divider()

# ==================================================
# IMAGE SECTION
# ==================================================

if input_type == "Image":

    uploaded = st.file_uploader(
        "📤 Upload an Image",
        type=["jpg","jpeg","png"]
    )

    if uploaded:

        image = Image.open(uploaded).convert("RGB")

        image_np = np.array(image)

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("📷 Original Image")

            st.image(
                image,
                use_container_width=True
            )

        with st.spinner("Detecting Objects..."):

            result, objects, class_count = processor.process_image(
                image_np
            )

        result = cv2.cvtColor(
            result,
            cv2.COLOR_BGR2RGB
        )

        with col2:

            st.subheader("🎯 Detection Result")

            st.image(
                result,
                use_container_width=True
            )

        st.divider()

        # ------------------------------------------

        total_objects = len(objects)

        unique_objects = len(class_count)

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "📦 Total Objects",
            total_objects
        )

        c2.metric(
            "🏷️ Unique Classes",
            unique_objects
        )

        c3.metric(
            "🎯 Confidence",
            f"{confidence:.2f}"
        )

        st.divider()

        # ------------------------------------------

        st.subheader("📋 Detected Objects")

        df = pd.DataFrame(
            class_count.items(),
            columns=[
                "Object",
                "Count"
            ]
        )

        st.dataframe(
            df,
            use_container_width=True
        )

        st.divider()

        st.subheader("📊 Object Distribution")

        fig = px.bar(
            df,
            x="Object",
            y="Count",
            color="Object",
            text="Count"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.divider()

        st.subheader("✅ Objects Present")

        cols = st.columns(4)

        i = 0

        for obj in sorted(set(objects)):

            cols[i % 4].success(obj.title())

            i += 1

# ==================================================
# VIDEO SECTION
# ==================================================

elif input_type == "Video":

    uploaded = st.file_uploader(
        "🎥 Upload a Video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded is not None:

        st.subheader("📹 Uploaded Video")

        st.video(uploaded)

        st.divider()

        if st.button("🚀 Start Detection", use_container_width=True):

            progress = st.progress(0)

            status = st.empty()

            status.info("Initializing YOLOv8 Model...")

            progress.progress(10)

            with st.spinner("Processing video..."):

                progress.progress(30)

                output_path, class_count = processor.process_video(uploaded)

                progress.progress(100)

            status.success("✅ Detection Completed Successfully!")

            st.balloons()

            st.divider()

            st.subheader("🎬 Processed Video")

            st.video(output_path)

            total_objects = sum(class_count.values())

            unique_classes = len(class_count)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "📦 Total Detections",
                    total_objects
                )

            with col2:
                st.metric(
                    "🏷️ Object Classes",
                    unique_classes
                )

            with col3:
                st.metric(
                    "🎯 Confidence",
                    f"{confidence:.2f}"
                )

            st.divider()

            st.subheader("📋 Detection Summary")

            df = pd.DataFrame(
                class_count.items(),
                columns=["Object", "Count"]
            )

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

            st.divider()

            st.subheader("📊 Object Distribution")

            fig = px.bar(
                df,
                x="Object",
                y="Count",
                color="Object",
                text="Count",
                title="Detected Objects"
            )

            fig.update_layout(
                template="plotly_dark",
                height=450
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.divider()

            st.subheader("📦 Objects Found")

            cols = st.columns(4)

            for i, (name, count) in enumerate(class_count.items()):

                cols[i % 4].success(
                    f"{name.title()} ({count})"
                )

            st.divider()

            with open(output_path, "rb") as file:

                st.download_button(
                    label="⬇ Download Processed Video",
                    data=file,
                    file_name="Detected_Output.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )     

# ==================================================
# FOOTER
# ==================================================

st.divider()

st.markdown(
"""
<div style='text-align:center;color:gray'>

### 🎯 AI Vision Dashboard

Built with ❤️ using

**YOLOv8 • OpenCV • Streamlit**

</div>
""",
unsafe_allow_html=True
)
                       