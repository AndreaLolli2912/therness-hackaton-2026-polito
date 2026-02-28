import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import sys

# Add video_processing_gru/src to path for internal imports
sys.path.append(os.path.join(os.getcwd(), "video_processing_gru", "src"))
try:
    from data.explorer import DatasetExplorer
except ImportError:
    st.error("Could not import DatasetExplorer. Check your path: " + os.path.join(os.getcwd(), "video_processing_gru", "src"))

st.set_page_config(page_title="Welding Defect Detection Dashboard", layout="wide")

st.title("👨‍🏭 Welding Defect Detection Dashboard (Video-GRU Edition)")

# --- Configuration & Sidebar ---
st.sidebar.header("⚙️ Configuration")

# Allow user to specify data root
default_root = os.getcwd()
data_root = st.sidebar.text_input("Dataset Root Path", default_root)

@st.cache_data
def get_dataset(root):
    try:
        explorer = DatasetExplorer(root)
        return explorer.scan()
    except NameError:
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error scanning dataset: {e}")
        return pd.DataFrame()

if st.sidebar.button("🔄 Reload / Rescan Dataset"):
    st.cache_data.clear()

df = get_dataset(data_root)

if not df.empty:
    st.sidebar.metric("Total Runs", len(df))
    st.sidebar.metric("Total Configs", df["config"].nunique())
    
    st.sidebar.subheader("Video Coverage")
    coverage = df["has_video"].sum() / len(df)
    st.sidebar.progress(coverage, text=f"Video: {coverage:.1%}")
else:
    st.sidebar.warning("⚠️ No dataset found at the specified path.")

# --- Main Content ---
tab1, tab2, tab3 = st.tabs(["📊 Dataset Analysis", "🔍 Sample Inspector", "🛠 Data Quality"])

if df.empty:
    st.info("Please provide a valid dataset root path in the sidebar to begin analysis.")
else:
    with tab1:
        st.header("Dataset Statistics")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(df, names='label_name', title='Distribution of Weld Types')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            runs_per_config = df.groupby('config').size()
            fig = px.histogram(runs_per_config, title="Number of Runs per Configuration Folder", 
                               labels={'value': 'Runs', 'count': 'Frequency'})
            st.plotly_chart(fig, use_container_width=True)
            
        st.subheader("Summary Table")
        summary = df.groupby(['label_code', 'label_name']).agg(
            num_runs=('run_id', 'count'),
            num_configs=('config', 'nunique'),
            avg_images=('num_images', 'mean')
        ).reset_index()
        st.dataframe(summary, use_container_width=True)

    with tab2:
        st.header("Inspect a specific run")
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            selected_label = st.selectbox("Filter by Label", ["All"] + sorted(list(df["label_name"].unique())))
        with c2:
            filtered_df = df if selected_label == "All" else df[df["label_name"] == selected_label]
            selected_config = st.selectbox("Select Configuration", ["All"] + sorted(list(filtered_df["config"].unique())))
        with c3:
            final_filtered_df = filtered_df if selected_config == "All" else filtered_df[filtered_df["config"] == selected_config]
            selected_run_id = st.selectbox("Select Run ID", final_filtered_df["run_id"].unique())
        
        run_info = df[df["run_id"] == selected_run_id].iloc[0]
        # Resolve full path using current data_root
        run_full_path = Path(data_root) / run_info["rel_path"]
        
        st.info(f"**Run ID:** {selected_run_id} | **Label:** {run_info['label_name']} | **Config:** {run_info['config']}")
        
        video_file = run_full_path / f"{selected_run_id}.avi"
        
        st.subheader("🎥 Video Preview")
        if video_file.exists():
            st.video(str(video_file))
        else:
            st.warning("Video file not found at " + str(video_file))

        st.subheader("🖼 Still Images")
        if run_info["num_images"] > 0:
            images = sorted(list((run_full_path / "images").glob("*.jpg")))
            cols = st.columns(min(len(images), 5))
            for i, img_path in enumerate(images[:10]): # Limit to first 10
                with cols[i % 5]:
                    st.image(str(img_path), use_container_width=True, caption=img_path.name)
        else:
            st.info("No images found in images/ folder.")

    with tab3:
        st.header("Data Integrity Check")
        missing_video = df[~df["has_video"]]
        no_images = df[df["num_images"] == 0]
        
        col1, col2 = st.columns(2)
        col1.metric("Missing Video", len(missing_video))
        col2.metric("No Images", len(no_images))
        
        if len(missing_video) > 0:
            st.subheader("Runs with missing Video")
            st.dataframe(missing_video)
        else:
            st.success("All runs have video modality!")
