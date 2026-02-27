import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
import cv2
import librosa
import librosa.display
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import sys

# Add video_processing/src to path for internal imports
sys.path.append(os.path.join(os.getcwd(), "video_processing", "src"))
from data.explorer import DatasetExplorer

st.set_page_config(page_title="Welding Defect Detection Dashboard", layout="wide")

st.title("👨‍🏭 Welding Defect Detection Dashboard")

# --- Configuration & Sidebar ---
st.sidebar.header("⚙️ Configuration")

# Allow user to specify data root - Default to current or Desktop
default_root = "/Users/timon/Desktop/Hackathon"
if not os.path.exists(default_root):
    default_root = os.getcwd()

data_root = st.sidebar.text_input("Dataset Root Path", default_root)
manifest_path = "dataset_manifest.csv"

@st.cache_data
def get_dataset(root):
    explorer = DatasetExplorer(root)
    try:
        return explorer.scan()
    except Exception as e:
        st.error(f"Error scanning dataset: {e}")
        return pd.DataFrame()

if st.sidebar.button("🔄 Reload / Rescan Dataset"):
    st.cache_data.clear()

df = get_dataset(data_root)

if not df.empty:
    st.sidebar.metric("Total Runs", len(df))
    st.sidebar.metric("Total Configs", df["config"].nunique())
    
    st.sidebar.subheader("Modality Coverage")
    coverage = {
        "CSV": df["has_csv"].sum() / len(df),
        "Audio": df["has_audio"].sum() / len(df),
        "Video": df["has_video"].sum() / len(df),
        "Images": (df["num_images"] > 0).sum() / len(df)
    }
    for mod, val in coverage.items():
        st.sidebar.progress(val, text=f"{mod}: {val:.1%}")
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
        
        # Files
        csv_file = run_full_path / f"{selected_run_id}.csv"
        audio_file = run_full_path / f"{selected_run_id}.flac"
        video_file = run_full_path / f"{selected_run_id}.avi"
        
        col_s, col_a = st.columns(2)
        
        with col_s:
            st.subheader("📈 Sensor Data")
            if csv_file.exists():
                sensor_df = pd.read_csv(csv_file)
                numeric_cols = ['Pressure', 'CO2 Weld Flow', 'Feed', 'Primary Weld Current', 'Wire Consumed', 'Secondary Weld Voltage']
                available_cols = [c for c in numeric_cols if c in sensor_df.columns]
                
                fig = go.Figure()
                for col in available_cols:
                    fig.add_trace(go.Scatter(y=sensor_df[col], name=col))
                fig.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("CSV file not found.")

        with col_a:
            st.subheader("🔊 Audio Data")
            if audio_file.exists():
                st.audio(str(audio_file))
                try:
                    y, sr = librosa.load(audio_file, sr=None)
                    fig, ax = plt.subplots(figsize=(10, 4))
                    librosa.display.waveshow(y, sr=sr, ax=ax)
                    ax.set_title("Waveform")
                    st.pyplot(fig)
                    
                    fig_spec, ax_spec = plt.subplots(figsize=(10, 4))
                    S = librosa.feature.melspectrogram(y=y, sr=sr)
                    S_dB = librosa.power_to_db(S, ref=np.max)
                    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax_spec)
                    ax_spec.set_title("Mel-frequency spectrogram")
                    st.pyplot(fig_spec)
                except Exception as e:
                    st.error(f"Error loading audio: {e}")
            else:
                st.warning("Audio file not found.")
        
        st.subheader("🖼 Still Images")
        if run_info["num_images"] > 0:
            images = sorted(list((run_full_path / "images").glob("*.jpg")))
            cols = st.columns(min(len(images), 5))
            for i, img_path in enumerate(images):
                with cols[i % 5]:
                    st.image(str(img_path), use_container_width=True, caption=img_path.name)
        else:
            st.info("No images found in images/ folder.")
        
        st.subheader("🎥 Video Preview")
        if video_file.exists():
            st.video(str(video_file))
        else:
            st.warning("Video file not found.")

    with tab3:
        st.header("Data Integrity Check")
        missing_csv = df[~df["has_csv"]]
        missing_audio = df[~df["has_audio"]]
        missing_video = df[~df["has_video"]]
        no_images = df[df["num_images"] == 0]
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Missing CSV", len(missing_csv))
        col2.metric("Missing Audio", len(missing_audio))
        col3.metric("Missing Video", len(missing_video))
        col4.metric("No Images", len(no_images))
        
        if len(missing_csv) + len(missing_audio) + len(missing_video) > 0:
            st.subheader("Corrupt / Incomplete Runs")
            st.dataframe(df[~df["has_csv"] | ~df["has_audio"] | ~df["has_video"]])
        else:
            st.success("All runs have complete CSV, Audio, and Video modalities!")
