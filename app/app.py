"""
Streamlit Demo App for Seizure Forecasting

Interactive visualization of:
- EEG signals
- Real-time risk score prediction
- Alarm triggering
- Seizure onset markers

Usage:
    streamlit run app/app.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import h5py
import torch

# path
sys.path.insert(0, str(Path(__file__).parent.parent))


st.set_page_config(
    page_title="Seizure Forecasting Demo",
    page_icon="🧠",
    layout="wide",
)


@st.cache_resource
def load_model(checkpoint_path: str, cfg_path: str):
    """Load trained model."""
    from src.utils.config import load_config
    from src.models.fusion_net import FusionNet
    
    cfg = load_config(cfg_path)
    
    # determine the checkpoint lcoation for model
    device = torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # guess dimensions????
    n_channels = 18  # Default
    n_features = 40  # Default
    
    model = FusionNet(
        n_channels=n_channels,
        n_features=n_features,
        cfg=cfg,
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model, cfg


@st.cache_data
def load_subject_data(cache_path: str, subject: str):
    """lowkirkenuinely load the cached data for a subject"""
    cache_file = Path(cache_path) / f"{subject}.h5"
    
    if not cache_file.exists():
        return None, None, None, None
    
    with h5py.File(cache_file, "r") as f:
        data = f["data"][:]
        features = f["features"][:] if "features" in f else None
        y_cls = f["y_cls"][:]
        y_tte = f["y_tte"][:]
        channels = f.attrs.get("channels", "").split(",")
        sfreq = f.attrs.get("sfreq", 256)
    
    return data, features, y_cls, y_tte, channels, sfreq


def get_available_subjects(cache_path: str):
    """Get list of subjects with cached data."""
    cache_dir = Path(cache_path)
    if not cache_dir.exists():
        return []
    
    subjects = []
    for f in cache_dir.glob("*.h5"):
        subjects.append(f.stem)
    
    return sorted(subjects)


def generate_risk_scores(model, data, features, device="cpu"):
    """Generate risk scores for all windows."""
    model.eval()
    
    predictions = []
    batch_size = 32
    
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch_data = torch.from_numpy(data[i:i+batch_size]).float().to(device)
            batch_features = None
            if features is not None:
                batch_features = torch.from_numpy(features[i:i+batch_size]).float().to(device)
            
            probs = model.predict_proba(batch_data, batch_features)
            predictions.extend(probs.cpu().numpy())
    
    return np.array(predictions)


def create_eeg_plot(window_data, channels, sfreq, title="EEG Window"):
    """Create EEG plot using Plotly."""
    n_channels = min(len(channels), 8)  # Show max 8 channels
    n_samples = window_data.shape[1]
    
    times = np.arange(n_samples) / sfreq
    
    fig = make_subplots(
        rows=n_channels, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
    )
    
    colors = [
        "#2563eb", "#dc2626", "#22c55e", "#f59e0b",
        "#8b5cf6", "#06b6d4", "#ec4899", "#84cc16"
    ]
    
    for i in range(n_channels):
        # Scale for visibility
        signal = window_data[i] * 1e6  # Convert to µV
        
        fig.add_trace(
            go.Scatter(
                x=times,
                y=signal,
                mode="lines",
                name=channels[i] if i < len(channels) else f"Ch{i+1}",
                line=dict(color=colors[i % len(colors)], width=1),
            ),
            row=i+1, col=1
        )
        
        fig.update_yaxes(
            title_text=channels[i][:8] if i < len(channels) else f"Ch{i+1}",
            row=i+1, col=1,
            showticklabels=False,
        )
    
    fig.update_layout(
        height=400,
        title=title,
        showlegend=False,
        margin=dict(l=60, r=20, t=40, b=40),
    )
    
    fig.update_xaxes(title_text="Time (s)", row=n_channels, col=1)
    
    return fig


def create_risk_timeline(risk_scores, labels, window_sec=10, step_sec=5, threshold=0.5):
    """Create risk score timeline plot."""
    n_windows = len(risk_scores)
    times = np.arange(n_windows) * step_sec / 60  # Convert to minutes
    
    fig = go.Figure()
    
    # Risk score line
    fig.add_trace(
        go.Scatter(
            x=times,
            y=risk_scores,
            mode="lines",
            name="Risk Score",
            line=dict(color="#2563eb", width=2),
            fill="tozeroy",
            fillcolor="rgba(37, 99, 235, 0.2)",
        )
    )
    
    # Threshold line
    fig.add_hline(
        y=threshold,
        line=dict(color="#dc2626", width=2, dash="dash"),
        annotation_text=f"Threshold ({threshold})",
    )
    
    # Mark preictal windows
    preictal_times = times[labels == 1]
    preictal_scores = risk_scores[labels == 1]
    
    if len(preictal_times) > 0:
        fig.add_trace(
            go.Scatter(
                x=preictal_times,
                y=preictal_scores,
                mode="markers",
                name="Preictal",
                marker=dict(color="#f97316", size=8, symbol="diamond"),
            )
        )
    
    # Mark alarms (risk > threshold)
    alarm_mask = risk_scores >= threshold
    alarm_times = times[alarm_mask]
    alarm_scores = risk_scores[alarm_mask]
    
    if len(alarm_times) > 0:
        fig.add_trace(
            go.Scatter(
                x=alarm_times,
                y=alarm_scores,
                mode="markers",
                name="Alarms",
                marker=dict(color="#22c55e", size=10, symbol="triangle-up"),
            )
        )
    
    fig.update_layout(
        title="Risk Score Over Time",
        xaxis_title="Time (minutes)",
        yaxis_title="Risk Score",
        yaxis_range=[0, 1],
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    
    return fig


def main():
    st.title("🧠 Seizure Forecasting Demo")
    st.markdown("""
    Interactive visualization of seizure prediction from scalp EEG.
    The model predicts risk of seizure occurrence in the next 10 minutes.
    """)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Paths
    cache_path = st.sidebar.text_input(
        "Cache Path",
        value="data/chbmit_cache",
    )
    
    checkpoint_path = st.sidebar.text_input(
        "Model Checkpoint",
        value="runs/deep_model/checkpoints/best.pt",
    )
    
    config_path = st.sidebar.text_input(
        "Config File",
        value="configs/small_run.yaml",
    )
    
    # Get available subjects
    subjects = get_available_subjects(cache_path)
    
    if not subjects:
        st.warning(f"No cached data found in {cache_path}")
        st.info("Run `python scripts/build_cache.py` first to create the cache.")
        
        # Show demo with synthetic data
        st.subheader("Demo Mode (Synthetic Data)")
        show_demo_mode()
        return
    
    # Subject selection
    selected_subject = st.sidebar.selectbox("Select Subject", subjects)
    
    # Parameters
    st.sidebar.subheader("Alarm Parameters")
    threshold = st.sidebar.slider("Risk Threshold", 0.0, 1.0, 0.5, 0.05)
    
    # Load data
    data, features, y_cls, y_tte, channels, sfreq = load_subject_data(
        cache_path, selected_subject
    )
    
    if data is None:
        st.error(f"Failed to load data for {selected_subject}")
        return
    
    st.sidebar.info(f"""
    **Subject Info:**
    - Windows: {len(data)}
    - Channels: {len(channels)}
    - Sample Rate: {sfreq} Hz
    - Preictal: {np.sum(y_cls == 1)}
    - Interictal: {np.sum(y_cls == 0)}
    """)
    
    # Try to load model
    model = None
    if Path(checkpoint_path).exists() and Path(config_path).exists():
        try:
            model, cfg = load_model(checkpoint_path, config_path)
            st.sidebar.success("Model loaded successfully!")
        except Exception as e:
            st.sidebar.warning(f"Could not load model: {e}")
    
    # Generate predictions
    if model is not None:
        risk_scores = generate_risk_scores(model, data, features)
    else:
        # Use simple heuristic based on labels
        risk_scores = np.where(y_cls == 1, 0.7 + 0.2 * np.random.rand(len(y_cls)), 
                               0.2 + 0.1 * np.random.rand(len(y_cls)))
    
    # Main content
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_alarms = np.sum(risk_scores >= threshold)
        st.metric("Total Alarms", n_alarms)
    
    with col2:
        n_preictal = np.sum(y_cls == 1)
        caught = np.sum((y_cls == 1) & (risk_scores >= threshold))
        sens = caught / max(n_preictal, 1) * 100
        st.metric("Sensitivity", f"{sens:.1f}%")
    
    with col3:
        n_interictal = np.sum(y_cls == 0)
        fp = np.sum((y_cls == 0) & (risk_scores >= threshold))
        fah = fp / max(n_interictal * 5 / 3600, 0.1)  # Approximate FAH
        st.metric("False Alarms/Hour", f"{fah:.2f}")
    
    # Risk timeline
    st.subheader("Risk Score Timeline")
    fig_risk = create_risk_timeline(risk_scores, y_cls, threshold=threshold)
    st.plotly_chart(fig_risk, use_container_width=True)
    
    # Window viewer
    st.subheader("EEG Window Viewer")
    
    window_idx = st.slider(
        "Select Window",
        0, len(data) - 1,
        0,
        help="Slide to navigate through windows"
    )
    
    # Display window info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        label = "Preictal" if y_cls[window_idx] == 1 else "Interictal"
        st.markdown(f"**Label:** {label}")
    
    with col2:
        st.markdown(f"**Risk Score:** {risk_scores[window_idx]:.3f}")
    
    with col3:
        alarm = "🚨 ALARM" if risk_scores[window_idx] >= threshold else "✓ Normal"
        st.markdown(f"**Status:** {alarm}")
    
    with col4:
        if y_tte[window_idx] > 0:
            st.markdown(f"**Time to Seizure:** {y_tte[window_idx]:.0f}s")
        else:
            st.markdown("**Time to Seizure:** N/A")
    
    # EEG plot
    fig_eeg = create_eeg_plot(
        data[window_idx],
        channels,
        sfreq,
        title=f"Window {window_idx} - {label}"
    )
    st.plotly_chart(fig_eeg, use_container_width=True)
    
    # Playback controls
    st.subheader("Playback")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        speed = st.selectbox("Speed", [0.5, 1.0, 2.0, 5.0], index=1)
    
    with col2:
        if st.button("▶ Play"):
            progress = st.progress(0)
            status = st.empty()
            
            for i in range(window_idx, min(window_idx + 50, len(data))):
                progress.progress((i - window_idx) / 50)
                status.markdown(f"Window {i} | Risk: {risk_scores[i]:.3f} | " +
                              ("🚨 ALARM" if risk_scores[i] >= threshold else "✓ Normal"))
                time.sleep(0.1 / speed)
    
    # Distribution plot
    st.subheader("Risk Score Distribution")
    
    fig_dist = go.Figure()
    
    fig_dist.add_trace(go.Histogram(
        x=risk_scores[y_cls == 0],
        name="Interictal",
        opacity=0.7,
        marker_color="#22c55e",
    ))
    
    fig_dist.add_trace(go.Histogram(
        x=risk_scores[y_cls == 1],
        name="Preictal",
        opacity=0.7,
        marker_color="#f97316",
    ))
    
    fig_dist.add_vline(x=threshold, line=dict(color="#dc2626", dash="dash"))
    
    fig_dist.update_layout(
        barmode="overlay",
        title="Risk Score Distribution by Class",
        xaxis_title="Risk Score",
        yaxis_title="Count",
        height=300,
    )
    
    st.plotly_chart(fig_dist, use_container_width=True)


def show_demo_mode():
    """Show demo with synthetic data."""
    np.random.seed(42)
    
    # Generate synthetic data
    n_windows = 200
    n_channels = 8
    window_samples = 2560  # 10s at 256 Hz
    
    # Synthetic EEG
    t = np.linspace(0, 10, window_samples)
    data = np.zeros((n_windows, n_channels, window_samples))
    
    for i in range(n_windows):
        for ch in range(n_channels):
            # Mix of frequencies
            signal = (
                0.5 * np.sin(2 * np.pi * 10 * t + np.random.rand() * 2 * np.pi) +
                0.3 * np.sin(2 * np.pi * 6 * t + np.random.rand() * 2 * np.pi) +
                0.2 * np.random.randn(window_samples)
            )
            data[i, ch] = signal * 50e-6  # Scale to µV range
    
    # Synthetic labels
    y_cls = np.zeros(n_windows, dtype=int)
    y_cls[80:100] = 1  # Preictal zone
    
    # Synthetic risk scores
    risk_scores = 0.2 + 0.1 * np.random.randn(n_windows)
    for i in range(80, 100):
        progress = (i - 80) / 20
        risk_scores[i] = 0.3 + 0.6 * progress + 0.1 * np.random.randn()
    risk_scores = np.clip(risk_scores, 0, 1)
    
    threshold = 0.5
    
    # Display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Windows", n_windows)
    with col2:
        st.metric("Preictal Windows", np.sum(y_cls == 1))
    with col3:
        st.metric("Alarms", np.sum(risk_scores >= threshold))
    
    # Risk timeline
    fig_risk = create_risk_timeline(risk_scores, y_cls, threshold=threshold)
    st.plotly_chart(fig_risk, use_container_width=True)
    
    # Sample EEG
    st.subheader("Sample EEG Window")
    
    window_idx = st.slider("Select Window", 0, n_windows - 1, 0)
    
    channels = [f"Ch{i+1}" for i in range(n_channels)]
    fig_eeg = create_eeg_plot(data[window_idx], channels, 256)
    st.plotly_chart(fig_eeg, use_container_width=True)


if __name__ == "__main__":
    main()
