import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set up the webpage
st.set_page_config(page_title="SHPB Analysis Dashboard", layout="wide")
st.title("SHPB Interactive Analysis Dashboard")

# Create tabs for the two different features
tab1, tab2 = st.tabs(["Multi-Pulse Overlay", "Video & Oscilloscope Alignment"])

# ==========================================
# TAB 1: ORIGINAL MULTI-FILE OVERLAY
# ==========================================
with tab1:
    st.header("Interactive Pulse Overlay")
    
    uploaded_files = st.file_uploader(
        "Upload your processed _threshold.csv files", 
        type=["csv"], 
        accept_multiple_files=True,
        key="multi_upload"
    )

    if uploaded_files:
        fig1 = go.Figure()

        for file in uploaded_files:
            df = pd.read_csv(file)
            
            time_cols = [col for col in df.columns if "Time" in col or "Shared" in col]
            if not time_cols:
                st.warning(f"Could not find a Time column in {file.name}")
                continue
                
            time_col = time_cols[0]
            strain_cols = [col for col in df.columns if "Strain" in col]
            
            for col in strain_cols:
                clean_df = df[[time_col, col]].dropna() 
                
                fig1.add_trace(go.Scatter(
                    x=clean_df[time_col], 
                    y=clean_df[col], 
                    mode='lines',
                    name=f"{file.name.replace('_threshold.csv', '')} - {col}"
                ))

        fig1.update_layout(
            xaxis_title="Time",
            yaxis_title="Strain (με)",
            hovermode="x unified", 
            legend=dict(yanchor="top", y=1, xanchor="left", x=1.02)
        )
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("Upload one or more CSV files to begin overlaying pulses.")

# ==========================================
# TAB 2: VIDEO AND OSCILLOSCOPE ALIGNMENT
# ==========================================
with tab2:
    st.header("Oscilloscope & Video Analysis Alignment")
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Oscilloscope Data")
        file_osc = st.file_uploader("Upload Oscilloscope / Threshold CSV", type=["csv"], key="osc")
        
        if file_osc:
            st.write("Define Wave Arrival Times (from Oscilloscope):")
            t_input_start = st.number_input("Input Pulse Start Time", value=0.0, format="%.5f")
            t_reflect_start = st.number_input("Reflected Pulse Start Time", value=0.0, format="%.5f")

    with col2:
        st.subheader("2. Video Analysis Data")
        file_vid = st.file_uploader("Upload Video Analysis CSV", type=["csv"], key="vid")
        
        if file_vid:
            st.write("Video Strain Calibration (Optional):")
            baseline_pixels = st.number_input("Baseline (Zero-Strain) Distance", value=0.0)
            pixel_multiplier = st.number_input("Pixel-to-Strain Multiplier", value=1.0, format="%.6f")

    if file_osc and file_vid:
        df_osc = pd.read_csv(file_osc)
        df_vid = pd.read_csv(file_vid)
        
        # Identify columns
        time_col_osc = [col for col in df_osc.columns if "Time" in col or "Shared" in col][0]
        time_col_vid = [col for col in df_vid.columns if "Time" in col][0]
        
        strain_cols = [col for col in df_osc.columns if "Strain" in col]
        strain_col = strain_cols[0] if strain_cols else df_osc.columns[1]
        
        dist_cols = [col for col in df_vid.columns if "Distance" in col]
        dist_col = dist_cols[0] if dist_cols else df_vid.columns[1]

        # Calculate impact time offset
        impact_time = (t_input_start + t_reflect_start) / 2
        
        # Align video time
        video_start_offset = df_vid[time_col_vid].iloc[0]
        df_vid['Aligned_Time'] = (df_vid[time_col_vid] - video_start_offset) + impact_time
        
        # Apply Strain Calculation
        df_vid['Calculated_Video_Strain'] = (df_vid[dist_col] - baseline_pixels) * pixel_multiplier

        # Create dual-axis plot
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])

        # Trace 1: Oscilloscope Strain
        fig2.add_trace(
            go.Scatter(x=df_osc[time_col_osc], y=df_osc[strain_col], name="Oscilloscope Strain", mode='lines'),
            secondary_y=False,
        )

        # Trace 2: Video Data
        # Toggle 'y' depending on if you want raw pixels or calculated strain
        fig2.add_trace(
            go.Scatter(x=df_vid['Aligned_Time'], y=df_vid['Calculated_Video_Strain'], name="Video Tracking", mode='lines+markers'),
            secondary_y=True,
        )

        fig2.update_layout(
            title="Aligned Stress Wave and Video Tracking",
            xaxis_title=f"Time ({time_col_osc})",
            hovermode="x unified",
            legend=dict(yanchor="top", y=1.1, xanchor="left", x=0.01, orientation="h")
        )
        
        fig2.update_yaxes(title_text="Oscilloscope Strain", secondary_y=False)
        
        # Update secondary y-axis label based on multiplier usage
        y2_label = "Calculated Strain" if pixel_multiplier != 1.0 else "Raw Distance (Pixels)"
        fig2.update_yaxes(title_text=y2_label, secondary_y=True)

        st.plotly_chart(fig2, use_container_width=True)
        st.info(f"Calculated sample impact time: **{impact_time:.5f}**. Video data aligned to this timestamp.")
    else:
        if file_osc or file_vid:
            st.warning("Upload both CSV files to view the alignment.")