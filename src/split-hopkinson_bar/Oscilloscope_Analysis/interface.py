import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Set up the webpage
st.set_page_config(page_title="SHPB Pulse Overlay", layout="wide")
st.title("Interactive Pulse Overlay Dashboard")

# 1. Drag-and-drop file uploader
uploaded_files = st.file_uploader(
    "Upload your processed _threshold.csv files", 
    type=["csv"], 
    accept_multiple_files=True
)

if uploaded_files:
    # Initialize the interactive Plotly figure
    fig = go.Figure()

    for file in uploaded_files:
        # Read each uploaded CSV
        df = pd.read_csv(file)
        
        # Identify the shared time column (based on your previous script's output)
        time_cols = [col for col in df.columns if "Time" in col or "Shared" in col]
        if not time_cols:
            st.warning(f"Could not find a Time column in {file.name}")
            continue
            
        time_col = time_cols[0]
        
        # Identify all strain columns
        strain_cols = [col for col in df.columns if "Strain" in col]
        
        # Plot each pulse onto the shared figure
        for col in strain_cols:
            # We drop NaNs so shorter pulses don't break the line
            clean_df = df[[time_col, col]].dropna() 
            
            fig.add_trace(go.Scatter(
                x=clean_df[time_col], 
                y=clean_df[col], 
                mode='lines',
                name=f"{file.name.replace('_threshold.csv', '')} - {col}"
            ))

    # Formatting the plot
    fig.update_layout(
        xaxis_title="Time (ms)",
        yaxis_title="Strain (με)",
        hovermode="x unified", 
        # Pushes the legend outside the graph to the right
        legend=dict(yanchor="top", y=1, xanchor="left", x=1.02)
    )

    # Render the plot
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Upload one or more CSV files to begin.")