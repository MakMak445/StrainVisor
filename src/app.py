from dash import Dash, html, dcc, callback, Output, Input, State, dash_table
import datetime
import base64
import pandas as pd
import io
import dash_player as dp
import plotly.express as px
import video_tools as vt
import base64, tempfile, os
import cv2 as cv
import subprocess
import imageio.v3 as iio
import dash_uploader as du
import numpy as np
'''
def save_to_tempfile(contents, ext):
    """
    contents: either a data-URI string (dash Upload.contents), raw bytes,
              or an existing filename.
    ext:      file extension (e.g. 'mp4' or 'csv')
    returns:  path to a real file on disk
    """
    # Already on disk?
    if isinstance(contents, str) and os.path.isfile(contents):
        return contents

    # Dash-style data URI?
    if isinstance(contents, str) and contents.startswith("data:"):
        header, b64 = contents.split(",", 1)
        raw = base64.b64decode(b64)

    # Raw bytes/bytearray?
    elif isinstance(contents, (bytes, bytearray)):
        raw = contents

    else:
        raise ValueError(f"Can't save contents of type {type(contents)}")

    tmp = tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False)
    tmp.write(raw)
    tmp.flush()
    return tmp.name
'''
def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return df

def delete_folder_contents():
    folder_path = 'uploads' # The folder you want to clear
    
    if not os.path.isdir(folder_path):
        return f"Error: Folder '{folder_path}' not found."

    # Loop through everything in the folder and delete it
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            return f'Failed to delete {file_path}. Reason: {e}'
            
    return f"Successfully cleared all contents of the '{folder_path}' folder."
'''
def process_video(video_path):
    cap = cv.VideoCapture(video_path)
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    print(fps, height, width, frame_count)
    cap.release()

    return html.Div([
        html.P(f"Frames: {frame_count}"),
        html.P(f"FPS: {fps}"),
        html.P(f"Resolution: {width}x{height}")
    ])
'''
app = Dash(__name__)
du.configure_upload(app, "uploads", use_upload_id=True)
delete_folder_contents()
app.layout = html.Div([
    html.H1("Drop Weight Analysis Tool", style={'textAlign': 'center'}), 
    html.Div([
        html.H3("Upload High speed impact footage"), 
        du.Upload(id='video-uploader', text='Drag and Drop a single video file to upload',
        filetypes=['mp4', 'mov', 'avi', 'cine'],
        max_file_size=2000, # 2GB limit
        max_files=1,
        ), 
    html.Div(id='status-output'),
    dcc.Store(id='Stored-vidpath'),
    dcc.Store(id='history-vidpath')
    ]), 
        html.Hr(), 
        html.Div([
        html.H3("Upload a CSV File of your force-time graph"),
        dcc.Store(id='last-file-store'),
        dcc.Upload(
            id='csv-upload',
            children=html.Div(['Drag and Drop or ', html.A('Select a CSV')], id='csv-box-text'),
            accept='.csv',
            multiple=False,
            style={
                'width': '60%', 'height': '80px', 'lineHeight': '80px',
                'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                'textAlign': 'center', 'margin': '10px auto'
            }
        ),
        html.Div(id='csv-preview'), 
        html.Hr(), 
        html.Button(id='submit-button-state', n_clicks=0, children='Submit'), 
        dcc.Graph(id='Output-figure', figure={})
    ]), 
        html.Hr(),
        html.Video(id='video-with-lines')
])
@du.callback(
    output = Output('Stored-vidpath', 'data'), # Update the store with the new path
    id='video-uploader'
)
def handle_upload(status):
    # 1. Extract the filepath string from the status object
    new_filepath = str(status.latest_file)

    # 2. Return ONLY the simple string
    return new_filepath

@app.callback(
    Output('history-vidpath', 'data'),
    Output('status-output', 'children'), 
    Input('Stored-vidpath', 'data'),
    State('history-vidpath', 'data')
)
def clear_space(new_filepath, old_filepath):
    # If an old filepath was stored, delete that file
    if old_filepath and os.path.exists(old_filepath):
        try:
            os.remove(old_filepath)
            status_message = f"Old file '{os.path.basename(old_filepath)}' deleted. "
        except OSError as e:
            status_message = f"Error deleting old file: {e}. "
    else:
        status_message = ""

    # Process the new file (your OpenCV logic would go here)
    new_filename = os.path.basename(new_filepath)
    status_message += f"New file '{new_filename}' is ready."

    # Return the status message to the user and the new filepath to the store
    return new_filepath, status_message 

@app.callback(
    Output('csv-box-text', 'children'),
    Input('csv-upload', 'filename'),
    State('csv-box-text', 'children'),
    prevent_initial_call=True
)
def change_vid_text(filename, current_text): 
    if not filename:
        return current_text
    return f'{filename}'

@app.callback(
    Output('Output-figure', 'figure'),
    Input('submit-button-state', 'n_clicks'),
    State('csv-upload', 'filename'),
    State('Stored-vidpath', 'data'),
    State('csv-upload', 'contents'),
    State('csv-upload', 'last_modified'),
    prevent_initial_call=True
)
def generate_graph(n_clicks, filename, vidpath, filecontents, date):
    if n_clicks == 0:
        return
    else:
        df = parse_contents(filecontents, filename, date)
        time = df['Time'].iloc[1:]
        volt = df['Channel A'].iloc[1:]
        print(time)
        print(volt)
        frames, strain = vt.generate_strain_graph(time, volt, vidpath)
        fig = px.scatter(
        x=frames,
        y=strain,
        labels={'x': 'Frame Number', 'y': 'Strain'},
        title='Strain vs. Frame Number'
    )
        return fig

if __name__ == '__main__':
    app.run(debug=True)
