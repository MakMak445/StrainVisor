from dash import Dash, html, dcc, callback, Output, Input, State
import plotly.express as px
import base64, os
import dash_uploader as du
import shutil
import requests

app = Dash(__name__)
VIDEO_API_URL = "http://localhost:8080"
CSV_API_URL = "http://localhost:8000"

video_endpoint = f"{VIDEO_API_URL}/process-video/"
csv_endpoint = f"{CSV_API_URL}/process-csv/"

du.configure_upload(app, "uploads", use_upload_id=True)
app.layout = html.Div([
    html.H1("Drop Weight Analysis Tool", style={'textAlign': 'center'}), 
    html.Div([ 
        html.H3("Upload High speed impact footage"), 
        du.Upload(id='video-uploader', 
                  text='Drag and Drop a single video file to upload',
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
        dcc.Input(id="skiprows", 
                  type="number", 
                  placeholder="Input number of rows to skip", 
                  disabled=True, 
                  style={'width': '400px', 'height': '20px', 'textAlign': 'center', 'display': 'flex', 'justifyContent': 'center'} # Sets width to 400px and height to 50px
        ), 
        ]),
        html.Div(id='csv-preview'), 
        html.Hr(), 
        html.Button(id='submit-button-state', n_clicks=0, children='Submit'), 
        dcc.Graph(id='Output-figure', figure={}), 
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
    Output("skiprows", "disabled"),
    Input('csv-upload', 'filename'),
    State('csv-box-text', 'children'),
    prevent_initial_call=True
)
def change_vid_text(filename, current_text): 
    if not filename:
        return current_text
    return f'{filename}', False

@app.callback(
    Output('Output-figure', 'figure'),
    Input('submit-button-state', 'n_clicks'),
    State('csv-upload', 'filename'),
    State('Stored-vidpath', 'data'),
    State('csv-upload', 'contents'),,
    State('skiprows', 'value'),
    prevent_initial_call=True
)
def generate_graph(n_clicks, filename: str, vidpath, filecontents: str, skiprows: int):
    if not isinstance(skiprows, int):
        skiprows = 0
    if n_clicks == 0:
        return
    else:
        csv_payload = {
            "contents": filecontents,
            "filename": filename,
            "skiprows": skiprows,
        }
        csv_response = requests.post(csv_endpoint, json=csv_payload)
        if csv_response.status_code != 200:
            print("CSV API failed:", csv_response.text)
            return
        csv_data = csv_response.json()
        time = csv_data["time"]
        volt = csv_data["volt"]
        print([csv_data["response"]])

        video_payload = {
            "time": time,
            "volt": volt,
            "vidpath": vidpath
        }
        video_response = requests.post(video_endpoint, json=video_payload)
        if video_response.status_code != 200:
            print("Video API failed:", video_response.text)
            return

        video_data = video_response.json()
        frames = video_data["frames"]
        strain = video_data["strain"]
        if len(frames)!=len(strain):
            raise IndexError('Frames and Strain are not the same length')
        fig = px.scatter(
        x=frames,
        y=strain,
        labels={'x': 'Frame Number', 'y': 'Strain'},
        title='Strain vs. Frame Number'
    )
        return fig

if __name__ == '__main__':
    app.run(debug=True)