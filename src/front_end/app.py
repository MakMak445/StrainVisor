from dash import Dash, html, dcc, Output, Input, State
import plotly.express as px
import dash_uploader as du
import requests
import numbers
import numpy as np

app = Dash(__name__)
server = app.server

VIDEO_API_URL = "http://video_backend:8000"
CSV_API_URL = "http://data_backend:8000"

video_endpoint = f"{VIDEO_API_URL}/process-video/"
csv_endpoint = f"{CSV_API_URL}/process-csv/"

du.configure_upload(app, "/app/uploads", use_upload_id=True)

app.layout = html.Div([
    html.H1("Drop Weight Analysis Tool"),

    # === VIDEO UPLOAD ===
    html.Div([
        html.H3("Upload High-Speed Impact Footage"),
        du.Upload(
            id='video-uploader',
            text='Drag and Drop a single video file to upload',
            filetypes=['mp4', 'mov', 'avi', 'cine'],
            max_file_size=2000,
            max_files=1,
        ),
        html.Div(id='status-output'),
        dcc.Store(id='Stored-vidpath'),
        dcc.Store(id='history-vidpath')
    ]),

    html.Hr(),

    # === CSV UPLOAD ===
    html.Div([
        html.H3("Upload a CSV File of Your Force-Time Graph"),
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

        # === INPUT FIELDS ===
        html.Div([
            html.Div([
                html.Label("Skip rows:"),
                dcc.Input(
                    id="skiprows",
                    type="number",
                    placeholder="Number of rows to skip",
                    disabled=True,
                    className="input-box"
                ),
            ]),
            html.Div([
                html.Label("Cross-sectional area (m²):"),
                dcc.Input(
                    id="cross_section_area",
                    type="number",
                    placeholder="Enter area in m²",
                    disabled=True,
                    className="input-box"
                ),
            ]),
            html.Div([
                html.Label("Force transducer coefficient (N/mV):"),
                dcc.Input(
                    id="force_transducer_coeff",
                    type="number",
                    placeholder="Enter coefficient in N/V",
                    disabled=True,
                    className="input-box"
                ),
            ]),
        ], className="input-container"),

    ]),

    html.Div(id='csv-preview'),

    html.Hr(),

    html.Button(id='submit-button-state', n_clicks=0, children='Submit'),

    dcc.Graph(id='Output-figure', figure={}),

    html.Hr(),

    html.Video(id='video-with-lines')
])

@du.callback(
    output=Output('Stored-vidpath', 'data'),
    id='video-uploader'
)
def handle_upload(status):
    return str(status.latest_file)


@app.callback(
    Output('csv-box-text', 'children'),
    Output("skiprows", "disabled"),
    Output("cross_section_area", "disabled"),
    Output("force_transducer_coeff", "disabled"),
    Input('csv-upload', 'filename'),
    State('csv-box-text', 'children'),
    prevent_initial_call=True
)
def change_vid_text(filename, current_text): 
    if not filename:
        return current_text, True
    return f'{filename}', False, False, False, 


@app.callback(
    Output('Output-figure', 'figure'),
    Input('submit-button-state', 'n_clicks'),
    State('csv-upload', 'filename'),
    State('Stored-vidpath', 'data'),
    State('csv-upload', 'contents'),
    State('skiprows', 'value'),
    State('cross_section_area', 'value'),
    State("force_transducer_coeff", 'value'),
    prevent_initial_call=True
)
def generate_graph(n_clicks, filename: str, vidpath, filecontents: str, skiprows: int, cross_sect_area: float, force_transducer_coeff: float):
    if not isinstance(skiprows, int):
        skiprows = 0
    if not isinstance(cross_sect_area, numbers.Real):
        cross_sect_area = 1
    if not isinstance(force_transducer_coeff, numbers.Real):
        force_transducer_coeff = 1
    if n_clicks == 0:
        return {}
    else:
        csv_payload = {"contents": filecontents,
                        "filename": filename, 
                        "skiprows": skiprows, 
                        }
        csv_response = requests.post(csv_endpoint, json=csv_payload)
        if csv_response.status_code != 200:
            print("CSV API failed:", csv_response.text)
            return {}
        csv_data = csv_response.json()
        time = csv_data["time"]
        volt = csv_data["volt"]

        video_payload = {"time": time, 
                         "volt": volt, 
                         "vidpath": vidpath,
                         }
        video_response = requests.post(video_endpoint, json=video_payload)
        if video_response.status_code != 200:
            print("Video API failed:", video_response.text)
            return {}

        video_data = video_response.json()
        stress = (np.asarray(video_data["stress"]) * force_transducer_coeff) / cross_sect_area
        strain = np.asarray(video_data["strain"])
        if len(stress) != len(strain):
            raise IndexError('Frames and Strain are not the same length')

        fig = px.scatter(
            x=strain,
            y=stress,
            labels={'x': 'Strain', 'y': 'Stress (Pa)'},
            title='Stress vs Strain'
        )
        return fig


if __name__ == '__main__':
    app.run(debug=True)
