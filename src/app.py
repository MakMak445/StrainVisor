from dash import Dash, html, dcc, callback, Output, Input, State, dash_table
import datetime
import base64
import pandas as pd
import io
import dash_player as dp
import plotly.express as px
import video_tools as vt
import base64, tempfile, os
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



app = Dash(__name__)

app.layout = html.Div([
    html.H1("Drop Weight Analysis Tool", style={'textAlign': 'center'}), 
    html.Div([
        html.H3("Upload High speed impact footage"), 
        dcc.Upload(
            id='video-upload',
            children=html.Div(['Drag and Drop or ', html.A('Select a Video')], id='vid-box-text'),
            accept='video/*,.cine,.mp4',
            multiple=False,
            style={
                'width': '60%', 'height': '80px', 'lineHeight': '80px',
                'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                'textAlign': 'center', 'margin': '10px auto'
            }
        ), 
        html.Hr(), 
        html.Div([
        html.H3("Upload a CSV File of your force-time graph"),
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
    ])
        ]), 
        html.Hr(),
        html.Video(id='video-with-lines')
])

@callback(
    Output('vid-box-text', 'children'),
    Input('video-upload', 'filename'),
    State('vid-box-text', 'children')
)
def change_vid_text(filename, current_text): 
    if not filename:
        return current_text
    return f'{filename}'

@callback(
    Output('csv-box-text', 'children'),
    Input('csv-upload', 'filename'),
    State('csv-box-text', 'children')
)
def change_vid_text(filename, current_text): 
    if not filename:
        return current_text
    return f'{filename}'

@callback(
    Output('Output-figure', 'figure'),
    Input('submit-button-state', 'n_clicks'),
    State('csv-upload', 'filename'),
    State('video-upload', 'filename'),
    State('csv-upload', 'contents'),
    State('video-upload', 'contents'),
    State('csv-upload', 'last_modified')
)
def generate_graph(n_clicks, filename, vidname, filecontents, vidcontents, date):
    if n_clicks ==0:
        return
    else:
        df = parse_contents(filecontents, filename, date)
        print(df.to_string)
        time = df['Time'].iloc[1:]
        volt = df['Channel A'].iloc[1:]
        print(time)
        print(volt)
        vt.generate_strain_graph(time, volt, )

if __name__ == '__main__':
    app.run(debug=True)
