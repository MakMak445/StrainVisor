from dash import Dash, html, dcc, callback, Output, Input, State
import base64
import pandas as pd
import io
import dash_player as dp

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
        html.Figure(id = 'Output-figure')
    ])
        ])
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
'''
    html.Div([
        html.H3("Upload a Video File"),
        dcc.Upload(
            id='video-upload',
            children=html.Div(['Drag and Drop or ', html.A('Select a Video')]),
            accept='video/*,.cine,.mp4',
            multiple=False,
            style={
                'width': '60%', 'height': '80px', 'lineHeight': '80px',
                'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                'textAlign': 'center', 'margin': '10px auto'
            }
        ),
        html.Div(id='video-preview')
    ]),
    html.Hr(),
    html.Div([
        html.H3("Upload a CSV File"),
        dcc.Upload(
            id='csv-upload',
            children=html.Div(['Drag and Drop or ', html.A('Select a CSV')]),
            accept='.csv',
            multiple=False,
            style={
                'width': '60%', 'height': '80px', 'lineHeight': '80px',
                'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                'textAlign': 'center', 'margin': '10px auto'
            }
        ),
        html.Div(id='csv-preview')
    ])
])

@callback(
    Output('video-preview', 'children'),
    Input('video-upload', 'contents'),
    State('video-upload', 'filename')
)
def preview_video(contents, filename):
    if not contents:
        return html.Div()

    # split off the “data:video/…;base64” header so we can pull out the mime type
    header, b64 = contents.split(',', 1)
    mime = header.split(';')[0].split(':')[1]  # e.g. "video/mp4"

    return dp.DashPlayer(url=contents,
        controls=True,
        playing=False,
        width="80%",
        height="360px"
    )

@callback(
    Output('csv-preview', 'children'),
    Input('csv-upload', 'contents'),
    State('csv-upload', 'filename')
)
def preview_csv(contents, filename):
    if contents is None:
        return html.Div()
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    except Exception as e:
        return html.Div([f"Error parsing CSV: {e"])
    # Show first few rows
    return html.Div([
        html.H5(f""),
        html.Pre(df.head().to_csv(index=False))
    ])
'''
if __name__ == '__main__':
    app.run(debug=True)
