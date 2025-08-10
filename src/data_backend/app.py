import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import io
from scipy.signal import find_peaks

def parse_contents(contents, filename):
    _, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
        return df
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

class CsvData(BaseModel):
    contents: str
    filename: str
    skiprows: int

app = FastAPI(title='Backend csv Analysis API')

@app.post("/process-csv/")
def obtain_time_and_volt(request: CsvData):
    try:
        df = parse_contents(request.contents, request.filename)
        skip_index = request.skiprows
        time = df.iloc[skip_index:, 0]
        volt = df.iloc[skip_index:, 1]
        return {
            "time": time.tolist(),
            "volt": volt.tolist(),
            "response":'file analysed successfully' 
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")