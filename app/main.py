from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
import pandas as pd
from .model import MachineLearningModel
from .schemas import PredictionInput

app = FastAPI(
    title="Manufacturing Predictor API",
    description="An API for predicting machine downtime based on operating parameters",
    version="1.0.0"
)

ml_model = MachineLearningModel()

@app.get("/", response_class=HTMLResponse)
async def root():
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Manufacturing Predictor API</title>
    <link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        /* Reset default margin and padding for all elements */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Fira Code', monospace;
            background-color: #000000;
            color: #ffffff;
            line-height: 1.6;
            padding: 40px 20px;
        }

        h1 {
            font-family: 'Fira Code', monospace;
            font-size: 36px;
            color: #00bfff; /* Blue text for heading */
            text-align: center;
            margin-bottom: 30px;
            font-weight: 600;
        }

        .container {
            background-color: #111111;
            border-radius: 12px;
            padding: 40px;
            margin: 20px auto;
            max-width: 900px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        }

        h2 {
            color: #00bfff; /* Blue text for subheadings */
            font-size: 28px;
            margin-bottom: 20px;
            font-weight: 500;
        }

        .doc-link {
            display: inline-block;
            padding: 14px 24px;
            background-color: #1e90ff; /* Lighter blue for links */
            color: white;
            text-decoration: none;
            border-radius: 6px;
            margin: 10px 0;
            font-size: 16px;
            transition: background-color 0.3s ease, transform 0.3s ease;
        }

        .doc-link:hover {
            background-color: #00bfff;
            transform: translateY(-2px);
        }

        .form-row {
            display: flex;
            flex-direction: column;
            gap: 15px;
            margin-bottom: 20px;
        }

        label {
            font-size: 16px;
            color: #ffffff; /* White text for labels */
            font-weight: 500;
        }

        input[type="file"],
        input[type="text"],
        input[type="submit"] {
            font-family: 'Fira Code', monospace;
            font-size: 16px;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #44475a;
            outline: none;
            background-color: #333333; /* Dark grey background for inputs */
            color: #ffffff; /* White text for inputs */
            transition: all 0.3s ease;
        }

        input[type="file"]:hover,
        input[type="text"]:hover {
            border-color: #00bfff; /* Blue border on hover */
        }

        input[type="submit"] {
            font-family: 'Fira Code', monospace; /* Standardized font for buttons */
            background-color: #00bfff; /* Blue button background */
            color: white;
            border: none;
            cursor: pointer;
            font-weight: bold;
            transition: background-color 0.3s ease, transform 0.3s ease;
        }

        input[type="submit"]:hover {
            background-color: #1e90ff; /* Lighter blue on hover */
            transform: translateY(-2px);
        }

        .method {
            display: inline-block;
            padding: 6px 18px;
            border-radius: 30px;
            font-weight: bold;
            margin-right: 10px;
        }

        .get { background-color: #61affe; color: white; }
        .post { background-color: #49cc90; color: white; }

        .endpoint {
            background-color: #222222; /* Dark background for endpoints */
            border: 1px solid #44475a;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }

        .footer {
            text-align: center;
            margin-top: 30px;
            font-size: 14px;
            color: #888888; /* Lighter text for footer */
        }

        .footer a {
            color: #1e90ff;
            text-decoration: none;
        }

        .footer a:hover {
            text-decoration: underline;
        }

    input[type=file]::file-selector-button {
    font-family: 'Fira Code', monospace; /* Standardized font for buttons */
    background-color: #00bfff; /* Blue button background */
    color: white;
    border: none;
    cursor: pointer;
    font-weight: bold;
    transition: background-color 0.3s ease, transform 0.3s ease;

    }

input[type=file]::file-selector-button:hover {
  background-color: #1e90ff; /* Lighter blue on hover */
  transform: translateY(-2px);
}
    </style>
</head>
<body>

    <h1 align="center">Manufacturing Predictor API</h1>

    <div class="container">
        <h2>Documentation (Inbuilt UI)</h2>
        <a href="/docs" class="doc-link">Interactive API Documentation (Swagger UI)</a>
        <a href="/redoc" class="doc-link">Alternative Documentation (ReDoc)</a>
    </div>

    <div class="container">
        <h2>Upload CSV Data</h2>
        <form action="/upload" method="post" enctype="multipart/form-data">
            <div class="form-row">
                <label for="file">Choose a CSV file to upload:</label>
                <input type="file" name="file" id="file" accept=".csv" required>
            </div>
            <input type="submit" value="Upload CSV">
        </form>
    </div>

    <div class="container">
        <h2>Train Model</h2>
        <form action="/train" method="post">
            <input type="submit" value="Train Model">
        </form>
    </div>

    <div class="container">
        <h2>Predict Downtime</h2>
        <form action="/predict" method="post">
            <div class="form-row">
                <label for="temperature">Temperature:</label>
                <input type="text" id="temperature" name="temperature" required>
            </div>
            <div class="form-row">
                <label for="run_time">Run Time:</label>
                <input type="text" id="run_time" name="run_time" required>
            </div>
            <input type="submit" value="Predict">
        </form>
    </div>

    <div class="footer">
        <p>Need help? Visit our <a href="/help">Help Center</a>.</p>
    </div>

</body>
</html>

    """
    return HTMLResponse(content=html_content)

@app.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Please upload a CSV file")
    
    try:
        df = pd.read_csv(file.file)
        ml_model.store_data(df)
        return JSONResponse(content={"message": "Data uploaded successfully"})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/train")
async def train_model():
    try:
        metrics = ml_model.train()
        return JSONResponse(content=metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import Form

@app.post("/predict")
async def predict(
    temperature: float = Form(...),
    run_time: float = Form(...)
):
    try:
        input_data = PredictionInput(Temperature=temperature, Run_Time=run_time)
        prediction = ml_model.predict(input_data)
        return JSONResponse(content=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

