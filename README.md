# Manufacturing Predictor API

This API predicts machine downtime based on operating parameters using machine learning.

## Setup Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd manufacturing-predictor
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Generate sample data:
```bash
python generate_sample_data.py
```

5. Start the API:
```bash
uvicorn app.main:app --reload
```

## API Endpoints

### 1. Upload Data
- **Endpoint:** `POST /upload`
- **Input:** CSV file with columns: Machine_ID, Temperature, Run_Time, Downtime_Flag
```bash
curl -X POST -F "file=@data/sample_data.csv" http://localhost:8000/upload
```

### 2. Train Model
- **Endpoint:** `POST /train`
- **Returns:** Model performance metrics
```bash
curl -X POST http://localhost:8000/train
```

### 3. Make Prediction
- **Endpoint:** `POST /predict`
- **Input:** JSON with Temperature and Run_Time values
```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"Temperature": 85, "Run_Time": 120}' \
     http://localhost:8000/predict
```

## Example Responses

### Training Response
```json
{
    "accuracy": 0.85,
    "f1_score": 0.83
}
```

### Prediction Response
```json
{
    "Downtime": "Yes",
    "Confidence": 0.85
}
```
