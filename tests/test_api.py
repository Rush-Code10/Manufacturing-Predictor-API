# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from app.main import app
import pandas as pd
import io
import json

client = TestClient(app)

def create_test_csv():
    test_data = {
        'Machine_ID': [1, 2, 3],
        'Temperature': [75.0, 85.0, 95.0],
        'Run_Time': [100.0, 120.0, 140.0],
        'Downtime_Flag': [0, 0, 1]
    }
    df = pd.DataFrame(test_data)
    csv_file = io.StringIO()
    df.to_csv(csv_file, index=False)
    csv_file.seek(0)
    return csv_file

def test_upload_endpoint_success():
    csv_file = create_test_csv()
    files = {'file': ('test.csv', csv_file, 'text/csv')}
    response = client.post("/upload", files=files)
    
    assert response.status_code == 200
    assert response.json() == {"message": "Data uploaded successfully"}

def test_upload_endpoint_invalid_file():
    files = {'file': ('test.txt', 'invalid content', 'text/plain')}
    response = client.post("/upload", files=files)
    
    assert response.status_code == 400
    assert "Please upload a CSV file" in response.json()["detail"]

def test_train_endpoint():

    csv_file = create_test_csv()
    files = {'file': ('test.csv', csv_file, 'text/csv')}
    client.post("/upload", files=files)
    
    response = client.post("/train")
    
    assert response.status_code == 200
    assert "accuracy" in response.json()
    assert "f1_score" in response.json()

def test_predict_endpoint():
    csv_file = create_test_csv()
    files = {'file': ('test.csv', csv_file, 'text/csv')}
    client.post("/upload", files=files)
    client.post("/train")
    
    test_input = {
        "Temperature": 85.0,
        "Run_Time": 120.0
    }
    response = client.post("/predict", json=test_input)
    
    assert response.status_code == 200
    assert "Downtime" in response.json()
    assert "Confidence" in response.json()
    assert isinstance(response.json()["Downtime"], str)
    assert isinstance(response.json()["Confidence"], float)

def test_predict_endpoint_invalid_input():
    invalid_input = {
        "Temperature": "invalid", 
        "Run_Time": 120.0
    }
    response = client.post("/predict", json=invalid_input)
    
    assert response.status_code == 422

def test_predict_without_training():

    test_client = TestClient(app)
    
    test_input = {
        "Temperature": 85.0,
        "Run_Time": 120.0
    }
    response = test_client.post("/predict", json=test_input)
    
    assert response.status_code == 500
    assert "Model not trained yet" in response.json()["detail"]

def test_complete_workflow():
    csv_file = create_test_csv()
    files = {'file': ('test.csv', csv_file, 'text/csv')}
    upload_response = client.post("/upload", files=files)
    assert upload_response.status_code == 200
    
    train_response = client.post("/train")
    assert train_response.status_code == 200
    
    test_input = {
        "Temperature": 85.0,
        "Run_Time": 120.0
    }
    predict_response = client.post("/predict", json=test_input)
    assert predict_response.status_code == 200
    
    prediction = predict_response.json()
    assert prediction["Downtime"] in ["Yes", "No"]
    assert 0 <= prediction["Confidence"] <= 1

if __name__ == "__main__":
    pytest.main([__file__])