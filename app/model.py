from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np

class MachineLearningModel:
    def __init__(self):
        self.model = DecisionTreeClassifier(random_state=42, max_depth=5) 
        self.scaler = StandardScaler()
        self.data = None
        self.features = ['Temperature', 'Run_Time']
        self.target = 'Downtime_Flag'
        
    def store_data(self, data: pd.DataFrame):
        required_columns = self.features + [self.target]
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
        self.data = data
        
    def train(self):
        if self.data is None:
            raise ValueError("No data available for training")
            
        X = self.data[self.features]
        y = self.data[self.target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model.fit(X_train_scaled, y_train)
        
        y_pred = self.model.predict(X_test_scaled)
        
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_score": float(f1_score(y_test, y_pred, average='weighted'))
        }
        
        return metrics
        
    def predict(self, input_data):
        if not self.model:
            raise ValueError("Model not trained yet")
            
        input_df = pd.DataFrame([{
            'Temperature': input_data.Temperature,
            'Run_Time': input_data.Run_Time
        }])
        
        input_scaled = self.scaler.transform(input_df)
        print("Input Scaled Values:", input_scaled)  

        prediction = self.model.predict(input_scaled)[0]
        probabilities = self.model.predict_proba(input_scaled)[0]
        print("Probabilities:", probabilities)
        
        probability = probabilities.max()
        
        return {
            "Downtime": "Yes" if prediction == 1 else "No",
            "Confidence": float(probability)
        }
