# -*- coding: utf-8 -*-
"""predict.ipynb
#Load Functions
"""

import torch
import torch.nn as nn
import pandas as pd
import joblib
from model import initialize_model
from model import PricePredictionModel

"""#Model Function"""

# Define Neural Network
class PricePredictionModel(nn.Module):
    def __init__(self):
        super(PricePredictionModel, self).__init__()
        self.fc1 = nn.Linear(5, 64)
        self.dropout1 = nn.Dropout(0.5)  # Dropout layer
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.5)  # Dropout layer
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

def load_model_and_scalers(model_path, scaler_X_path, scaler_y_path, Brand_encoder_path, Model_encoder_path, Status_encoder_path,input_size, device):
    # Load the model
    #model = initialize_model(input_size).to(device)
    model_load = torch.load(model_path, map_location=device)
    model_load.eval()

    # Load the scalers
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)

    # Load Encoders
    Brand_encoder = joblib.load(Brand_encoder_path)
    Model_encoder = joblib.load(Model_encoder_path)
    Status_encoder = joblib.load(Status_encoder_path)

    return model_load, scaler_X, scaler_y, Brand_encoder, Model_encoder, Status_encoder

'''
    new_data = pd.DataFrame({
    'Brand': ['Toyota'],
    'Model': ['Camry'],
    'Year': [2020],
    'Status': ['Used'],
    'Mileage': [15000.0],
              })
'''
def preprocess_input(new_data, scaler_X, Brand_encoder,Model_encoder,Status_encoder):

    # Encode and normalize new data using the same MinMaxScaler
    new_data['Brand'] = Brand_encoder.transform(new_data['Brand'])
    new_data['Model'] = Model_encoder.transform(new_data['Model'])
    new_data['Status'] = Status_encoder.transform(new_data['Status'])

    new_data_scaled = scaler_X.transform(new_data.values)
    new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32)

    return new_data_tensor

def predict_price(model, X_tensor, scaler_y, device):
    # Move the input tensor to the appropriate device
    X_tensor = X_tensor.to(device)

    # Make predictions
    with torch.no_grad():
        predicted_price_normalized = model( X_tensor)
        predicted_price = scaler_y.inverse_transform(predicted_price_normalized.numpy().reshape(-1, 1))

    return predicted_price

"""#__ main __"""

if __name__ == "__main__":
    # Define file paths
    model_path = 'models/car_price_model_full.pth'
    scaler_X_path = 'models/scaler_X.pkl'
    scaler_y_path = 'models/scaler_y.pkl'
    Brand_encoder_path = 'models/Brand_encoder.pkl'
    Model_encoder_path = 'models/Model_encoder.pkl'
    Status_encoder_path = 'models/Status_encoder.pkl'

    # Example data
    new_data = pd.DataFrame({
    'Brand': ['Toyota'],
    'Model': ['Camry'],
    'Year': [2020],
    'Status': ['Used'],
    'Mileage': [15000.0],
              })

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the model and scalers
    input_size = len(new_data.columns)
    model, scaler_X, scaler_y, Brand_encoder, Model_encoder, Status_encoder = load_model_and_scalers(model_path, scaler_X_path, scaler_y_path, Brand_encoder_path, Model_encoder_path, Status_encoder_path,input_size, device)

    # Preprocess the input data
    X_tensor = preprocess_input(new_data, scaler_X, Brand_encoder,Model_encoder,Status_encoder)

    # Predict the car price
    predicted_price = predict_price(model, X_tensor, scaler_y, device)

    # Print the predicted prices
    #print("Input Data:")
    #print(new_data)
    #print(f"Predicted Price: {predicted_price[0][0]:.2f}")

    new_data['Predicted Price'] = predicted_price
    print(new_data)