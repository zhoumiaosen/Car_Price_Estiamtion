# -*- coding: utf-8 -*-
"""train.ipynb
#Load Functions
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
import joblib

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

"""#Locoal Function Build"""

# Define Neural Network
class PricePredictionModel(nn.Module):
    def __init__(self):
        super(PricePredictionModel, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 64)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

"""#Data Wash"""

# Load data
df = pd.read_csv('data/cars.csv', encoding='utf-16')
# Change Mileage NaN to 0 if Status is new
df['Mileage'] = df.apply(lambda row: 0 if (pd.isna(row['Mileage']) and row['Status'] == 'New') else row['Mileage'], axis=1)
# Drop rows where 'Status' is 'Used' and 'Mileage' is NaN
df = df.drop(df[(df['Status'] == 'Used') & (df['Mileage'].isna())].index)
# Drop rows with NaN in 'Price' column
df.dropna(subset=['Price'], inplace=True)
# Drop 'Dealer' column
df = df.drop('Dealer', axis=1)

"""#Encodeing Features to Traning Lables"""

# Encoder the Brand to traning lables
Brand_encoder = LabelEncoder()
df['Brand'] = Brand_encoder.fit_transform(df['Brand'])
# Encoder the Model to traning lables
Model_encoder = LabelEncoder()
df['Model'] = Model_encoder.fit_transform(df['Model'])
# Encoder the Status to traning lables
Status_encoder = LabelEncoder()
df['Status'] = Status_encoder.fit_transform(df['Status'])
# Save Encoder: Brand_encoder, Model_encoder, Status_encoder
joblib.dump(Brand_encoder, 'models/Brand_encoder.pkl')
joblib.dump(Model_encoder, 'models/Model_encoder.pkl')
joblib.dump(Status_encoder, 'models/Status_encoder.pkl')

"""#Normalize All Features & Target"""

# Split data into features (X) and target (y)
X = df.drop('Price', axis=1).values
y = df['Price'].values

# Normalize all features using MinMaxScaler
scaler_X = MinMaxScaler()
X = scaler_X.fit_transform(X)

# Normalize the target variable (y) using MinMaxScaler
scaler_y = MinMaxScaler()
y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()  # Flatten to match original shape

# Save Scaler: scaler_X & scaler_y
joblib.dump(scaler_X, 'models/scaler_X.pkl')
joblib.dump(scaler_y, 'models/scaler_y.pkl')

"""# Model Build"""

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

# Initialize the model, loss function, and optimizer
model = PricePredictionModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 300
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

torch.save(model, 'models/car_price_model_full.pth')