## Overview

This project aims to predict the price of used cars based on various features like brand, model, year, mileage, and status. The prediction model is built using a neural network in PyTorch, and the data preprocessing is handled with scikit-learn.

## Project Structure
```plaintext
car_price_prediction/
│
├── data/
│   └── cars.csv                     	  # Original dataset
│
├── models/
│   ├── car_price_model.pth         	  # Saved model state dictionary
│   ├── scaler_X.pkl                	  # Saved X scaler (MinMaxScaler for features)
│   ├── scaler_y.pkl                	  # Saved y scaler (MinMaxScaler for target)
│   ├── Model_encoder.pkl         	  # Saved encoder (encoder for car model)
│   ├── Brand_encoder.pkl            	  # Saved encoder (encoder for car brand)
│   └── Status.pkl              	  # Saved encoder (encoder for car status)
│
├── src/
│   ├── train.py                          # Script for training the model
│   ├── predict.py                        # Script for making predictions
│   └── model.py                          # Model definition (PyTorch neural network)
│
├── notebooks/
│   └── Used_Car_Price_Estimation.ipynb   # Jupyter notebook for data exploration and preprocessing
│
└── README.md                             # Project overview and instructions
```

## Requirements

- Python 3.7+
- PyTorch
- scikit-learn
- pandas
- joblib (optional, for saving and loading scalers)
- Jupyter Notebook (optional, for data exploration)

You can install the necessary packages using the following command:

```bash
pip install torch scikit-learn pandas joblib
```

## How to Run

### 1. Data Exploration

Before training the model, you can explore the dataset using the Jupyter notebook provided:

```bash
jupyter notebook notebook/Used_Car_Price_Estimation.ipynb
```

### 2. Training the Model
To train the model, run the train.py script:

```bash
python src/train.py
```

This script will:

- Load the data from data/car.csv.
- Encode categorical variables and normalize the features using MinMaxScaler.
- Train a neural network model using PyTorch.
- Save the trained model and scalers in the models/ directory.


### 3. Model Architecture
The model is a simple feedforward neural network with the following layers:

Input Layer: Matches the number of features after preprocessing (5 in this project).
Hidden Layers: Two hidden layers with ReLU activation.
Output Layer: A single neuron with ReLU to predict the price.

### 4. Saving and Loading the Model
Model: The trained model's state dictionary is saved as car_price_model.pth in the models/ directory.
Scalers: The MinMaxScaler objects used to normalize features (scaler_X.pkl) and the target variable (scaler_y.pkl) are also saved in the models/ directory.
Encoder: The Encoder for model (Model_encoder.pkl), brand (Brand_encoder.pkl), and status (Status_encoder.pkl) are all saved in the models/ directory.

To load the model scalers, and encoder:
```bash
import torch
import joblib
# Load the model
model_load = torch.load(car_price_model.pth, map_location=device)
model_load.eval()

# Load the scalers
scaler_X = joblib.load(scaler_X.pkl)
scaler_y = joblib.load(scaler_y.pkl)

# Load Encoders
Brand_encoder = joblib.load(Brand_encoder.pkl)
Model_encoder = joblib.load(Model_encoder.pkl)
Status_encoder = joblib.load(Status_encoder.pkl)
```

### 5. Making Predictions
To make predictions using the trained model, run the predict.py script:

```bash
python src/predict.py
```

This script will:

- Load the saved model and scalers from the models/ directory.
- Preprocess the input data in the same way as the training data.
- Output the predicted price for the given car features.

### 6. Predict Results
Input Data:<br>
<img src="assets/images/inputdata.png" alt="Diagram" width="150">

Results:<br>
<img src="assets/images/results.png" alt="Diagram" width="350">


## Next Steps
**Model Tuning:**
 Experiment with different neural network architectures and hyperparameters to improve model performance.
**Feature Engineering:**
 Consider adding more features or transforming existing ones to capture more information from the data.
**Evaluation:** 
Implement more comprehensive evaluation metrics, like MAE or R^2, to better assess the model's performance.
**Model Deploy:**
 Deploy the trained model to a website for user interaction and real-time predictions.
**Data Expansion:** 
 Gather and integrate additional datasets to enhance the model’s accuracy and robustness.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request to contribute to the project.

## License
This project is licensed under the MIT License.
