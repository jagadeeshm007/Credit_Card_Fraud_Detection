from __future__ import print_function
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.exceptions import NotFittedError
import pickle
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

def load_model_and_scaler():

    try:
        with open(r'pickle_files/model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open(r'pickle_files/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    
    except FileNotFoundError:
        print("File not found. Please check the file path.")
    except pickle.UnpicklingError:
        print("Error in unpickling the file. The file might be corrupted or not a pickle file.")
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

def preprocess_input(amount, features, scaler):
    try:
        input_data = np.array([amount] + features).reshape(1, -1)
        
        # Suppress specific warning
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=UserWarning)
            input_data = scaler.transform(input_data)
        
        input_data = normalize(input_data, norm="l1")
        return input_data
    except NotFittedError as e:
        print(f"Scaler is not fitted: {e}")
        return None

def predict_fraud(amount, features):
    model, scaler = load_model_and_scaler()
    if model is None or scaler is None:
        print("Failed to load model or scaler.")
        return None
    
    input_data = preprocess_input(amount, features, scaler)
    if input_data is None:
        print("Failed to preprocess input.")
        return None

    prediction = model.predict(input_data)
    return prediction


if __name__ == "__main__":
    # Example usage input
    amount = 123.45
    features = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 28 features
    

    prediction = predict_fraud(amount, features)
    
    if prediction is not None:
        print(f"The transaction is {'fraudulent' if prediction == 1 else 'not fraudulent'}.")
    else:
        print("Prediction could not be made.")




