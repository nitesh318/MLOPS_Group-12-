import logging
import pickle
import numpy as np
from flask import Flask, request, jsonify
from sklearn.metrics import classification_report, confusion_matrix

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)


# Load the model and scaler
def load_model_and_scaler():
    try:
        with open('training/pkl/model.pkl', 'rb') as file:
            model = pickle.load(file)
            logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error("Error loading model: %s", str(e))
        model = None

    try:
        with open('training/pkl/scaler.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
            logging.info("Scaler loaded successfully.")
    except Exception as e:
        logging.error("Error loading scaler: %s", str(e))
        scaler = None

    return model, scaler


best_model, scaler = load_model_and_scaler()

# Check if the model and scaler are loaded correctly
if best_model is None or scaler is None:
    logging.error("Failed to load model and/or scaler.")


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(-1, 4)
    scaled_features = scaler.transform(features)

    logging.info("Input features: %s", features)
    logging.info("Scaled features: %s", scaled_features)

    predictions = best_model.predict(scaled_features)
    logging.info("Raw predictions: %s", predictions)

    if np.all(predictions == 0):
        logging.warning("All predictions are zero for input: %s", features)
        return jsonify({
            'message': 'All predictions are 0. Please adjust your input values for better results.'
        }), 200

    return jsonify({
        'predictions': predictions.tolist(),
        'message': 'Predictions generated successfully.'
    })


@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.get_json()
    logging.info("Received evaluation data: %s", data)

    try:
        features = np.array(data['features'])
        true_labels = np.array(data['true_labels'])
        scaled_features = scaler.transform(features)
        predictions = best_model.predict(scaled_features)

        cm = confusion_matrix(true_labels, predictions)
        report = classification_report(true_labels, predictions, output_dict=True)

        response = {
            'classification_report': {str(label): {
                "precision": float(report[str(label)]['precision']),
                "recall": float(report[str(label)]['recall']),
                "f1-score": float(report[str(label)]['f1-score']),
                "support": int(report[str(label)]['support'])
            } for label in report if label.isdigit()},
            'accuracy': float(report['accuracy']),
            'macro avg': {key: float(report['macro avg'][key]) for key in
                          ['precision', 'recall', 'f1-score', 'support']},
            'weighted avg': {key: float(report['weighted avg'][key]) for key in
                             ['precision', 'recall', 'f1-score', 'support']},
            'confusion_matrix': cm.tolist()
        }

        return jsonify(response)

    except Exception as e:
        logging.error("Error during evaluation: %s", str(e))
        return jsonify({'error': 'Invalid input data'}), 400


if __name__ == "__main__":
    app.run(debug=False, port=9001)  # Set debug=False for production
