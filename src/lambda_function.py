import json
import logging
import pickle
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load model and scaler
model_path = 'model.pkl'
scaler_path = 'scaler.pkl'

# Load model
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Load scaler
with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)


def lambda_handler(event, context):
    logging.info("Received event: %s", event)

    try:
        # Parse the body from the event
        body = json.loads(event['body'])  # Parse the JSON body

        if 'features' not in body:
            logging.error("Missing 'features' key in the request body.")
            return {
                'statusCode': 400,
                'body': json.dumps({'error': "Missing 'features' key in the request body."})
            }

        logging.info("Extracting features.")
        features = np.array(body['features']).reshape(-1, 4)

        logging.info("Scaling features.")
        scaled_features = scaler.transform(features)

        logging.info("Making predictions.")
        predictions = model.predict(scaled_features).tolist()

        response = {
            "predictions": predictions,
            "prediction_probabilities": model.predict_proba(scaled_features).tolist(),
            "model_name": "RandomForestClassifier",
            "version": "1.0.0",
            "timestamp": "2024-09-21T10:42:01.116340",
            "input_features": body['features']
        }

        logging.info("Predictions generated successfully.")
        return {
            'statusCode': 200,
            'body': json.dumps(response)
        }

    except Exception as e:
        logging.error("Error in lambda_handler: %s", str(e))
        return {
            'statusCode': 400,
            'body': json.dumps({'error': str(e), 'timestamp': "2024-09-21T10:42:01.116340"})
        }
