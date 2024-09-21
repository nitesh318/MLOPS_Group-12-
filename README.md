Name of the active BITS contributors in this assignment:

1.Nitesh Bhushan (2022ac05001)  
2.Vishnu C (2022ac05028)  
3.Patil Sandeep Bhimrao (2022ac05149)  
4.Ravindra N C (2022ac05024)  
5.Anwaruddin Biswas (2022ac05278)



```markdown
# End-to-End Machine Learning Workflow with KizenML and XAI

## Project Overview
This project demonstrates an end-to-end machine learning (ML) pipeline that utilizes KizenML for exploratory data analysis (EDA), Scikit-learn for model training and evaluation, SHAP for Explainable AI (XAI), and AWS Lambda for model deployment. The model is trained on the Iris dataset, and the complete pipeline is deployed as a scalable and interpretable ML service using Docker and AWS Elastic Container Registry (ECR).

## Workflow Outline

### 1. Data Collection and Preprocessing
The Iris dataset was selected as the input data for this task. Key preprocessing steps included:
- **Data Cleaning**: Removed duplicate rows.
- **Feature Scaling**: Used `StandardScaler` to standardize the dataset’s features.
- **Feature Selection**: Retained all the features from the Iris dataset.

**Tools Used**:
- **Pandas** for data manipulation
- **Scikit-learn** for scaling and feature processing

### 2. Model Selection, Training, and Hyperparameter Tuning
The RandomForestClassifier from Scikit-learn was chosen for model training, and hyperparameters were optimized using GridSearchCV to ensure the model’s best performance.

- **Model**: RandomForestClassifier
- **Best Parameters**: Identified via GridSearchCV tuning
- **Evaluation Metrics**: Precision, Recall, F1 Score, ROC AUC Score

**Tools Used**:
- **Scikit-learn** for model training and evaluation
- **GridSearchCV** for hyperparameter tuning

### 3. Explainable AI (XAI) Implementation
Explainable AI (XAI) techniques were applied using SHAP (SHapley Additive exPlanations) to provide insights into the decision-making process of the RandomForest model. The SHAP values highlight the importance of each feature in contributing to the model’s predictions.

**Tools Used**:
- **SHAP** for XAI
- **Scikit-learn** for model prediction explanations

### 4. Model Deployment Using AWS Lambda and Docker

#### Docker for Local Testing
Before deploying to AWS Lambda, the application was containerized using Docker. This allowed for easy testing of the Flask API locally and ensured that the dependencies and environment remained consistent across different platforms.

#### Dockerfile Overview
A `Dockerfile` was created to package the Python application, including the trained model and scaler, along with the necessary libraries. The following steps outline the key parts of the Docker process:

1. **Base Image**: Used the official AWS Lambda Python 3.9 image as the base.
   ```dockerfile
   FROM public.ecr.aws/lambda/python:3.9
   ```

2. **Copy Application Code**: Copied the Lambda function and other required files (model, scaler) into the container.
   ```dockerfile
   COPY src/lambda_function.py ${LAMBDA_TASK_ROOT}
   COPY training/pkl/model.pkl ${LAMBDA_TASK_ROOT}
   COPY training/pkl/scaler.pkl ${LAMBDA_TASK_ROOT}
   ```

3. **Install Dependencies**: Installed all necessary Python libraries defined in `requirements.txt`.
   ```dockerfile
   RUN pip install --no-cache-dir -r requirements.txt --target ${LAMBDA_TASK_ROOT}
   ```

4. **Set CMD**: Defined the entry point for the Lambda function.
   ```dockerfile
   CMD ["lambda_function.lambda_handler"]
   ```

#### Local Testing with Docker
To test the application locally using Docker:

1. **Build the Docker Image**:
   ```bash
   docker build -t ml-deployment .
   ```

2. **Run the Docker Container**:
   ```bash
   docker run -p 9001:9001 ml-deployment
   ```

3. **Access the API**:
   The Flask API, which served the model’s predictions, was accessible at `http://localhost:9001/predict`.

#### Pushing the Docker Image to AWS ECR

To deploy the application to AWS Lambda using ECR (Elastic Container Registry), the following steps were performed:

1. **Create an ECR Repository**:
   Create a repository in AWS ECR where the Docker image will be stored.

   ```bash
   aws ecr create-repository --repository-name ml-deployment --region <region-name>
   ```

2. **Authenticate Docker to ECR**:
   Authenticate your Docker client with ECR:
   ```bash
   aws ecr get-login-password --region <region-name> | docker login --username AWS --password-stdin <aws-account-id>.dkr.ecr.<region-name>.amazonaws.com
   ```

3. **Tag the Docker Image**:
   Tag the local Docker image with the ECR repository URL:
   ```bash
   docker tag ml-deployment:latest <aws-account-id>.dkr.ecr.<region-name>.amazonaws.com/ml-deployment:latest
   ```

4. **Push the Docker Image to ECR**:
   Push the image to your ECR repository:
   ```bash
   docker push <aws-account-id>.dkr.ecr.<region-name>.amazonaws.com/ml-deployment:latest
   ```

#### Deploying to AWS Lambda
After pushing the image to ECR, it was linked to an AWS Lambda function for deployment.

1. **Create the Lambda Function**:
   In the AWS Management Console, create a new Lambda function and select the "Container Image" option. Choose the image from ECR that was pushed in the previous steps.

2. **Configure API Gateway**:
   To make the Lambda function accessible over the web, an API Gateway was set up. This allowed external requests to invoke the Lambda function and get predictions.

3. **Test the Deployment**:
   The deployed Lambda function can now handle incoming requests and return predictions. The function was tested by invoking the API Gateway endpoint and submitting a `POST` request with the following JSON body:

   ```json
   {
       "features": [5.1, 3.5, 1.4, 0.2]
   }
   ```

   The API responds with the model’s prediction and probabilities.

### Conclusion
This project successfully demonstrates an end-to-end machine learning workflow, from data preprocessing and model training to deployment with Docker, AWS Lambda, and ECR. Explainable AI (XAI) techniques were applied to ensure transparency and interpretability of the model's predictions.

The Docker containerization process made local testing easy and seamless, while AWS ECR and Lambda provided scalable and serverless deployment capabilities.
