# Use the official AWS Lambda Python 3.9 image
FROM public.ecr.aws/lambda/python:3.9

# Set up working directory
WORKDIR ${LAMBDA_TASK_ROOT}

# Copy the lambda function code
COPY src/lambda_function.py ${LAMBDA_TASK_ROOT}

# Copy model and scaler
COPY training/pkl/model.pkl ${LAMBDA_TASK_ROOT}
COPY training/pkl/scaler.pkl ${LAMBDA_TASK_ROOT}

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --target ${LAMBDA_TASK_ROOT}

# Set the CMD to your Lambda handler function (lambda_function.lambda_handler)
CMD ["lambda_function.lambda_handler"]
