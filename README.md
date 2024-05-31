# Automated Diabetic Retinopathy Screening System

This project aims to develop an automated system using neural networks to screen for diabetic retinopathy, which is a leading cause of blindness among diabetic patients. The system analyzes retinal photographs to detect the presence and severity of retinopathy, providing a valuable tool for early diagnosis and treatment.

# How to Use the Code
To use this project, follow these steps:

- # Clone the Repository:
  Clone the repository to your local machine using the following command:
  git clone https://github.com/VadlamudiVarshitaa/DiabeticRetinopathy.git

- # Setup Environment:
  Ensure you have Python and the necessary libraries installed. You can install the required libraries using the following command:
  pip install -r requirements.txt

- # Prepare Data:
  Place your dataset of retinal photographs in the data directory. The dataset should be in the format where each image corresponds to a label indicating the presence or absence of diabetic retinopathy.
Update the data_loader.py script if necessary to match the format of your dataset.

- # Train the Model:
  Open and run the Jupyter Notebook train_model.ipynb to train the neural network on the dataset. This notebook includes steps for data preprocessing, model training, and evaluation.

- # Evaluate the Model:
  Use the evaluate_model.py script to evaluate the trained model on a test dataset. This script will output performance metrics such as accuracy, precision, recall, and F1-score.

- # Make Predictions:
  Use the predict.py script to make predictions on new retinal photographs. Place the images you want to analyze in the input_images directory and run the script to get predictions.

# Key Features
- Data Preprocessing: Automated procedures to preprocess and augment retinal images for training.
- Model Training: A convolutional neural network (CNN) specifically designed for image classification tasks, optimized for detecting diabetic retinopathy.
- Evaluation Metrics: Comprehensive evaluation metrics to assess model performance, including confusion matrix, accuracy, precision, recall, and F1-score.
- User-Friendly Predictions: An easy-to-use interface for making predictions on new retinal photographs.

# Contact
For any inquiries or further information, please contact:
- Name: V. Varshitaa
- Email: varshitaa21@gmail.com
