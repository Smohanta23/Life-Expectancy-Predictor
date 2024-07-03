# Life Expectancy Predictor

A real-time application that estimates a person's life expectancy based on age and gender predictions from facial images. This project leverages deep learning models for age and gender detection and integrates a pre-trained regression model to dynamically predict life expectancy.

## Features

- **Real-time Face Detection**: Utilizes OpenCV's deep learning-based face detector to locate faces in video streams.
- **Age and Gender Prediction**: Implements deep learning models to predict the age and gender of the detected face.
- **Life Expectancy Estimation**: Uses a pre-trained regression model to estimate life expectancy based on the predicted age and gender.
- **Dynamic Interface**: Displays the predicted age, gender, and life expectancy in a visually appealing and responsive interface.

## Data Preparation and Model Training

1. **Data Source**: 
   - The dataset used for training the life expectancy model is sourced from Kaggle, titled "Life Expectancy Data.csv".

2. **Data Processing**:
   - **Data Cleaning**: 
     - Removed unnecessary spaces from column names.
     - Checked and handled missing values by either dropping rows or filling them with mean values.
   - **Feature Engineering**: 
     - Encoded categorical variables such as country status (Developing or Developed).
     - Selected relevant features for the prediction model, excluding non-predictive columns like Country and Year.

3. **Model Training**:
   - **Model Selection**: 
     - Used RandomForestRegressor from scikit-learn for its robustness and accuracy in regression tasks.
   - **Training and Evaluation**:
     - Split the data into training and testing sets.
     - Trained the model on the training set and evaluated it on the testing set using metrics like Mean Squared Error (MSE) and R-squared (R²).
   - **Saving the Model**: 
     - Saved the trained model along with the feature names for future use.

## Implementation in Real-time Application

The trained life expectancy model is integrated into a real-time video processing application:
- **Face Detection**: Identifies faces in the video stream.
- **Age and Gender Prediction**: Predicts the age range and gender for each detected face.
- **Life Expectancy Prediction**: Estimates the life expectancy dynamically based on the predicted age and gender using the pre-trained regression model.

## How to Use

1. **Clone the Repository**:
   ```sh
   git clone https://github.com/Smohanta23/Life-Expectancy-Predictor.git
   cd Life-Expectancy-Predictor
2. **requirements**:
   ```sh
   opencv-python-headless==4.5.5.64
   scikit-learn==0.24.2
   pandas==1.3.3
   joblib==1.0.1
   numpy==1.21.2
3. **Install the "requirements.txt" file**
4. **Run the Application**:
   ```sh
   python main.py


