import cv2
import joblib
import pandas as pd

class ModelWithFeatures:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names

def load_models():
    # Load face detection model
    face_proto = "deploy.prototxt"
    face_model = "res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(face_model, face_proto)

    # Load age estimation model
    age_proto = "age_deploy.prototxt"
    age_model = "age_net.caffemodel"
    ageNet = cv2.dnn.readNet(age_model, age_proto)

    # Load gender estimation model
    gender_proto = "gender_deploy.prototxt"
    gender_model = "gender_net.caffemodel"
    genderNet = cv2.dnn.readNet(gender_model, gender_proto)

    return faceNet, ageNet, genderNet

def faceBox(net, frame, conf_threshold=0.7):
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), MODEL_MEAN_VALUES, swapRB=False)

    net.setInput(blob)
    detections = net.forward()
    bboxs = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxs.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame, bboxs

def predict_age_gender(face, ageNet, genderNet, ageList, genderList):
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

    # Set inputs and forward pass through ageNet
    ageNet.setInput(blob)
    age_preds = ageNet.forward()
    age = ageList[age_preds[0].argmax()]

    # Set inputs and forward pass through genderNet
    genderNet.setInput(blob)
    gender_preds = genderNet.forward()
    gender_index = gender_preds[0].argmax()
    gender = genderList[gender_index]

    return gender, age


def process_features(age_numeric, gender, feature_names):
    features = {
        'Age': age_numeric,
        'Gender': 0 if gender == 'Male' else 1,
        'Status': 0,  # Placeholder, update with actual data (0 for Developing, 1 for Developed)
        'Adult_Mortality': 0,  # Placeholder, update with actual data
        'infant_deaths': 0,  # Placeholder, update with actual data
        'Alcohol': 0,  # Placeholder, update with actual data
        'percentage_expenditure': 0,  # Placeholder, update with actual data
        'Hepatitis_B': 0,  # Placeholder, update with actual data
        'Measles': 0,  # Placeholder, update with actual data
        'BMI': 0,  # Placeholder, update with actual data
        'under-five_deaths': 0,  # Placeholder, update with actual data
        'Polio': 0,  # Placeholder, update with actual data
        'Total_expenditure': 0,  # Placeholder, update with actual data
        'Diphtheria': 0,  # Placeholder, update with actual data
        'HIV/AIDS': 0,  # Placeholder, update with actual data
        'GDP': 0,  # Placeholder, update with actual data
        'Population': 0,  # Placeholder, update with actual data
        'thinness__1-19_years': 0,  # Placeholder, update with actual data
        'thinness_5-9_years': 0,  # Placeholder, update with actual data
        'Income_composition_of_resources': 0,  # Placeholder, update with actual data
        'Schooling': 0  # Placeholder, update with actual data
    }
    
    features = {key: features[key] for key in feature_names}
    return pd.DataFrame([features])

def estimate_life_expectancy(age_category, gender, model_with_features):
    age_mapping = {
    '(0-2)': 1.5,
    '(4-6)': 5,
    '(8-12)': 10,
    '(14-20)': 17,
    '(21-24)': 22.5,
    '(25-32)': 28.5,
    '(33-37)': 35,
    '(38-43)': 40.5,
    '(44-47)': 45.5,
    '(48-53)': 50.5,
    '(54-59)': 57,
    '(60-70)': 65,
    '(71-80)': 75.5,
    '(81-90)': 85.5,
    '(91-99)': 95
}
    
    age_numeric = age_mapping.get(age_category, None)
    if age_numeric is None:
        raise ValueError(f"Age category '{age_category}' not found in mapping.")
    
    features = process_features(age_numeric, gender, model_with_features.feature_names)
    predicted_life_expectancy = model_with_features.model.predict(features)
    
    return predicted_life_expectancy[0]

def main():
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(14-20)', '(21-24)', '(25-32)', '(33-37)', '(38-43)', '(44-47)', '(48-53)', '(54-59)', '(60-70)', '(71-80)', '(81-90)', '(91-99)']
    genderList = ['Male', 'Female']
    padding = 20

    faceNet, ageNet, genderNet = load_models()
    model_with_features = joblib.load('life_expectancy_model_with_features.pkl')

    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        ret, frame = video.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        frame, bbox = faceBox(faceNet, frame)

        for bb in bbox:
            face = frame[max(0, bb[1] - padding):min(bb[3] + padding, frame.shape[0] - 1), 
                         max(0, bb[0] - padding):min(bb[2] + padding, frame.shape[1] - 1)]

            # Debug print
            print(f"Processing face with shape {face.shape}")

            gender, age_category = predict_age_gender(face, ageNet, genderNet, ageList, genderList)
            remaining_years = estimate_life_expectancy(age_category, gender, model_with_features)

            # Draw face box
            cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)

            # Display gender and age inside the face box
            label1 = f"Gender: {gender}, Age: {age_category}"
            cv2.putText(frame, label1, (bb[0], bb[1] - 10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)

            # Display life expectancy below gender and age
            label2 = f"Life Expectancy: {remaining_years:.2f} years"
            text_size2, _ = cv2.getTextSize(label2, cv2.FONT_HERSHEY_COMPLEX, 0.6, 1)
            text_x2 = bb[0] + (bb[2] - bb[0]) // 2 - text_size2[0] // 2
            text_y2 = bb[3] + 20
            cv2.putText(frame, label2, (text_x2, text_y2), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)

        cv2.imshow("Age-Gender Prediction and Life Expectancy Estimation", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or cv2.getWindowProperty("Age-Gender Prediction and Life Expectancy Estimation", cv2.WND_PROP_VISIBLE) < 1:
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
