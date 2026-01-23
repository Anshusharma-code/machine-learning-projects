"""
Flask Web Application for Naive Bayes Student Performance Prediction
This app allows users to input student data and get pass/fail predictions
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import os

app = Flask(__name__)

class StudentPredictor:
    """
    A class to handle the Naive Bayes model for student performance prediction
    """
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.load_and_train_model()
    
    def load_and_train_model(self):
        """
        Load the dataset and train the Naive Bayes model
        """
        try:
            # Load the dataset
            data_path = os.path.join(os.path.dirname(__file__), 'data.csv')
            df = pd.read_csv(data_path)
            
            # Encode the result column (Pass/Fail to 1/0)
            df["result"] = df["result"].map({"Fail": 0, "Pass": 1})
            
            # Define features and target
            X = df[["study_hours", "attendance", "previous_marks", "assignment_completion"]]
            y = df["result"]
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train the Naive Bayes model
            self.model = GaussianNB()
            self.model.fit(X_train, y_train)
            self.is_trained = True
            
            print("Model trained successfully!")
            
        except Exception as e:
            print(f"Error training model: {e}")
            self.is_trained = False
    
    def predict(self, study_hours, attendance, previous_marks, assignment_completion):
        """
        Make a prediction for a single student
        Returns: (prediction, probability)
        """
        if not self.is_trained:
            return None, None
        
        # Create a DataFrame with the input data
        student_data = pd.DataFrame([{
            "study_hours": study_hours,
            "attendance": attendance,
            "previous_marks": previous_marks,
            "assignment_completion": assignment_completion
        }])
        
        # Make prediction
        prediction = self.model.predict(student_data)[0]
        probability = self.model.predict_proba(student_data)[0]
        
        return prediction, probability

# Initialize the predictor
predictor = StudentPredictor()

@app.route('/')
def index():
    """
    Render the main page with the input form
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests from the frontend
    """
    try:
        # Get data from the request
        data = request.get_json()
        
        # Extract input values
        study_hours = float(data['study_hours'])
        attendance = float(data['attendance'])
        previous_marks = float(data['previous_marks'])
        assignment_completion = float(data['assignment_completion'])
        
        # Validate input ranges
        if not (0 <= study_hours <= 24):
            return jsonify({'error': 'Study hours must be between 0 and 24'}), 400
        if not (0 <= attendance <= 100):
            return jsonify({'error': 'Attendance must be between 0 and 100'}), 400
        if not (0 <= previous_marks <= 100):
            return jsonify({'error': 'Previous marks must be between 0 and 100'}), 400
        if not (0 <= assignment_completion <= 100):
            return jsonify({'error': 'Assignment completion must be between 0 and 100'}), 400
        
        # Make prediction
        prediction, probability = predictor.predict(
            study_hours, attendance, previous_marks, assignment_completion
        )
        
        if prediction is None:
            return jsonify({'error': 'Model not available'}), 500
        
        # Format the response
        result = {
            'prediction': 'Pass' if prediction == 1 else 'Fail',
            'probability_pass': round(probability[1] * 100, 2),
            'probability_fail': round(probability[0] * 100, 2),
            'confidence': round(max(probability) * 100, 2)
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/result')
def result():
    """
    Render the result page (for future use if needed)
    """
    return render_template('result.html')

if __name__ == '__main__':
    print("Starting Naive Bayes Student Predictor Web App...")
    print("Visit http://localhost:5000 to use the application")
    app.run(debug=True, host='0.0.0.0', port=5000)