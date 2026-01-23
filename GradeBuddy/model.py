"""
Naive Bayes Model for Student Performance Prediction
This module contains the machine learning logic for predicting student pass/fail outcomes
"""

import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import pickle
import os

class NaiveBayesStudentPredictor:
    """
    A comprehensive Naive Bayes classifier for predicting student performance
    
    Features used for prediction:
    - study_hours: Hours spent studying per day
    - attendance: Attendance percentage
    - previous_marks: Previous exam marks (0-100)
    - assignment_completion: Assignment completion percentage
    """
    
    def __init__(self, data_file='data.csv'):
        self.model = GaussianNB()
        self.is_trained = False
        self.accuracy = 0
        self.data_file = data_file
        self.feature_names = ["study_hours", "attendance", "previous_marks", "assignment_completion"]
        
    def load_data(self):
        """
        Load and preprocess the dataset
        Returns: X (features), y (target)
        """
        try:
            # Load the dataset
            df = pd.read_csv(self.data_file)
            print("Dataset loaded successfully!")
            print(f"Dataset shape: {df.shape}")
            print("\nFirst 5 rows:")
            print(df.head())
            
            # Check the distribution of results
            print("\nResult distribution:")
            print(df["result"].value_counts())
            
            # Encode the result column (Pass/Fail to 1/0)
            df["result"] = df["result"].map({"Fail": 0, "Pass": 1})
            
            # Define features and target
            X = df[self.feature_names]
            y = df["result"]
            
            print(f"\nFeatures: {self.feature_names}")
            print(f"Target: Pass (1) / Fail (0)")
            
            return X, y
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None
    
    def train_model(self):
        """
        Train the Naive Bayes model with the loaded data
        """
        # Load the data
        X, y = self.load_data()
        
        if X is None or y is None:
            print("Failed to load data. Cannot train model.")
            return False
        
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTraining set size: {X_train.shape[0]}")
        print(f"Testing set size: {X_test.shape[0]}")
        
        # Train the Naive Bayes model
        self.model.fit(X_train, y_train)
        
        # Make predictions on the test set
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate accuracy
        self.accuracy = accuracy_score(y_test, y_pred)
        
        # Print model performance
        print(f"\n=== MODEL PERFORMANCE ===")
        print(f"Accuracy: {self.accuracy:.3f}")
        
        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Fail', 'Pass']))
        
        # ROC-AUC Score
        auc_score = roc_auc_score(y_test, y_prob)
        print(f"ROC-AUC Score: {auc_score:.3f}")
        
        self.is_trained = True
        print("\nModel trained successfully!")
        
        return True
    
    def predict_single_student(self, study_hours, attendance, previous_marks, assignment_completion):
        """
        Predict the outcome for a single student
        
        Args:
            study_hours (float): Hours spent studying per day
            attendance (float): Attendance percentage (0-100)
            previous_marks (float): Previous exam marks (0-100)
            assignment_completion (float): Assignment completion percentage (0-100)
        
        Returns:
            dict: Prediction results including probability
        """
        if not self.is_trained:
            return {"error": "Model is not trained yet"}
        
        # Create input data
        student_data = pd.DataFrame([{
            "study_hours": study_hours,
            "attendance": attendance,
            "previous_marks": previous_marks,
            "assignment_completion": assignment_completion
        }])
        
        # Make prediction
        prediction = self.model.predict(student_data)[0]
        probabilities = self.model.predict_proba(student_data)[0]
        
        result = {
            "prediction": "Pass" if prediction == 1 else "Fail",
            "probability_fail": round(probabilities[0] * 100, 2),
            "probability_pass": round(probabilities[1] * 100, 2),
            "confidence": round(max(probabilities) * 100, 2),
            "input_data": {
                "study_hours": study_hours,
                "attendance": attendance,
                "previous_marks": previous_marks,
                "assignment_completion": assignment_completion
            }
        }
        
        return result
    
    def save_model(self, filename='naive_bayes_model.pkl'):
        """
        Save the trained model to a file
        """
        if not self.is_trained:
            print("Model is not trained yet. Cannot save.")
            return False
        
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"Model saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, filename='naive_bayes_model.pkl'):
        """
        Load a pre-trained model from a file
        """
        try:
            with open(filename, 'rb') as f:
                self.model = pickle.load(f)
            self.is_trained = True
            print(f"Model loaded from {filename}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

def main():
    """
    Main function to demonstrate the Naive Bayes model
    """
    print("=== NAIVE BAYES STUDENT PERFORMANCE PREDICTOR ===\n")
    
    # Initialize the predictor
    predictor = NaiveBayesStudentPredictor()
    
    # Train the model
    if predictor.train_model():
        print("\n=== TESTING WITH SAMPLE STUDENTS ===")
        
        # Test with sample students
        test_students = [
            {"study_hours": 6, "attendance": 85, "previous_marks": 78, "assignment_completion": 90},
            {"study_hours": 2, "attendance": 45, "previous_marks": 38, "assignment_completion": 30},
            {"study_hours": 8, "attendance": 95, "previous_marks": 92, "assignment_completion": 98},
            {"study_hours": 1, "attendance": 30, "previous_marks": 25, "assignment_completion": 15}
        ]
        
        for i, student in enumerate(test_students, 1):
            print(f"\nStudent {i}:")
            print(f"  Study Hours: {student['study_hours']}")
            print(f"  Attendance: {student['attendance']}%")
            print(f"  Previous Marks: {student['previous_marks']}")
            print(f"  Assignment Completion: {student['assignment_completion']}%")
            
            result = predictor.predict_single_student(**student)
            print(f"  Prediction: {result['prediction']}")
            print(f"  Confidence: {result['confidence']}%")
            print(f"  Pass Probability: {result['probability_pass']}%")
        
        # Save the model
        predictor.save_model()

if __name__ == "__main__":
    main()





