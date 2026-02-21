from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class ModelPredictor:
    """Enhanced model predictor with proper error handling and validation"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.metadata = None
        self.load_model()
    
    def load_model(self):
        """Load the best trained model with metadata"""
        try:
            # Try to load the new best model first
            if Path("models/best_house_price_model.pkl").exists():
                self.model = joblib.load("models/best_house_price_model.pkl")
                self.metadata = joblib.load("models/model_metadata.pkl")
                
                if self.metadata['uses_scaling']:
                    self.scaler = joblib.load("models/scaler.pkl")
                
                logger.info(f"Loaded best model: {self.metadata['model_name']}")
                logger.info(f"Model R²: {self.metadata['metrics']['test_r2']:.4f}")
                
            # Fallback to old model if new one doesn't exist
            elif Path("models/house_price_model.pkl").exists():
                self.model = joblib.load("models/house_price_model.pkl")
                # Create minimal metadata for old model
                self.metadata = {
                    'model_name': 'linear_regression',
                    'feature_columns': ['area_sqft', 'bedrooms'],
                    'uses_scaling': False
                }
                logger.warning("Using legacy model with limited features")
            else:
                raise FileNotFoundError("No trained model found")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def validate_input(self, area, bedrooms, bathrooms=None, floors=None, 
                      age_years=None, distance_city_km=None, nearby_schools=None,
                      parking_spaces=None, crime_rate=None, location_score=None):
        """Validate and sanitize input data"""
        errors = []
        
        # Basic validation
        if area <= 0:
            errors.append("Area must be positive")
        if area > 10000:  # Reasonable upper limit
            errors.append("Area seems unreasonably large (>10,000 sqft)")
            
        if bedrooms < 0:
            errors.append("Bedrooms cannot be negative")
        if bedrooms > 20:  # Reasonable upper limit
            errors.append("Number of bedrooms seems unreasonable (>20)")
        
        # Additional validations for extended features
        if bathrooms is not None and bathrooms < 0:
            errors.append("Bathrooms cannot be negative")
        if floors is not None and floors < 1:
            errors.append("Floors must be at least 1")
        if age_years is not None and (age_years < 0 or age_years > 200):
            errors.append("Age must be between 0 and 200 years")
        if distance_city_km is not None and distance_city_km < 0:
            errors.append("Distance cannot be negative")
        if crime_rate is not None and crime_rate < 0:
            errors.append("Crime rate cannot be negative")
        if location_score is not None and (location_score < 1 or location_score > 10):
            errors.append("Location score must be between 1 and 10")
            
        return errors
    
    def prepare_features(self, area, bedrooms, bathrooms=None, floors=None,
                        age_years=None, distance_city_km=None, nearby_schools=None,
                        parking_spaces=None, crime_rate=None, location_score=None):
        """Prepare feature vector based on available model features"""
        
        if len(self.metadata['feature_columns']) == 2:
            # Legacy model with only 2 features
            features = [area, bedrooms]
        else:
            # Enhanced model with all features
            # Set defaults for missing values
            feature_values = {
                'area_sqft': area,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms or 2,  # Default values
                'floors': floors or 1,
                'age_years': age_years or 10,
                'distance_city_km': distance_city_km or 8,
                'nearby_schools': nearby_schools or 2,
                'parking_spaces': parking_spaces or 1,
                'crime_rate': crime_rate or 3.5,
                'location_score': location_score or 7
            }
            
            # Create engineered features
            feature_values['room_ratio'] = feature_values['bedrooms'] / feature_values['bathrooms']
            feature_values['age_location_score'] = feature_values['location_score'] / (feature_values['age_years'] + 1)
            
            # Extract features in correct order
            features = [feature_values[col] for col in self.metadata['feature_columns']]
        
        return np.array(features).reshape(1, -1)
    
    def predict(self, area, bedrooms, **kwargs):
        """Make prediction with proper preprocessing"""
        try:
            # Validate inputs
            validation_errors = self.validate_input(area, bedrooms, **kwargs)
            if validation_errors:
                return {'error': '; '.join(validation_errors)}
            
            # Prepare features
            features = self.prepare_features(area, bedrooms, **kwargs)
            
            # Apply scaling if needed
            if self.metadata['uses_scaling'] and self.scaler:
                features = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.model.predict(features)[0]
            
            # Ensure prediction is reasonable
            if prediction < 0:
                logger.warning("Model predicted negative price, setting to minimum")
                prediction = 1000000  # Minimum reasonable price
            
            return {
                'prediction': prediction,
                'model_used': self.metadata['model_name'],
                'features_used': len(self.metadata['feature_columns'])
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {'error': f'Prediction failed: {str(e)}'}

# Initialize predictor
try:
    predictor = ModelPredictor()
except Exception as e:
    logger.error(f"Failed to initialize predictor: {e}")
    predictor = None

@app.route("/")
def home():
    """Render home page with model info"""
    model_info = None
    if predictor and predictor.metadata:
        model_info = {
            'model_name': predictor.metadata['model_name'],
            'features_count': len(predictor.metadata['feature_columns']),
            'uses_scaling': predictor.metadata['uses_scaling']
        }
        if 'metrics' in predictor.metadata:
            model_info['r2_score'] = f"{predictor.metadata['metrics']['test_r2']:.3f}"
    
    return render_template("index.html", model_info=model_info)

@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction requests with enhanced error handling"""
    if not predictor:
        return render_template("index.html", 
                             prediction_text="❌ Model not available. Please train the model first.",
                             error=True)
    
    try:
        # Get basic inputs
        area = float(request.form.get("area", 0))
        bedrooms = int(request.form.get("bedrooms", 0))
        
        # Get optional extended inputs
        optional_inputs = {}
        for field in ['bathrooms', 'floors', 'age_years', 'distance_city_km', 
                     'nearby_schools', 'parking_spaces', 'crime_rate', 'location_score']:
            value = request.form.get(field)
            if value and value.strip():
                try:
                    optional_inputs[field] = float(value) if '.' in value else int(value)
                except ValueError:
                    pass  # Skip invalid values
        
        # Make prediction
        result = predictor.predict(area, bedrooms, **optional_inputs)
        
        if 'error' in result:
            return render_template("index.html", 
                                 prediction_text=f"❌ Error: {result['error']}", 
                                 error=True)
        
        # Format successful prediction
        prediction_text = f"🏠 Predicted House Price: ₹{result['prediction']:,.0f}"
        model_info_text = f"Model: {result['model_used']} | Features: {result['features_used']}"
        
        return render_template("index.html", 
                             prediction_text=prediction_text,
                             model_info_text=model_info_text,
                             success=True)
        
    except ValueError as e:
        return render_template("index.html", 
                             prediction_text="❌ Please enter valid numbers for all fields", 
                             error=True)
    except Exception as e:
        logger.error(f"Prediction route error: {e}")
        return render_template("index.html", 
                             prediction_text="❌ An unexpected error occurred", 
                             error=True)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API endpoint for programmatic access"""
    if not predictor:
        return jsonify({'error': 'Model not available'}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        area = data.get('area')
        bedrooms = data.get('bedrooms')
        
        if area is None or bedrooms is None:
            return jsonify({'error': 'area and bedrooms are required'}), 400
        
        # Get optional parameters
        optional_params = {k: v for k, v in data.items() 
                          if k not in ['area', 'bedrooms'] and v is not None}
        
        result = predictor.predict(float(area), int(bedrooms), **optional_params)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API prediction error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route("/health")
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy' if predictor else 'unhealthy',
        'model_loaded': predictor is not None,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    if predictor and predictor.metadata:
        status['model_info'] = {
            'name': predictor.metadata['model_name'],
            'features': len(predictor.metadata['feature_columns']),
            'scaling': predictor.metadata['uses_scaling']
        }
    
    return jsonify(status)

@app.errorhandler(404)
def not_found(error):
    return render_template("index.html", 
                         prediction_text="❌ Page not found", 
                         error=True), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template("index.html", 
                         prediction_text="❌ Internal server error", 
                         error=True), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
