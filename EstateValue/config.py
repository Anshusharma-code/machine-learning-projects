"""
Configuration settings for EstateValue ML model
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
STATIC_DIR = PROJECT_ROOT / "static"
TEMPLATES_DIR = PROJECT_ROOT / "templates"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)

# Data settings
DATA_FILE = DATA_DIR / "house_pricing.csv"
MIN_DATASET_SIZE = 20  # Minimum samples for reliable training

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Feature engineering settings
FEATURE_COLUMNS = [
    'area_sqft', 'bedrooms', 'bathrooms', 'floors', 'age_years',
    'distance_city_km', 'nearby_schools', 'parking_spaces', 
    'crime_rate', 'location_score'
]

ENGINEERED_FEATURES = [
    'room_ratio',  # bedrooms / bathrooms
    'age_location_score'  # location_score / (age_years + 1)
]

# Model hyperparameters
MODEL_CONFIGS = {
    'linear': {
        'params': {},
        'scaled': False
    },
    'ridge': {
        'params': {'alpha': [0.1, 1.0, 10.0, 100.0]},
        'scaled': True
    },
    'lasso': {
        'params': {'alpha': [0.01, 0.1, 1.0, 10.0]},
        'scaled': True
    },
    'random_forest': {
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10]
        },
        'scaled': False
    }
}

# Validation settings
INPUT_VALIDATION = {
    'area_sqft': {'min': 100, 'max': 10000},
    'bedrooms': {'min': 1, 'max': 20},
    'bathrooms': {'min': 1, 'max': 10},
    'floors': {'min': 1, 'max': 5},
    'age_years': {'min': 0, 'max': 200},
    'distance_city_km': {'min': 0, 'max': 100},
    'nearby_schools': {'min': 0, 'max': 20},
    'parking_spaces': {'min': 0, 'max': 10},
    'crime_rate': {'min': 0, 'max': 10},
    'location_score': {'min': 1, 'max': 10}
}

# Flask settings
FLASK_CONFIG = {
    'DEBUG': True,
    'HOST': '0.0.0.0',
    'PORT': 5000
}

# Logging settings
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}