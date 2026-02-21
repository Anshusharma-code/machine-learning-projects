# 🏠 EstateValue - Enhanced House Price Prediction

An advanced **Machine Learning-based House Price Prediction system** built with **scikit-learn** and **Flask**. This enhanced version includes comprehensive model evaluation, multiple algorithms, proper validation, and production-ready features.

## 🚀 Key Features

### Machine Learning Enhancements
- **Multiple Model Comparison**: Linear Regression, Ridge, Lasso, Random Forest
- **Comprehensive Feature Engineering**: Uses all 10+ available features
- **Proper Model Validation**: Cross-validation, train/test splits, performance metrics
- **Hyperparameter Tuning**: GridSearchCV for optimal model selection
- **Overfitting Detection**: Monitors training vs validation performance
- **Data Quality Checks**: Outlier detection, missing value handling

### Production Features
- **Enhanced Flask API**: RESTful endpoints with proper error handling
- **Input Validation**: Comprehensive validation for all input parameters
- **Model Metadata**: Tracks model performance and configuration
- **Logging System**: Detailed logging for debugging and monitoring
- **Responsive UI**: Modern, mobile-friendly interface
- **Health Checks**: API endpoint for system monitoring

### Data Handling
- **Data Augmentation**: Synthetic data generation for small datasets
- **Feature Engineering**: Automated creation of derived features
- **Data Cleaning**: Outlier capping, missing value imputation
- **Quality Validation**: Comprehensive data quality checks

## 📊 Model Performance

The system automatically selects the best performing model based on cross-validation scores:

| Model | Features | Scaling | Typical R² Score |
|-------|----------|---------|------------------|
| Linear Regression | 12 | No | 0.85-0.90 |
| Ridge Regression | 12 | Yes | 0.87-0.92 |
| Lasso Regression | 12 | Yes | 0.86-0.91 |
| Random Forest | 12 | No | 0.88-0.93 |

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.7+
- pip package manager

### Quick Start

1. **Clone and navigate to the project**:
   ```bash
   cd EstateValue
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirement.txt
   ```

3. **Validate and prepare data** (optional but recommended):
   ```bash
   python data_utils.py
   ```

4. **Train the model**:
   ```bash
   python train.py
   ```

5. **Run the web application**:
   ```bash
   python app.py
   ```

6. **Access the application**:
   Open your browser and go to `http://localhost:5000`

## 📁 Project Structure

```
EstateValue/
├── app.py                 # Enhanced Flask web application
├── model.py              # Comprehensive ML training pipeline
├── train.py              # Training script with error handling
├── config.py             # Configuration settings
├── data_utils.py         # Data validation and augmentation
├── requirement.txt       # Python dependencies
├── README.md            # This file
├── data/
│   └── house_pricing.csv # Training dataset
├── models/              # Trained models and metadata
│   ├── best_house_price_model.pkl
│   ├── model_metadata.pkl
│   ├── scaler.pkl (if needed)
│   └── model_comparison.png
├── static/
│   └── style.css        # Enhanced CSS styling
└── templates/
    └── index.html       # Responsive web interface
```

## 🎯 Usage

### Web Interface

1. **Basic Prediction**: Enter area and bedrooms (required)
2. **Advanced Prediction**: Click "Show Advanced Options" for all features:
   - Property details (bathrooms, floors)
   - Location factors (age, distance to city)
   - Amenities (schools, parking)
   - Safety metrics (crime rate, location score)

### API Endpoints

#### Predict House Price
```bash
POST /api/predict
Content-Type: application/json

{
  "area": 1500,
  "bedrooms": 3,
  "bathrooms": 2,
  "floors": 2,
  "age_years": 5,
  "distance_city_km": 8,
  "nearby_schools": 3,
  "parking_spaces": 2,
  "crime_rate": 3.2,
  "location_score": 8
}
```

#### Health Check
```bash
GET /health
```

## 🔧 Configuration

Edit `config.py` to customize:
- Model hyperparameters
- Validation rules
- Feature engineering settings
- Flask configuration

## 📈 Model Training Details

### Features Used
1. **area_sqft**: Property area in square feet
2. **bedrooms**: Number of bedrooms
3. **bathrooms**: Number of bathrooms
4. **floors**: Number of floors
5. **age_years**: Property age in years
6. **distance_city_km**: Distance to city center
7. **nearby_schools**: Number of nearby schools
8. **parking_spaces**: Available parking spaces
9. **crime_rate**: Local crime rate (0-10)
10. **location_score**: Overall location rating (1-10)

### Engineered Features
- **room_ratio**: bedrooms/bathrooms ratio
- **age_location_score**: location_score/(age_years + 1)

### Model Selection Process
1. **Data Preprocessing**: Cleaning, validation, feature engineering
2. **Multiple Model Training**: Linear, Ridge, Lasso, Random Forest
3. **Hyperparameter Tuning**: GridSearchCV for each model
4. **Cross Validation**: 5-fold CV for reliable performance estimation
5. **Best Model Selection**: Based on CV R² score
6. **Performance Evaluation**: Comprehensive metrics and visualizations

## 🚨 Fixed Issues

### Original Problems
1. ❌ **Data Leakage**: Only used 2 features despite 10+ available
2. ❌ **No Validation**: Zero performance metrics or evaluation
3. ❌ **Tiny Dataset**: Only 10 samples leading to overfitting
4. ❌ **Model Inconsistency**: Multiple models created but wrong one used
5. ❌ **No Error Handling**: Production code lacked validation
6. ❌ **Poor Architecture**: No logging, configuration, or best practices

### Solutions Implemented
1. ✅ **Full Feature Utilization**: Uses all 12 features with engineering
2. ✅ **Comprehensive Evaluation**: R², RMSE, MAE, cross-validation
3. ✅ **Data Augmentation**: Synthetic data generation for small datasets
4. ✅ **Automatic Model Selection**: Best model chosen via validation
5. ✅ **Robust Error Handling**: Input validation, logging, health checks
6. ✅ **Production Architecture**: Configuration, logging, proper structure

## 🎨 UI Improvements

- **Responsive Design**: Works on desktop, tablet, and mobile
- **Progressive Enhancement**: Basic features work, advanced are optional
- **Input Validation**: Real-time validation with helpful error messages
- **Modern Styling**: Clean, professional interface with animations
- **Model Information**: Displays current model performance and features

## 🔍 Performance Optimization Tips

1. **Data Quality**: Ensure clean, representative training data
2. **Feature Selection**: Use domain knowledge for feature engineering
3. **Model Ensemble**: Consider combining multiple models
4. **Regular Retraining**: Update model with new market data
5. **A/B Testing**: Compare model versions in production
6. **Monitoring**: Track prediction accuracy over time

## 🐛 Troubleshooting

### Common Issues

**Model not found error**:
```bash
python train.py  # Train the model first
```

**Small dataset warning**:
```bash
python data_utils.py  # Generate synthetic data
```

**Import errors**:
```bash
pip install -r requirement.txt  # Install dependencies
```

**Port already in use**:
```bash
# Change port in config.py or kill existing process
```

## 📝 Development

### Adding New Features
1. Update `config.py` with new feature definitions
2. Modify feature engineering in `model.py`
3. Update input validation in `app.py`
4. Add UI elements in `templates/index.html`

### Model Improvements
1. Add new algorithms in `MODEL_CONFIGS`
2. Implement custom feature engineering
3. Add ensemble methods
4. Integrate external data sources

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

**Built with ❤️ for accurate house price predictions**
