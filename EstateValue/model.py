import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HousePricePredictor:
    """Enhanced House Price Prediction Model with proper ML practices"""
    
    def __init__(self, data_path="data/house_pricing.csv"):
        self.data_path = data_path
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.best_model = None
        self.model_metrics = {}
        
    def load_and_explore_data(self):
        """Load data with comprehensive EDA and validation"""
        logger.info("Loading and exploring dataset...")
        
        # Load dataset
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Dataset shape: {self.df.shape}")
        
        # Basic data validation
        if len(self.df) < 50:
            logger.warning(f"Very small dataset ({len(self.df)} samples). Consider collecting more data.")
        
        # Check for missing values
        missing_data = self.df.isnull().sum()
        if missing_data.any():
            logger.warning(f"Missing values found:\n{missing_data[missing_data > 0]}")
        
        # Display basic statistics
        print("\n=== Dataset Info ===")
        print(self.df.info())
        print("\n=== Statistical Summary ===")
        print(self.df.describe())
        
        # Check for outliers using IQR method
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = self.df[(self.df[col] < Q1 - 1.5*IQR) | (self.df[col] > Q3 + 1.5*IQR)]
            if len(outliers) > 0:
                logger.info(f"Found {len(outliers)} outliers in {col}")
        
        return self.df
    
    def feature_engineering(self):
        """Enhanced feature engineering using all available features"""
        logger.info("Performing feature engineering...")
        
        # Use all relevant features instead of just 2
        feature_cols = ['area_sqft', 'bedrooms', 'bathrooms', 'floors', 'age_years', 
                       'distance_city_km', 'nearby_schools', 'parking_spaces', 
                       'crime_rate', 'location_score']
        
        # Validate features exist
        available_features = [col for col in feature_cols if col in self.df.columns]
        if len(available_features) != len(feature_cols):
            missing = set(feature_cols) - set(available_features)
            logger.warning(f"Missing features: {missing}")
        
        self.feature_columns = available_features
        
        # Create additional engineered features
        self.df['price_per_sqft'] = self.df['price'] / self.df['area_sqft']
        self.df['room_ratio'] = self.df['bedrooms'] / self.df['bathrooms']
        self.df['age_location_score'] = self.df['location_score'] / (self.df['age_years'] + 1)
        
        # Add engineered features to feature list
        engineered_features = ['room_ratio', 'age_location_score']
        self.feature_columns.extend(engineered_features)
        
        # Handle any infinite or NaN values from feature engineering
        self.df = self.df.replace([np.inf, -np.inf], np.nan)
        self.df = self.df.fillna(self.df.median())
        
        logger.info(f"Using {len(self.feature_columns)} features: {self.feature_columns}")
        
    def prepare_data(self):
        """Prepare training and testing data with proper validation"""
        logger.info("Preparing training and testing data...")
        
        X = self.df[self.feature_columns]
        y = self.df['price']
        
        # Validate target variable
        if y.isnull().any():
            logger.error("Target variable contains null values")
            raise ValueError("Target variable contains null values")
        
        # Split data with stratification consideration for small datasets
        test_size = 0.2 if len(self.df) > 20 else 0.1
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=True
        )
        
        logger.info(f"Training set: {self.X_train.shape}, Test set: {self.X_test.shape}")
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
    def train_models(self):
        """Train multiple models with hyperparameter tuning"""
        logger.info("Training multiple models with hyperparameter optimization...")
        
        # Define models with hyperparameter grids
        model_configs = {
            'linear': {
                'model': LinearRegression(),
                'params': {},
                'scaled': False
            },
            'ridge': {
                'model': Ridge(),
                'params': {'alpha': [0.1, 1.0, 10.0, 100.0]},
                'scaled': True
            },
            'lasso': {
                'model': Lasso(max_iter=2000),
                'params': {'alpha': [0.01, 0.1, 1.0, 10.0]},
                'scaled': True
            },
            'random_forest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100],
                    'max_depth': [None, 5, 10],
                    'min_samples_split': [2, 5]
                },
                'scaled': False
            }
        }
        
        best_score = -np.inf
        
        for name, config in model_configs.items():
            logger.info(f"Training {name} model...")
            
            # Select appropriate data (scaled or unscaled)
            X_train_data = self.X_train_scaled if config['scaled'] else self.X_train
            X_test_data = self.X_test_scaled if config['scaled'] else self.X_test
            
            if config['params'] and len(self.X_train) >= 5:
                # Hyperparameter tuning with GridSearchCV (only if enough data)
                cv_folds = min(3, len(self.X_train))  # Reduce CV folds for small datasets
                grid_search = GridSearchCV(
                    config['model'], 
                    config['params'], 
                    cv=cv_folds, 
                    scoring='r2',
                    n_jobs=-1
                )
                grid_search.fit(X_train_data, self.y_train)
                model = grid_search.best_estimator_
                logger.info(f"Best params for {name}: {grid_search.best_params_}")
            else:
                model = config['model']
                model.fit(X_train_data, self.y_train)
            
            # Evaluate model
            train_pred = model.predict(X_train_data)
            test_pred = model.predict(X_test_data)
            
            # Calculate metrics
            train_r2 = r2_score(self.y_train, train_pred)
            test_r2 = r2_score(self.y_test, test_pred) if len(self.y_test) > 0 else 0
            
            metrics = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': np.sqrt(mean_squared_error(self.y_train, train_pred)),
                'test_rmse': np.sqrt(mean_squared_error(self.y_test, test_pred)) if len(self.y_test) > 0 else 0,
                'train_mae': mean_absolute_error(self.y_train, train_pred),
                'test_mae': mean_absolute_error(self.y_test, test_pred) if len(self.y_test) > 0 else 0
            }
            
            # Cross-validation score (only if enough data)
            if len(self.X_train) >= 3:
                cv_folds = min(3, len(self.X_train))
                try:
                    cv_scores = cross_val_score(model, X_train_data, self.y_train, 
                                              cv=cv_folds, scoring='r2')
                    metrics['cv_r2_mean'] = cv_scores.mean()
                    metrics['cv_r2_std'] = cv_scores.std()
                except:
                    # Fallback if CV fails
                    metrics['cv_r2_mean'] = train_r2
                    metrics['cv_r2_std'] = 0
            else:
                # Use training R² as fallback for very small datasets
                metrics['cv_r2_mean'] = train_r2
                metrics['cv_r2_std'] = 0
            
            self.models[name] = {
                'model': model,
                'scaled': config['scaled'],
                'metrics': metrics
            }
            self.model_metrics[name] = metrics
            
            # Track best model based on cross-validation R² or training R²
            score = metrics['cv_r2_mean'] if not np.isnan(metrics['cv_r2_mean']) else metrics['train_r2']
            if score > best_score:
                best_score = score
                self.best_model = name
        
        logger.info(f"Best model: {self.best_model} (Score: {best_score:.4f})")
    
    def evaluate_models(self):
        """Comprehensive model evaluation and comparison"""
        logger.info("Evaluating all trained models...")
        
        print("\n" + "="*80)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*80)
        
        for name, model_info in self.models.items():
            metrics = model_info['metrics']
            print(f"\n{name.upper()} MODEL:")
            print(f"  Training R²:     {metrics['train_r2']:.4f}")
            
            if len(self.y_test) > 0:
                print(f"  Test R²:         {metrics['test_r2']:.4f}")
                print(f"  Test RMSE:       ₹{metrics['test_rmse']:,.0f}")
                print(f"  Test MAE:        ₹{metrics['test_mae']:,.0f}")
            else:
                print(f"  Test R²:         N/A (no test data)")
                print(f"  Test RMSE:       N/A")
                print(f"  Test MAE:        N/A")
            
            if not np.isnan(metrics['cv_r2_mean']):
                print(f"  CV R² (mean±std): {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")
            else:
                print(f"  CV R²:           N/A (insufficient data)")
            
            # Check for overfitting (only if we have test data)
            if len(self.y_test) > 0:
                overfitting = metrics['train_r2'] - metrics['test_r2']
                if overfitting > 0.1:
                    print(f"  ⚠️  Potential overfitting detected (diff: {overfitting:.3f})")
        
        if self.best_model:
            print(f"\n🏆 BEST MODEL: {self.best_model.upper()}")
        else:
            print(f"\n⚠️ No best model selected")
            # Select first model as fallback
            self.best_model = list(self.models.keys())[0]
            print(f"Using {self.best_model} as fallback")
        
    def save_best_model(self):
        """Save the best performing model and scaler"""
        logger.info(f"Saving best model: {self.best_model}")
        
        best_model_info = self.models[self.best_model]
        
        # Create models directory if it doesn't exist
        Path("models").mkdir(exist_ok=True)
        
        # Save model
        joblib.dump(best_model_info['model'], "models/best_house_price_model.pkl")
        
        # Save scaler if model uses scaling
        if best_model_info['scaled']:
            joblib.dump(self.scaler, "models/scaler.pkl")
        
        # Save feature columns and model metadata
        model_metadata = {
            'model_name': self.best_model,
            'feature_columns': self.feature_columns,
            'uses_scaling': best_model_info['scaled'],
            'metrics': best_model_info['metrics']
        }
        joblib.dump(model_metadata, "models/model_metadata.pkl")
        
        logger.info("Model saved successfully!")
        
    def plot_results(self):
        """Create visualization plots for model comparison"""
        logger.info("Creating visualization plots...")
        
        # Model comparison plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # R² comparison
        models = list(self.model_metrics.keys())
        test_r2 = [self.model_metrics[m]['test_r2'] for m in models]
        cv_r2 = [self.model_metrics[m]['cv_r2_mean'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        ax1.bar(x - width/2, test_r2, width, label='Test R²', alpha=0.8)
        ax1.bar(x + width/2, cv_r2, width, label='CV R²', alpha=0.8)
        ax1.set_xlabel('Models')
        ax1.set_ylabel('R² Score')
        ax1.set_title('Model R² Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # RMSE comparison
        test_rmse = [self.model_metrics[m]['test_rmse'] for m in models]
        ax2.bar(models, test_rmse, alpha=0.8, color='orange')
        ax2.set_xlabel('Models')
        ax2.set_ylabel('RMSE (₹)')
        ax2.set_title('Model RMSE Comparison')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Feature importance (for best model if available)
        best_model_obj = self.models[self.best_model]['model']
        if hasattr(best_model_obj, 'coef_'):
            importance = np.abs(best_model_obj.coef_)
            ax3.barh(self.feature_columns, importance)
            ax3.set_xlabel('Absolute Coefficient Value')
            ax3.set_title(f'Feature Importance ({self.best_model})')
            ax3.grid(True, alpha=0.3)
        elif hasattr(best_model_obj, 'feature_importances_'):
            importance = best_model_obj.feature_importances_
            ax3.barh(self.feature_columns, importance)
            ax3.set_xlabel('Feature Importance')
            ax3.set_title(f'Feature Importance ({self.best_model})')
            ax3.grid(True, alpha=0.3)
        
        # Prediction vs Actual for best model
        best_model_info = self.models[self.best_model]
        X_test_data = self.X_test_scaled if best_model_info['scaled'] else self.X_test
        y_pred = best_model_info['model'].predict(X_test_data)
        
        ax4.scatter(self.y_test, y_pred, alpha=0.7)
        ax4.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        ax4.set_xlabel('Actual Price (₹)')
        ax4.set_ylabel('Predicted Price (₹)')
        ax4.set_title(f'Actual vs Predicted ({self.best_model})')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('models/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main training pipeline"""
    predictor = HousePricePredictor()
    
    # Execute full ML pipeline
    predictor.load_and_explore_data()
    predictor.feature_engineering()
    predictor.prepare_data()
    predictor.train_models()
    predictor.evaluate_models()
    predictor.save_best_model()
    predictor.plot_results()
    
    print("\n✅ Model training completed successfully!")
    print("📁 Best model saved to: models/best_house_price_model.pkl")

if __name__ == "__main__":
    main()
