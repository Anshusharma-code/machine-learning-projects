#!/usr/bin/env python3
"""
Enhanced training script for EstateValue house price prediction model
Run this script to train and evaluate the ML model with proper validation
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from model import HousePricePredictor
from config import *

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG['level']),
        format=LOGGING_CONFIG['format'],
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(MODELS_DIR / 'training.log')
        ]
    )

def main():
    """Main training pipeline with error handling"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting EstateValue model training...")
        
        # Check if data file exists
        if not DATA_FILE.exists():
            logger.error(f"Data file not found: {DATA_FILE}")
            print("❌ Error: house_pricing.csv not found in data/ directory")
            return 1
        
        # Initialize and run training pipeline
        predictor = HousePricePredictor(str(DATA_FILE))
        
        # Execute full ML pipeline
        df = predictor.load_and_explore_data()
        
        # Check dataset size
        if len(df) < MIN_DATASET_SIZE:
            logger.warning(f"Dataset very small ({len(df)} samples). Results may be unreliable.")
            response = input(f"Dataset has only {len(df)} samples. Continue? (y/N): ")
            if response.lower() != 'y':
                print("Training cancelled.")
                return 0
        
        predictor.feature_engineering()
        predictor.prepare_data()
        predictor.train_models()
        predictor.evaluate_models()
        predictor.save_best_model()
        predictor.plot_results()
        
        print("\n" + "="*60)
        print("✅ MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"📁 Best model saved to: {MODELS_DIR}/best_house_price_model.pkl")
        print(f"📊 Training plots saved to: {MODELS_DIR}/model_comparison.png")
        print(f"📝 Training log saved to: {MODELS_DIR}/training.log")
        print("\n🚀 You can now run the Flask app with: python app.py")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        print("\n⏹️ Training interrupted by user")
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        print(f"\n❌ Training failed: {e}")
        print("Check the log file for detailed error information.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)