"""
Data utilities for EstateValue project
Includes data validation, augmentation, and quality checks
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from config import DATA_FILE, FEATURE_COLUMNS, INPUT_VALIDATION

logger = logging.getLogger(__name__)

class DataValidator:
    """Validate and clean house pricing data"""
    
    def __init__(self, data_path=DATA_FILE):
        self.data_path = data_path
        
    def load_data(self):
        """Load and perform basic validation"""
        try:
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Loaded {len(self.df)} samples from {self.data_path}")
            return self.df
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def validate_data_quality(self):
        """Comprehensive data quality validation"""
        issues = []
        
        # Check for missing values
        missing = self.df.isnull().sum()
        if missing.any():
            issues.append(f"Missing values: {dict(missing[missing > 0])}")
        
        # Check for duplicates
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            issues.append(f"Duplicate rows: {duplicates}")
        
        # Check data types
        for col in self.df.columns:
            if col != 'price' and not pd.api.types.is_numeric_dtype(self.df[col]):
                issues.append(f"Non-numeric column: {col}")
        
        # Check value ranges
        for col, limits in INPUT_VALIDATION.items():
            if col in self.df.columns:
                out_of_range = (
                    (self.df[col] < limits['min']) | 
                    (self.df[col] > limits['max'])
                ).sum()
                if out_of_range > 0:
                    issues.append(f"{col}: {out_of_range} values out of range [{limits['min']}, {limits['max']}]")
        
        # Check for negative prices
        if 'price' in self.df.columns:
            negative_prices = (self.df['price'] <= 0).sum()
            if negative_prices > 0:
                issues.append(f"Negative or zero prices: {negative_prices}")
        
        return issues
    
    def clean_data(self):
        """Clean and fix data issues"""
        logger.info("Cleaning data...")
        
        # Remove duplicates
        initial_size = len(self.df)
        self.df = self.df.drop_duplicates()
        removed_duplicates = initial_size - len(self.df)
        if removed_duplicates > 0:
            logger.info(f"Removed {removed_duplicates} duplicate rows")
        
        # Handle missing values
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        
        # Remove rows with invalid prices
        if 'price' in self.df.columns:
            valid_price_mask = self.df['price'] > 0
            invalid_prices = (~valid_price_mask).sum()
            if invalid_prices > 0:
                logger.warning(f"Removing {invalid_prices} rows with invalid prices")
                self.df = self.df[valid_price_mask]
        
        # Cap outliers using IQR method
        for col in numeric_cols:
            if col != 'price':  # Don't cap target variable
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_before = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
                self.df[col] = self.df[col].clip(lower_bound, upper_bound)
                
                if outliers_before > 0:
                    logger.info(f"Capped {outliers_before} outliers in {col}")
        
        return self.df
    
    def generate_synthetic_data(self, target_size=100):
        """Generate synthetic data to augment small datasets"""
        if len(self.df) >= target_size:
            logger.info("Dataset already large enough, skipping synthetic data generation")
            return self.df
        
        logger.info(f"Generating synthetic data to reach {target_size} samples...")
        
        # Calculate how many synthetic samples to generate
        samples_needed = target_size - len(self.df)
        
        # Use existing data statistics to generate realistic synthetic data
        synthetic_data = []
        
        for _ in range(samples_needed):
            # Randomly select a base sample
            base_idx = np.random.randint(0, len(self.df))
            base_sample = self.df.iloc[base_idx].copy()
            
            # Add controlled noise to create variation
            for col in self.df.columns:
                if col != 'price' and pd.api.types.is_numeric_dtype(self.df[col]):
                    # Add noise proportional to column standard deviation
                    noise_factor = 0.1  # 10% noise
                    std_dev = self.df[col].std()
                    noise = np.random.normal(0, std_dev * noise_factor)
                    base_sample[col] = max(0, base_sample[col] + noise)
            
            # Recalculate price based on a simple relationship
            # This is a simplified model - in practice, you'd want more sophisticated pricing
            area_factor = base_sample['area_sqft'] * 4000  # Base price per sqft
            bedroom_factor = base_sample['bedrooms'] * 500000  # Additional per bedroom
            location_factor = base_sample.get('location_score', 7) * 200000
            age_penalty = base_sample.get('age_years', 10) * 10000
            
            estimated_price = area_factor + bedroom_factor + location_factor - age_penalty
            # Add some randomness
            price_noise = np.random.normal(0, estimated_price * 0.1)
            base_sample['price'] = max(1000000, estimated_price + price_noise)  # Minimum 10 lakh
            
            synthetic_data.append(base_sample)
        
        # Combine original and synthetic data
        synthetic_df = pd.DataFrame(synthetic_data)
        augmented_df = pd.concat([self.df, synthetic_df], ignore_index=True)
        
        logger.info(f"Generated {samples_needed} synthetic samples")
        logger.info(f"Total dataset size: {len(augmented_df)}")
        
        return augmented_df
    
    def save_cleaned_data(self, output_path=None):
        """Save cleaned data to file"""
        if output_path is None:
            output_path = self.data_path.parent / f"cleaned_{self.data_path.name}"
        
        self.df.to_csv(output_path, index=False)
        logger.info(f"Saved cleaned data to {output_path}")
        return output_path

def main():
    """Data validation and cleaning pipeline"""
    logging.basicConfig(level=logging.INFO)
    
    validator = DataValidator()
    
    # Load and validate data
    df = validator.load_data()
    print(f"📊 Dataset shape: {df.shape}")
    
    # Check data quality
    issues = validator.validate_data_quality()
    if issues:
        print("\n⚠️ Data Quality Issues Found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ No data quality issues found")
    
    # Clean data
    cleaned_df = validator.clean_data()
    print(f"📊 Cleaned dataset shape: {cleaned_df.shape}")
    
    # Generate synthetic data if needed
    if len(cleaned_df) < 50:
        print(f"\n🔄 Dataset too small ({len(cleaned_df)} samples), generating synthetic data...")
        augmented_df = validator.generate_synthetic_data(target_size=100)
        validator.df = augmented_df
        print(f"📊 Augmented dataset shape: {augmented_df.shape}")
    
    # Save cleaned data
    output_path = validator.save_cleaned_data()
    print(f"\n💾 Cleaned data saved to: {output_path}")
    
    print("\n✅ Data validation and cleaning completed!")

if __name__ == "__main__":
    main()