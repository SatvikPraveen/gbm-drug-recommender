"""
Data Processing Module

Handles GDSC (Genomics of Drug Sensitivity in Cancer) data processing.

Functionality:
- Load drug screening data (IC50 values) from GDSC database
- Load cell line annotations and filter for GBM cell lines
- Merge datasets and clean missing values
- Calculate effectiveness metrics based on IC50 thresholds
- Export processed data for downstream analysis

Data Sources:
- GDSC drug screening dataset (IC50 values)
- Cell line annotations (tissue types, cancer types)
- Drug metadata (SMILES, targets, mechanisms)

Outputs:
- cleaned_data.csv - Processed drug-cell line combinations
- Data includes: drug_name, cell_line, IC50, effectiveness labels

Usage:
    loader = GDSCDataLoader()
    data = loader.process_pipeline(filter_gbm=True, save_output=True)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import logging
from tqdm import tqdm

# Try importing pyreadr, but make it optional
try:
    import pyreadr
    PYREADR_AVAILABLE = True
except ImportError:
    PYREADR_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("pyreadr not available. Will use CSV files instead.")

from .config import (
    GDSC1_FILE, GDSC2_FILE, MERGED_DATA_FILE, CLEANED_DATA_FILE,
    GBM_CELL_LINES, IMPUTATION_STRATEGY, MISSING_THRESHOLD,
    Z_SCORE_THRESHOLD, IC50_THRESHOLD_EFFECTIVE, AUC_THRESHOLD_EFFECTIVE
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GDSCDataLoader:
    """
    Load and process GDSC datasets from RDS or CSV files.
    
    Automatically handles both RDS and CSV formats:
    - Prefers CSV files if available (no pyreadr dependency)
    - Falls back to RDS if pyreadr is installed
    - Provides clear error messages if files are missing
    """
    
    def __init__(self, gdsc1_path: Path = GDSC1_FILE, gdsc2_path: Path = GDSC2_FILE):
        """
        Initialize data loader
        
        Args:
            gdsc1_path: Path to GDSC1 file (RDS or CSV)
            gdsc2_path: Path to GDSC2 file (RDS or CSV)
        """
        self.gdsc1_path = gdsc1_path
        self.gdsc2_path = gdsc2_path
        self.gdsc1_data = None
        self.gdsc2_data = None
        self.merged_data = None
        
    def load_rds_file(self, filepath: Path) -> pd.DataFrame:
        """
        Load RDS or CSV file and convert to pandas DataFrame
        Automatically detects format based on file extension.
        If RDS is not available, looks for CSV alternative.
        
        Args:
            filepath: Path to RDS or CSV file
            
        Returns:
            DataFrame containing the data
        """
        filepath = Path(filepath)
        
        # Check if CSV version exists (e.g., GDSC1.csv instead of GDSC1.rds)
        csv_path = filepath.with_suffix('.csv')
        
        if csv_path.exists():
            logger.info(f"Loading CSV file: {csv_path}")
            try:
                df = pd.read_csv(csv_path)
                logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns from CSV")
                return df
            except Exception as e:
                logger.error(f"Error loading CSV file {csv_path}: {e}")
                raise
        
        # Try RDS if pyreadr is available and file exists
        if filepath.exists() and filepath.suffix == '.rds':
            if not PYREADR_AVAILABLE:
                raise ImportError(
                    f"pyreadr is not installed, cannot read {filepath}. "
                    f"Please convert to CSV: {csv_path} or install pyreadr/rpy2."
                )
            
            logger.info(f"Loading RDS file: {filepath}")
            try:
                result = pyreadr.read_r(str(filepath))
                # RDS files typically have one object, get the first one
                df = result[None] if None in result else list(result.values())[0]
                logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns from RDS")
                return df
            except Exception as e:
                logger.error(f"Error loading RDS file {filepath}: {e}")
                raise
        
        # Neither CSV nor RDS found
        raise FileNotFoundError(
            f"Could not find data file. Tried:\n"
            f"  - CSV: {csv_path}\n"
            f"  - RDS: {filepath}\n"
            f"Please convert RDS files to CSV or install pyreadr."
        )
    
    def load_gdsc_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load both GDSC1 and GDSC2 datasets
        
        Returns:
            Tuple of (gdsc1_df, gdsc2_df)
        """
        self.gdsc1_data = self.load_rds_file(self.gdsc1_path)
        self.gdsc2_data = self.load_rds_file(self.gdsc2_path)
        
        return self.gdsc1_data, self.gdsc2_data
    
    def standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names across datasets
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with standardized column names
        """
        # Common column name mappings
        column_mapping = {
            'CELL_LINE_NAME': 'cell_line',
            'COSMIC_ID': 'cosmic_id',
            'DRUG_NAME': 'drug_name',
            'DRUG_ID': 'drug_id',
            'PUTATIVE_TARGET': 'target',
            'PATHWAY_NAME': 'pathway',
            'LN_IC50': 'ln_ic50',
            'AUC': 'auc',
            'RMSE': 'rmse',
            'Z_SCORE': 'z_score',
            'MAX_CONC': 'max_conc',
            'TCGA_DESC': 'tissue'
        }
        
        # Rename columns if they exist
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        return df
    
    def filter_gbm_cell_lines(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter data for GBM-specific cell lines
        
        Args:
            df: Input DataFrame
            
        Returns:
            Filtered DataFrame with only GBM cell lines
        """
        if 'cell_line' not in df.columns:
            logger.warning("'cell_line' column not found. Skipping GBM filtering.")
            return df
        
        logger.info(f"Filtering for GBM cell lines: {GBM_CELL_LINES}")
        
        # Case-insensitive matching
        df_filtered = df[df['cell_line'].str.upper().isin([cl.upper() for cl in GBM_CELL_LINES])].copy()
        
        logger.info(f"Found {len(df_filtered)} records for GBM cell lines")
        
        return df_filtered  # type: ignore[return-value]
    
    def merge_datasets(self, gdsc1: pd.DataFrame, gdsc2: pd.DataFrame) -> pd.DataFrame:
        """
        Merge GDSC1 and GDSC2 datasets
        
        Args:
            gdsc1: GDSC1 DataFrame
            gdsc2: GDSC2 DataFrame
            
        Returns:
            Merged DataFrame
        """
        logger.info("Merging GDSC1 and GDSC2 datasets")
        
        # Standardize column names
        gdsc1 = self.standardize_column_names(gdsc1)
        gdsc2 = self.standardize_column_names(gdsc2)
        
        # Add source column
        gdsc1['source'] = 'GDSC1'
        gdsc2['source'] = 'GDSC2'
        
        # Find common columns
        common_cols = list(set(gdsc1.columns) & set(gdsc2.columns))
        logger.info(f"Common columns: {common_cols}")
        
        # Concatenate datasets
        merged = pd.concat([gdsc1[common_cols], gdsc2[common_cols]], ignore_index=True)
        
        logger.info(f"Merged dataset shape: {merged.shape}")
        
        self.merged_data = merged
        return merged  # type: ignore[return-value]
    
    def convert_ln_ic50_to_ic50(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert LN_IC50 to IC50 in μM
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with IC50 column
        """
        if 'ln_ic50' in df.columns:
            # Convert from natural log to μM
            df['ic50'] = np.exp(df['ln_ic50'])
            logger.info("Converted ln_ic50 to ic50")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, 
                             strategy: str = IMPUTATION_STRATEGY) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df: Input DataFrame
            strategy: Imputation strategy ('mean', 'median', 'drop')
            
        Returns:
            DataFrame with handled missing values
        """
        logger.info(f"Handling missing values with strategy: {strategy}")
        
        # Report missing values
        missing_counts = df.isnull().sum()
        missing_pct = (missing_counts / len(df)) * 100
        
        logger.info("Missing value percentages:")
        for col in missing_pct[missing_pct > 0].index:
            logger.info(f"  {col}: {missing_pct[col]:.2f}%")
        
        # Drop columns with too many missing values
        cols_to_drop = missing_pct[missing_pct > MISSING_THRESHOLD * 100].index
        if len(cols_to_drop) > 0:
            logger.info(f"Dropping columns with >{MISSING_THRESHOLD*100}% missing: {list(cols_to_drop)}")
            df = df.drop(columns=cols_to_drop)
        
        # Impute remaining missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if strategy == 'mean':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif strategy == 'median':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif strategy == 'drop':
            df = df.dropna()
        
        logger.info(f"Dataset shape after handling missing values: {df.shape}")
        
        return df
    
    def remove_outliers(self, df: pd.DataFrame, 
                       columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Remove outliers using Z-score method
        
        Args:
            df: Input DataFrame
            columns: List of columns to check for outliers
            
        Returns:
            DataFrame with outliers removed
        """
        if columns is None:
            columns = ['ic50', 'auc', 'z_score']
            columns = [col for col in columns if col in df.columns]
        
        logger.info(f"Removing outliers from columns: {columns}")
        
        initial_len = len(df)
        
        for col in columns:
            if col in df.columns:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df = df[z_scores < Z_SCORE_THRESHOLD].copy()  # type: ignore[assignment]
        
        removed = initial_len - len(df)
        logger.info(f"Removed {removed} outlier records ({removed/initial_len*100:.2f}%)")
        
        return df
    
    def add_efficacy_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add labels for drug efficacy based on IC50 and AUC thresholds
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with efficacy labels
        """
        if 'ic50' in df.columns:
            df['is_effective_ic50'] = df['ic50'] < IC50_THRESHOLD_EFFECTIVE
        
        if 'auc' in df.columns:
            df['is_effective_auc'] = df['auc'] < AUC_THRESHOLD_EFFECTIVE
        
        # Combined efficacy (both criteria)
        if 'is_effective_ic50' in df.columns and 'is_effective_auc' in df.columns:
            df['is_effective'] = df['is_effective_ic50'] & df['is_effective_auc']
        
        logger.info("Added efficacy labels")
        
        return df
    
    def get_drug_summary_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate summary statistics for each drug
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with drug summary statistics
        """
        if 'drug_name' not in df.columns:
            logger.warning("'drug_name' column not found")
            return pd.DataFrame()
        
        summary_cols = []
        
        if 'ic50' in df.columns:
            summary_cols.append('ic50')
        if 'auc' in df.columns:
            summary_cols.append('auc')
        if 'z_score' in df.columns:
            summary_cols.append('z_score')
        
        if not summary_cols:
            logger.warning("No summary columns found")
            return pd.DataFrame()
        
        drug_summary = df.groupby('drug_name')[summary_cols].agg(['mean', 'std', 'min', 'max', 'count'])
        drug_summary.columns = ['_'.join(col).strip() for col in drug_summary.columns.values]
        drug_summary = drug_summary.reset_index()
        
        logger.info(f"Generated summary statistics for {len(drug_summary)} drugs")
        
        return drug_summary
    
    def process_pipeline(self, filter_gbm: bool = True, 
                         save_output: bool = True) -> pd.DataFrame:
        """
        Complete data processing pipeline
        
        Args:
            filter_gbm: Whether to filter for GBM cell lines only
            save_output: Whether to save processed data to file
            
        Returns:
            Processed DataFrame
        """
        logger.info("=" * 60)
        logger.info("Starting GDSC data processing pipeline")
        logger.info("=" * 60)
        
        # Load datasets
        gdsc1, gdsc2 = self.load_gdsc_datasets()
        
        # Merge datasets
        merged = self.merge_datasets(gdsc1, gdsc2)
        
        # Filter for GBM if requested
        if filter_gbm:
            merged = self.filter_gbm_cell_lines(merged)
        
        # Convert LN_IC50 to IC50
        merged = self.convert_ln_ic50_to_ic50(merged)
        
        # Handle missing values
        merged = self.handle_missing_values(merged)
        
        # Remove outliers
        merged = self.remove_outliers(merged)
        
        # Add efficacy labels
        merged = self.add_efficacy_labels(merged)
        
        # Save if requested
        if save_output:
            logger.info(f"Saving cleaned data to {CLEANED_DATA_FILE}")
            CLEANED_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
            merged.to_csv(CLEANED_DATA_FILE, index=False)
        
        logger.info("=" * 60)
        logger.info("Data processing pipeline complete")
        logger.info(f"Final dataset shape: {merged.shape}")
        logger.info("=" * 60)
        
        return merged


def load_processed_data() -> pd.DataFrame:
    """
    Load previously processed data
    
    Returns:
        Processed DataFrame
    """
    if CLEANED_DATA_FILE.exists():
        logger.info(f"Loading processed data from {CLEANED_DATA_FILE}")
        return pd.read_csv(CLEANED_DATA_FILE)
    else:
        logger.warning(f"Processed data file not found: {CLEANED_DATA_FILE}")
        logger.info("Running processing pipeline...")
        loader = GDSCDataLoader()
        return loader.process_pipeline()


if __name__ == "__main__":
    # Example usage
    loader = GDSCDataLoader()
    data = loader.process_pipeline()
    
    print("\nDataset Info:")
    print(data.info())
    print("\nFirst few rows:")
    print(data.head())
    
    # Get drug summary statistics
    drug_summary = loader.get_drug_summary_statistics(data)
    print("\nDrug Summary Statistics:")
    print(drug_summary.head(10))
