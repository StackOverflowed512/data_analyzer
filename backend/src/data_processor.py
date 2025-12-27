"""Data processing module for automobile dataset analysis."""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import chardet

from config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles loading, processing, and analyzing automobile dataset."""
    
    def __init__(self, dataset_path: Optional[str] = None):
        """
        Initialize the data processor.
        
        Args:
            dataset_path: Path to the automobile dataset CSV file
        """
        self.dataset_path = dataset_path or settings.DEFAULT_DATASET_PATH
        self.df: Optional[pd.DataFrame] = None
        self.columns: List[str] = []
        
    def load_dataset(self, dataset_path: Optional[str] = None) -> bool:
        """
        Load the automobile dataset from CSV file.
        
        Args:
            dataset_path: Path to the CSV file (uses default if not provided)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            path_to_load = dataset_path or self.dataset_path
            
            if not Path(path_to_load).exists():
                logger.error(f"Dataset file not found: {path_to_load}")
                return False
            
            logger.info(f"Loading dataset from: {path_to_load}")
            
            # Detect encoding more robustly
            detected_encoding = 'utf-8'
            try:
                with open(path_to_load, 'rb') as f:
                    raw_data = f.read(100000)  # Read more data for better detection
                    result = chardet.detect(raw_data)
                    if result and result.get('encoding'):
                        detected_encoding = result['encoding']
                        logger.info(f"Detected encoding: {detected_encoding} (confidence: {result.get('confidence', 0)})")
            except Exception as e:
                logger.warning(f"Could not detect encoding, will try fallback encodings: {str(e)}")
            
            # Try encodings in order of likelihood
            encodings_to_try = [
                detected_encoding,
                'utf-8',
                'utf-8-sig',  # UTF-8 with BOM
                'latin-1',
                'cp1252',
                'iso-8859-1',
                'ascii'
            ]
            
            # Remove duplicates while preserving order
            encodings_to_try = list(dict.fromkeys(encodings_to_try))
            
            self.df = None
            last_error = None
            
            for encoding in encodings_to_try:
                try:
                    logger.info(f"Attempting to load with encoding: {encoding}")
                    self.df = pd.read_csv(path_to_load, encoding=encoding, on_bad_lines='skip')
                    logger.info(f"Successfully loaded CSV with encoding: {encoding}")
                    break
                except (UnicodeDecodeError, LookupError) as e:
                    logger.debug(f"Failed to load with {encoding}: {str(e)}")
                    last_error = e
                    continue
        
            # Validate dataframe
            if self.df is None or self.df.empty:
                error_msg = f"Dataset is empty after trying all encodings. Last error: {str(last_error)}"
                logger.error(error_msg)
                return False
            
            self.columns = self.df.columns.tolist()
            self.dataset_path = path_to_load
            
            logger.info(f"Dataset loaded successfully: {len(self.df)} rows, {len(self.columns)} columns")
            logger.info(f"Columns: {self.columns}")
            
            return True
            
        except pd.errors.ParserError as e:
            logger.error(f"CSV parsing error: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            return False
    
    def get_columns(self) -> List[str]:
        """Get list of available columns in the dataset."""
        return self.columns.copy()
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get basic information about the dataset.
        
        Returns:
            Dictionary containing dataset information
        """
        if self.df is None:
            return {"error": "Dataset not loaded"}
            
        return {
            "rows": len(self.df),
            "columns": len(self.columns),
            "column_names": self.columns,
            "numeric_columns": self.df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": self.df.select_dtypes(include=['object']).columns.tolist(),
            "missing_values": self.df.isnull().sum().to_dict()
        }
    
    def process_data_for_chart(self, chart_specs: Dict[str, Any]) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Process data based on chart specifications from LLM. This is a more robust
        implementation with a clear order of operations.

        Order of Operations:
        1.  Start with a copy of the full dataframe.
        2.  Apply aggregation if specified ('agg'). This is a major transformation.
        3.  If no aggregation, apply column filtering if specified ('columns_only').
        4.  Apply sorting to the resulting dataframe ('sort_by').
        5.  Clean up data by dropping NA values from essential chart columns.
        6.  Limit the number of rows for performance.
        
        Args:
            chart_specs: Chart specifications from LLM analysis
            
        Returns:
            Tuple of (processed_dataframe, error_message)
        """
        if self.df is None:
            return None, "Dataset not loaded"
            
        try:
            processed_df = self.df.copy()

            # Extract specifications from the dictionary
            x_col = chart_specs.get("x")
            y_col = chart_specs.get("y")
            agg_method = chart_specs.get("agg")
            sort_by = chart_specs.get("sort_by")
            sort_order = chart_specs.get("sort_order", "asc")
            columns_only = chart_specs.get("columns_only")

            # --- Step 1: Aggregation ---
            # This step transforms the data, so it must happen before sorting on the new data.
            if agg_method and x_col and y_col:
                processed_df = self._apply_aggregation(processed_df, x_col, y_col, agg_method)
            
            # --- Step 2: Column Filtering (only if not aggregating) ---
            elif columns_only:
                valid_columns = [col for col in columns_only if col in self.columns]
                if not valid_columns:
                    return None, f"None of the requested columns {columns_only} exist."
                
                # Ensure the sort_by column is included if specified
                if sort_by and sort_by not in valid_columns:
                    valid_columns.append(sort_by)
                
                processed_df = processed_df[valid_columns]
            
            # --- Step 3: Sorting ---
            # This is now a distinct step, ensuring it runs on aggregated or filtered data.
            if sort_by:
                sort_by_col = sort_by[0] if isinstance(sort_by, list) else sort_by
                if sort_by_col in processed_df.columns:
                    ascending = sort_order.lower() in ["asc", "ascending"]
                    processed_df = processed_df.sort_values(by=sort_by_col, ascending=ascending)
                    logger.info(f"Applied sorting by {sort_by_col} ({sort_order})")
                else:
                    logger.warning(f"Sort column '{sort_by_col}' not found in the dataframe. Skipping sort.")

            # --- Step 4: Clean NA values for charting ---
            required_cols = [col for col in [x_col, y_col] if col is not None and col in processed_df.columns]
            if required_cols:
                processed_df = processed_df.dropna(subset=required_cols)
            
            # --- Step 5: Limit data points for performance ---
            if len(processed_df) > settings.MAX_CHART_POINTS:
                # Use head() for sorted data to show top/bottom results, otherwise sample
                if sort_by:
                    processed_df = processed_df.head(settings.MAX_CHART_POINTS)
                else:
                    processed_df = processed_df.sample(n=settings.MAX_CHART_POINTS, random_state=42)
                logger.warning(f"Data limited to {settings.MAX_CHART_POINTS} points for performance")
            
            return processed_df.reset_index(drop=True), None
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            return None, f"Data processing error: {str(e)}"
    
    def _apply_aggregation(self, df: pd.DataFrame, x_col: str, y_col: str, agg_method: str) -> pd.DataFrame:
        """
        Apply aggregation to the dataframe.
        
        Args:
            df: Input dataframe
            x_col: X-axis column name
            y_col: Y-axis column name
            agg_method: Aggregation method (mean, sum, count, median)
            
        Returns:
            Aggregated dataframe
        """
        try:
            if agg_method == "count":
                # For count, we count occurrences of x_col values
                result = df.groupby(x_col).size().reset_index(name=y_col)
            else:
                # For other aggregations, group by x_col and aggregate y_col
                agg_func = {
                    "mean": "mean",
                    "sum": "sum", 
                    "median": "median"
                }.get(agg_method, "mean")
                
                result = df.groupby(x_col)[y_col].agg(agg_func).reset_index()
            
            logger.info(f"Applied {agg_method} aggregation: {len(result)} groups")
            return result
            
        except Exception as e:
            logger.error(f"Aggregation failed: {str(e)}")
            return df
    
    def get_column_types(self) -> Dict[str, str]:
        """
        Get data types for each column.
        
        Returns:
            Dictionary mapping column names to their data types
        """
        if self.df is None:
            return {}
            
        type_mapping = {}
        for col in self.columns:
            if self.df[col].dtype in ['int64', 'float64']:
                type_mapping[col] = 'numeric'
            else:
                type_mapping[col] = 'categorical'
                
        return type_mapping
    
    def preview_data(self, n_rows: int = 5) -> Optional[pd.DataFrame]:
        """
        Get a preview of the dataset.
        
        Args:
            n_rows: Number of rows to return
            
        Returns:
            Preview dataframe or None if dataset not loaded
        """
        if self.df is None:
            return None
            
        return self.df.head(n_rows)
    
    def get_full_dataset(self, page: int = 1, page_size: int = 50) -> Dict[str, Any]:
        """
        Get paginated dataset for viewing.
        
        Args:
            page: Page number (1-indexed)
            page_size: Number of rows per page
            
        Returns:
            Dictionary with paginated data and metadata
        """
        if self.df is None:
            return {"error": "Dataset not loaded"}
        
        total_rows = len(self.df)
        total_pages = (total_rows + page_size - 1) // page_size
        
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        page_data = self.df.iloc[start_idx:end_idx]
        
        return {
            "data": page_data.to_dict('records'),
            "pagination": {
                "current_page": page,
                "page_size": page_size,
                "total_rows": total_rows,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_previous": page > 1
            }
        }
    
    def get_column_statistics(self) -> Dict[str, Any]:
        """
        Get detailed statistics for each column.
        
        Returns:
            Dictionary with column statistics
        """
        if self.df is None:
            return {"error": "Dataset not loaded"}
        
        stats = {}
        
        for col in self.columns:
            col_stats = {
                "name": col,
                "type": str(self.df[col].dtype),
                "non_null_count": int(self.df[col].count()),
                "null_count": int(self.df[col].isnull().sum()),
                "unique_values": int(self.df[col].nunique())
            }
            
            if self.df[col].dtype in ['int64', 'float64']:
                col_stats.update({
                    "data_type": "numeric",
                    "min": float(self.df[col].min()),
                    "max": float(self.df[col].max()),
                    "mean": float(self.df[col].mean()),
                    "median": float(self.df[col].median()),
                    "std": float(self.df[col].std())
                })
            else:
                col_stats.update({
                    "data_type": "categorical",
                    "top_values": self.df[col].value_counts().head(5).to_dict()
                })
            
            stats[col] = col_stats
        
        return stats
    
    def filter_data(self, filters: Dict[str, Any]) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Filter dataset based on provided filters.
        
        Args:
            filters: Dictionary with filter specifications
            
        Returns:
            Tuple of (filtered_dataframe, error_message)
        """
        if self.df is None:
            return None, "Dataset not loaded"
        
        try:
            filtered_df = self.df.copy()
            
            for column, filter_spec in filters.items():
                if column not in self.columns:
                    continue
                
                filter_type = filter_spec.get("type")
                value = filter_spec.get("value")
                
                if filter_type == "equals":
                    filtered_df = filtered_df[filtered_df[column] == value]
                elif filter_type == "not_equals":
                    filtered_df = filtered_df[filtered_df[column] != value]
                elif filter_type == "greater_than":
                    filtered_df = filtered_df[filtered_df[column] > value]
                elif filter_type == "less_than":
                    filtered_df = filtered_df[filtered_df[column] < value]
                elif filter_type == "contains":
                    filtered_df = filtered_df[filtered_df[column].astype(str).str.contains(str(value), case=False, na=False)]
                elif filter_type == "in":
                    if isinstance(value, list):
                        filtered_df = filtered_df[filtered_df[column].isin(value)]
                elif filter_type == "range":
                    min_val = filter_spec.get("min")
                    max_val = filter_spec.get("max")
                    if min_val is not None:
                        filtered_df = filtered_df[filtered_df[column] >= min_val]
                    if max_val is not None:
                        filtered_df = filtered_df[filtered_df[column] <= max_val]
            
            return filtered_df, None
            
        except Exception as e:
            logger.error(f"Filtering failed: {str(e)}")
            return None, f"Filtering error: {str(e)}"
    
    def search_data(self, search_term: str, columns: Optional[List[str]] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Search for data containing the search term.
        
        Args:
            search_term: Term to search for
            columns: Specific columns to search in (if None, search all text columns)
            
        Returns:
            Tuple of (search_results_dataframe, error_message)
        """
        if self.df is None:
            return None, "Dataset not loaded"
        
        try:
            if columns is None:
                # Search in all object/string columns
                columns = self.df.select_dtypes(include=['object']).columns.tolist()
            
            # Create a mask for rows that contain the search term
            mask = pd.Series([False] * len(self.df))
            
            for col in columns:
                if col in self.df.columns:
                    mask |= self.df[col].astype(str).str.contains(search_term, case=False, na=False)
            
            result_df = self.df[mask]
            return result_df, None
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return None, f"Search error: {str(e)}"
    
    def to_json_safe(self, df):
        """Convert DataFrame to JSON-safe format, replacing NaN with None."""
        import pandas as pd
        import numpy as np
        if df is None or df.empty:
            return []
        
        # Replace NaN with None
        df_clean = df.where(pd.notna(df), None)
        records = df_clean.to_dict('records')
        
        # Additional cleanup for any remaining NaN/inf values
        cleaned_records = []
        for row in records:
            cleaned_row = {}
            for k, v in row.items():
                if isinstance(v, float):
                    if np.isnan(v) or np.isinf(v):
                        cleaned_row[k] = None
                    else:
                        cleaned_row[k] = v
                else:
                    cleaned_row[k] = v
            cleaned_records.append(cleaned_row)
        
        return cleaned_records


# Global data processor instance
data_processor = DataProcessor()