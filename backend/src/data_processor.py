"""Data processing module for automobile dataset analysis."""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import chardet

from config.settings import settings
from sqlalchemy import create_engine, text
import os

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
        self.current_dataset_id: Optional[int] = None
        self.current_dataset_name: Optional[str] = None
        
        # Database configuration
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        os.makedirs(self.data_dir, exist_ok=True)
        self.db_path = os.path.join(self.data_dir, "app_database.db")
        self.db_engine = create_engine(f"sqlite:///{self.db_path}")
        
        # Initialize metadata table
        self._init_db()

    def _init_db(self):
        """Initialize database tables."""
        try:
            with self.db_engine.connect() as conn:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS datasets (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT NOT NULL,
                        upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        row_count INTEGER,
                        column_count INTEGER,
                        columns TEXT
                    )
                """))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")

    def save_dataset(self, filename: str) -> bool:
        """Save current dataframe as a new dataset in database."""
        if self.df is None:
            return False
        try:
            import json
            from datetime import datetime
            
            # Save actual data to a new table or overwrite distinct one
            # We'll create a new entry every time for history, or we could check for duplicates.
            # For this request: "all datasets uploaded". So we add a new one.
            
            # 1. Insert metadata
            columns_json = json.dumps(self.columns)
            with self.db_engine.connect() as conn:
                result = conn.execute(text("""
                    INSERT INTO datasets (filename, row_count, column_count, columns, upload_date)
                    VALUES (:filename, :row_count, :col_count, :columns, :date)
                """), {
                    "filename": filename,
                    "row_count": len(self.df),
                    "col_count": len(self.columns),
                    "columns": columns_json,
                    "date": datetime.now()
                })
                dataset_id = result.lastrowid
                conn.commit()
            
            # 2. Save data to specific table
            table_name = f"dataset_{dataset_id}"
            self.df.to_sql(table_name, self.db_engine, if_exists='replace', index=False)
            
            self.current_dataset_id = dataset_id
            self.current_dataset_name = filename
            
            logger.info(f"Dataset '{filename}' saved with ID {dataset_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save dataset to database: {str(e)}")
            return False

    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all available datasets."""
        try:
            with self.db_engine.connect() as conn:
                result = conn.execute(text("SELECT id, filename, upload_date, row_count, column_count FROM datasets ORDER BY upload_date DESC"))
                datasets = []
                for row in result:
                    datasets.append({
                        "id": row[0],
                        "filename": row[1],
                        "upload_date": str(row[2]),
                        "row_count": row[3],
                        "column_count": row[4]
                    })
                return datasets
        except Exception as e:
            logger.error(f"Failed to list datasets: {str(e)}")
            return []

    def load_dataset_by_id(self, dataset_id: int) -> bool:
        """Load a specific dataset from database by ID."""
        try:
            # Get metadata
            with self.db_engine.connect() as conn:
                result = conn.execute(text("SELECT filename FROM datasets WHERE id = :id"), {"id": dataset_id})
                row = result.fetchone()
                if not row:
                    logger.error(f"Dataset ID {dataset_id} not found")
                    return False
                filename = row[0]

            # Load data
            table_name = f"dataset_{dataset_id}"
            self.df = pd.read_sql(table_name, self.db_engine)
            
            if self.df is None or self.df.empty:
                return False
                
            self.columns = self.df.columns.tolist()
            self.current_dataset_id = dataset_id
            self.current_dataset_name = filename
            
            logger.info(f"Loaded dataset '{filename}' (ID: {dataset_id}) from database")
            return True
        except Exception as e:
            logger.error(f"Failed to load dataset by ID: {str(e)}")
            return False

    def load_from_db(self) -> bool:
        """Load most recent dataset from database."""
        try:
            if not os.path.exists(self.db_path):
                return False
                
            # Find most recent dataset
            with self.db_engine.connect() as conn:
                # Check for legacy table first (from previous step)
                # If 'datasets' table doesn't exist but 'dataset' does, migrate or just load 'dataset'
                # But we initialized 'datasets' in __init__, so it exists.
                
                result = conn.execute(text("SELECT id FROM datasets ORDER BY upload_date DESC LIMIT 1"))
                row = result.fetchone()
                
                if row:
                    return self.load_dataset_by_id(row[0])
                else:
                    # Fallback for checking if legacy 'dataset' table exists from previous step
                    # To allow smooth transition if app was running
                    result_legacy = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='dataset'"))
                    if result_legacy.fetchone():
                        logger.info("Found legacy 'dataset' table, loading it...")
                        self.df = pd.read_sql('dataset', self.db_engine)
                        self.columns = self.df.columns.tolist()
                        self.current_dataset_name = "Restored Session"
                        # We could migrate it to new structure here, but let's just use it
                        return True
                        
            return False
        except Exception as e:
            logger.error(f"Failed to load from database: {str(e)}")
            return False
        
    def load_dataset(self, dataset_path: Optional[str] = None) -> bool:
        """
        Load the automobile dataset from CSV file.
        
        Args:
            dataset_path: Path to the CSV file (uses default if not provided)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # If no specific path provided, try loading from DB first
            if dataset_path is None:
                if self.load_from_db():
                    return True
            
            path_to_load = dataset_path or self.dataset_path
            
            if not Path(path_to_load).exists():
                logger.error(f"Dataset file not found: {path_to_load}")
                return False
            
            logger.info(f"Loading dataset from: {path_to_load}")
            
            # Determine file type
            is_excel = path_to_load.lower().endswith(('.xlsx', '.xls'))
            
            self.df = None
            last_error = None
            
            if is_excel:
                try:
                    logger.info(f"Attempting to load Excel file: {path_to_load}")
                    self.df = pd.read_excel(path_to_load)
                    logger.info("Successfully loaded Excel file")
                except Exception as e:
                    logger.error(f"Failed to load Excel file: {str(e)}")
                    last_error = e
                    return False
            else:
                # Detect encoding more robustly for CSV
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
                error_msg = f"Dataset is empty after loading. Last error: {str(last_error)}"
                logger.error(error_msg)
                return False
            
            self.columns = self.df.columns.tolist()
            self.dataset_path = path_to_load
            
            logger.info(f"Dataset loaded successfully: {len(self.df)} rows, {len(self.columns)} columns")
            logger.info(f"Columns: {self.columns}")
            
            # Save to database for persistence
            filename = os.path.basename(path_to_load) if path_to_load else "Uploaded Dataset"
            self.save_dataset(filename)
            
            return True
            
        except pd.errors.ParserError as e:
            logger.error(f"Parsing error: {str(e)}")
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
            
        missing_values = self.df.isnull().sum().to_dict()
        # Convert numpy ints to standard python ints for json serialization
        missing_values = {k: int(v) for k, v in missing_values.items()}
        
        return {
            "name": self.current_dataset_name or "Unknown Dataset",
            "id": self.current_dataset_id,
            "rows": int(len(self.df)),
            "columns": int(len(self.columns)),
            "column_names": self.columns,
            "numeric_columns": self.df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": self.df.select_dtypes(include=['object']).columns.tolist(),
            "missing_values": missing_values
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
            "data": self.to_json_safe(page_data),
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
                top_values = self.df[col].value_counts().head(5).to_dict()
                # Convert keys and values to standard python types
                cleaned_top_values = {}
                for k, v in top_values.items():
                    key = str(k) # Ensure key is string
                    val = int(v) # Ensure count is int
                    cleaned_top_values[key] = val
                    
                col_stats.update({
                    "data_type": "categorical",
                    "top_values": cleaned_top_values
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


    
    def generate_smart_suggestions(self) -> List[Dict[str, str]]:
        """
        Generate smart suggested queries based on dataset columns.
        
        Returns:
            List of dictionaries with 'query' and 'description' keys
        """
        if self.df is None or not self.columns:
            return []
            
        suggestions = []
        
        # Classify columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # 1. Distribution of a categorical column
        if categorical_cols:
            col = categorical_cols[0]
            suggestions.append({
                "query": f"Show distribution of {col}", 
                "description": f"View how many items belong to each {col}"
            })
            
            if len(categorical_cols) > 1:
                col2 = categorical_cols[1]
                suggestions.append({
                    "query": f"Show {col2} counts as bar chart",
                    "description": f"Bar chart showing frequency of {col2}"
                })

        # 2. Average of numeric by categorical
        if numeric_cols and categorical_cols:
            num = numeric_cols[0]
            cat = categorical_cols[0]
            suggestions.append({
                "query": f"Show average {num} by {cat}",
                "description": f"Compare average {num} across different {cat}"
            })
            
            if len(numeric_cols) > 1:
                num2 = numeric_cols[1]
                suggestions.append({
                    "query": f"Compare {num2} by {cat}",
                    "description": f"Analyze {num2} statistics for each {cat}"
                })

        # 3. Correlation/Scatter plot (Numeric vs Numeric)
        if len(numeric_cols) >= 2:
            num1 = numeric_cols[0]
            num2 = numeric_cols[1]
            suggestions.append({
                "query": f"Plot {num1} vs {num2}",
                "description": f"Scatter plot showing relationship between {num1} and {num2}"
            })
        
        # 4. Top/Bottom analysis
        if numeric_cols and categorical_cols:
            num = numeric_cols[0]
            cat = categorical_cols[0]
            suggestions.append({
                "query": f"Top 10 {cat} with highest {num}",
                "description": f"List the top performing {cat} based on {num}"
            })

        # 5. Histogram of numeric
        if numeric_cols:
            col = numeric_cols[0]
            suggestions.append({
                "query": f"Show distribution of {col}",
                "description": f"Histogram showing how {col} is distributed"
            })
            
        return suggestions[:8]  # Return top 8 suggestions

# Global data processor instance
data_processor = DataProcessor()