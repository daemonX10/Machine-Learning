# Pandas Interview Questions - General Questions

## Question 1

**How can you read and write data from and to a CSV file in Pandas?**

### Answer

#### Theory
CSV (Comma-Separated Values) files are one of the most common data exchange formats in data science. Pandas provides robust and flexible functions for reading CSV files (`pd.read_csv()`) and writing DataFrames to CSV format (`DataFrame.to_csv()`). These functions offer extensive customization options for handling various CSV formats, encodings, data types, and edge cases commonly encountered in real-world data.

#### Code Example

```python
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ======================== READING CSV FILES ========================

def demonstrate_csv_reading():
    """Comprehensive demonstration of CSV reading capabilities."""
    
    print("=== CSV READING DEMONSTRATIONS ===")
    
    # First, create sample CSV files for demonstration
    create_sample_csv_files()
    
    # 1. Basic CSV reading
    print("\n1. Basic CSV Reading:")
    try:
        df_basic = pd.read_csv('sample_data.csv')
        print(f"   Shape: {df_basic.shape}")
        print(f"   Columns: {list(df_basic.columns)}")
        print("   First 3 rows:")
        print(df_basic.head(3))
    except FileNotFoundError:
        print("   Sample file not found - creating one...")
        create_sample_csv_files()
        df_basic = pd.read_csv('sample_data.csv')
        print(f"   Shape: {df_basic.shape}")
    
    # 2. Advanced reading with parameters
    print("\n2. Advanced CSV Reading with Parameters:")
    
    # Read with specific data types
    dtype_specification = {
        'id': 'int32',
        'name': 'string',
        'age': 'int16',
        'salary': 'float32',
        'department': 'category'
    }
    
    df_typed = pd.read_csv(
        'sample_data.csv',
        dtype=dtype_specification,
        parse_dates=['hire_date'],  # Automatically parse date columns
        index_col='id',             # Set 'id' as index
        usecols=['id', 'name', 'age', 'salary', 'department', 'hire_date']  # Read specific columns
    )
    
    print(f"   Data types optimized:")
    print(df_typed.dtypes)
    print(f"   Memory usage: {df_typed.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    # 3. Handling problematic CSV files
    print("\n3. Handling Problematic CSV Files:")
    
    # Read file with different separators
    try:
        df_semicolon = pd.read_csv(
            'sample_semicolon.csv',
            sep=';',                    # Different separator
            decimal=',',                # Different decimal separator
            encoding='utf-8',           # Specify encoding
            skipinitialspace=True,      # Skip spaces after delimiter
            skip_blank_lines=True       # Skip blank lines
        )
        print(f"   Semicolon-separated file shape: {df_semicolon.shape}")
    except FileNotFoundError:
        print("   Semicolon file not found - creating one...")
        create_problematic_csv()
        df_semicolon = pd.read_csv('sample_semicolon.csv', sep=';', decimal=',')
    
    # 4. Reading large files efficiently
    print("\n4. Reading Large Files Efficiently:")
    
    # Read in chunks
    def read_csv_in_chunks(filename: str, chunk_size: int = 1000):
        """Read large CSV files in chunks."""
        chunks = []
        for chunk in pd.read_csv(filename, chunksize=chunk_size):
            # Process each chunk (e.g., filter, transform)
            processed_chunk = chunk[chunk['age'] > 25]  # Example processing
            chunks.append(processed_chunk)
        
        return pd.concat(chunks, ignore_index=True)
    
    # Example with chunked reading
    if os.path.exists('sample_data.csv'):
        df_chunked = read_csv_in_chunks('sample_data.csv', chunk_size=50)
        print(f"   Chunked reading result shape: {df_chunked.shape}")
    
    # 5. Error handling and data validation
    print("\n5. Error Handling and Data Validation:")
    
    def robust_csv_reader(filename: str, **kwargs) -> Dict[str, Any]:
        """Robust CSV reader with comprehensive error handling."""
        
        result = {
            'success': False,
            'dataframe': None,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        try:
            # First, try to read with basic parameters
            df = pd.read_csv(filename, **kwargs)
            
            # Validate the data
            validation_issues = []
            
            # Check for completely empty DataFrame
            if df.empty:
                validation_issues.append("DataFrame is empty")
            
            # Check for columns with all missing values
            all_missing_cols = df.columns[df.isnull().all()].tolist()
            if all_missing_cols:
                validation_issues.append(f"Columns with all missing values: {all_missing_cols}")
            
            # Check for duplicate column names
            if df.columns.duplicated().any():
                validation_issues.append("Duplicate column names found")
            
            # Check for mixed data types in columns
            mixed_type_cols = []
            for col in df.select_dtypes(include=['object']).columns:
                if df[col].dtype == 'object':
                    # Check if column contains mixed numeric and string data
                    sample = df[col].dropna().head(100)
                    if len(sample) > 0:
                        numeric_count = pd.to_numeric(sample, errors='coerce').notna().sum()
                        if 0 < numeric_count < len(sample):
                            mixed_type_cols.append(col)
            
            if mixed_type_cols:
                validation_issues.append(f"Columns with mixed data types: {mixed_type_cols}")
            
            result.update({
                'success': True,
                'dataframe': df,
                'warnings': validation_issues,
                'info': {
                    'shape': df.shape,
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
                    'dtypes': df.dtypes.to_dict(),
                    'missing_values': df.isnull().sum().to_dict()
                }
            })
            
        except FileNotFoundError:
            result['errors'].append(f"File not found: {filename}")
        except pd.errors.EmptyDataError:
            result['errors'].append("No data found in file")
        except pd.errors.ParserError as e:
            result['errors'].append(f"Parser error: {str(e)}")
        except UnicodeDecodeError as e:
            result['errors'].append(f"Encoding error: {str(e)}")
        except Exception as e:
            result['errors'].append(f"Unexpected error: {str(e)}")
        
        return result
    
    # Test robust reader
    result = robust_csv_reader('sample_data.csv')
    if result['success']:
        print(f"   Robust reader successful: {result['info']['shape']}")
        if result['warnings']:
            print(f"   Warnings: {result['warnings']}")
    else:
        print(f"   Robust reader failed: {result['errors']}")

# ======================== WRITING CSV FILES ========================

def demonstrate_csv_writing():
    """Comprehensive demonstration of CSV writing capabilities."""
    
    print("\n=== CSV WRITING DEMONSTRATIONS ===")
    
    # Create sample DataFrame for writing examples
    sample_df = create_sample_dataframe()
    
    # 1. Basic CSV writing
    print("\n1. Basic CSV Writing:")
    
    # Simple write
    sample_df.to_csv('output_basic.csv', index=False)
    print(f"   Written to 'output_basic.csv' - Shape: {sample_df.shape}")
    
    # Verify the write
    verification_df = pd.read_csv('output_basic.csv')
    print(f"   Verification read - Shape: {verification_df.shape}")
    
    # 2. Advanced writing options
    print("\n2. Advanced CSV Writing Options:")
    
    # Write with custom formatting
    sample_df.to_csv(
        'output_formatted.csv',
        index=False,                    # Don't include row indices
        sep=';',                        # Use semicolon separator
        encoding='utf-8',               # Specify encoding
        date_format='%Y-%m-%d',         # Format dates
        float_format='%.2f',            # Format floats to 2 decimal places
        na_rep='NULL',                  # Replace NaN with 'NULL'
        columns=['name', 'age', 'department', 'salary'],  # Select specific columns
        header=['Employee_Name', 'Age', 'Department', 'Salary']  # Custom headers
    )
    print(f"   Written formatted CSV with custom options")
    
    # 3. Writing subsets and filtered data
    print("\n3. Writing Data Subsets:")
    
    # Write filtered data
    high_earners = sample_df[sample_df['salary'] > sample_df['salary'].median()]
    high_earners.to_csv('high_earners.csv', index=False)
    print(f"   High earners subset written: {len(high_earners)} rows")
    
    # Write specific columns
    summary_columns = ['name', 'department', 'salary']
    sample_df[summary_columns].to_csv('summary.csv', index=False)
    print(f"   Summary with selected columns written")
    
    # 4. Appending to existing CSV files
    print("\n4. Appending to Existing Files:")
    
    def append_to_csv(df: pd.DataFrame, filename: str, **kwargs):
        """Append DataFrame to existing CSV file."""
        
        if os.path.exists(filename):
            # File exists, append without header
            df.to_csv(filename, mode='a', header=False, index=False, **kwargs)
            print(f"   Appended {len(df)} rows to existing file")
        else:
            # File doesn't exist, create with header
            df.to_csv(filename, mode='w', header=True, index=False, **kwargs)
            print(f"   Created new file with {len(df)} rows")
    
    # Create initial file
    initial_data = sample_df.head(5)
    append_to_csv(initial_data, 'incremental.csv')
    
    # Append more data
    additional_data = sample_df.tail(3)
    append_to_csv(additional_data, 'incremental.csv')
    
    # Verify combined file
    combined_df = pd.read_csv('incremental.csv')
    print(f"   Combined file size: {len(combined_df)} rows")
    
    # 5. Memory-efficient writing for large DataFrames
    print("\n5. Memory-Efficient Writing:")
    
    def write_large_dataframe_chunked(df: pd.DataFrame, filename: str, chunk_size: int = 1000):
        """Write large DataFrame in chunks to manage memory."""
        
        total_rows = len(df)
        chunks_written = 0
        
        for i in range(0, total_rows, chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            
            # Write first chunk with header, subsequent chunks without
            if i == 0:
                chunk.to_csv(filename, mode='w', header=True, index=False)
            else:
                chunk.to_csv(filename, mode='a', header=False, index=False)
            
            chunks_written += 1
        
        print(f"   Large DataFrame written in {chunks_written} chunks")
        return chunks_written
    
    # Simulate large DataFrame
    large_df = pd.concat([sample_df] * 5, ignore_index=True)  # Replicate for demo
    chunks = write_large_dataframe_chunked(large_df, 'large_output.csv', chunk_size=20)
    
    # 6. Specialized CSV writing scenarios
    print("\n6. Specialized Writing Scenarios:")
    
    # Write with compression
    sample_df.to_csv('compressed_output.csv.gz', compression='gzip', index=False)
    print(f"   Compressed CSV written")
    
    # Compare file sizes
    normal_size = os.path.getsize('output_basic.csv')
    compressed_size = os.path.getsize('compressed_output.csv.gz')
    compression_ratio = normal_size / compressed_size
    print(f"   Compression ratio: {compression_ratio:.2f}x")
    
    # Write to string buffer (for testing or in-memory operations)
    from io import StringIO
    
    csv_buffer = StringIO()
    sample_df.to_csv(csv_buffer, index=False)
    csv_string = csv_buffer.getvalue()
    print(f"   CSV written to string buffer: {len(csv_string)} characters")

# ======================== HELPER FUNCTIONS ========================

def create_sample_csv_files():
    """Create sample CSV files for demonstration."""
    
    # Basic sample data
    data = {
        'id': range(1, 101),
        'name': [f'Employee_{i}' for i in range(1, 101)],
        'age': np.random.randint(22, 65, 100),
        'salary': np.random.normal(75000, 20000, 100).round(2),
        'department': np.random.choice(['Engineering', 'Marketing', 'Sales', 'HR'], 100),
        'hire_date': pd.date_range('2020-01-01', periods=100, freq='W').strftime('%Y-%m-%d')
    }
    
    df = pd.DataFrame(data)
    df.to_csv('sample_data.csv', index=False)
    print("   Created 'sample_data.csv'")

def create_problematic_csv():
    """Create CSV with common formatting issues."""
    
    # Data with European formatting (semicolon separator, comma decimal)
    problematic_data = """name;age;salary;department
John Doe;30;75000,50;Engineering
Jane Smith;28;68000,75;Marketing
Bob Johnson;35;82000,25;Sales"""
    
    with open('sample_semicolon.csv', 'w', encoding='utf-8') as f:
        f.write(problematic_data)
    print("   Created 'sample_semicolon.csv' with European formatting")

def create_sample_dataframe() -> pd.DataFrame:
    """Create sample DataFrame for writing demonstrations."""
    
    np.random.seed(42)
    
    data = {
        'name': [f'Employee_{i}' for i in range(1, 21)],
        'age': np.random.randint(25, 60, 20),
        'department': np.random.choice(['Engineering', 'Marketing', 'Sales', 'HR'], 20),
        'salary': np.random.normal(70000, 15000, 20).round(2),
        'hire_date': pd.date_range('2021-01-01', periods=20, freq='M'),
        'performance_rating': np.random.uniform(3.0, 5.0, 20).round(1)
    }
    
    df = pd.DataFrame(data)
    
    # Introduce some missing values
    df.loc[2, 'performance_rating'] = np.nan
    df.loc[7, 'age'] = np.nan
    
    return df

# ======================== ADVANCED TECHNIQUES ========================

def advanced_csv_techniques():
    """Demonstrate advanced CSV handling techniques."""
    
    print("\n=== ADVANCED CSV TECHNIQUES ===")
    
    # 1. Dynamic data type inference
    print("\n1. Dynamic Data Type Inference:")
    
    def infer_and_optimize_dtypes(filename: str) -> pd.DataFrame:
        """Read CSV and automatically optimize data types."""
        
        # First pass: read small sample to infer types
        sample_df = pd.read_csv(filename, nrows=1000)
        
        # Analyze and optimize dtypes
        dtype_dict = {}
        
        for col in sample_df.columns:
            if sample_df[col].dtype == 'object':
                # Try to convert to numeric
                numeric_series = pd.to_numeric(sample_df[col], errors='coerce')
                if not numeric_series.isna().all():
                    # Determine best numeric type
                    if numeric_series.dtype == 'int64':
                        min_val, max_val = numeric_series.min(), numeric_series.max()
                        if min_val >= 0:
                            if max_val < 256:
                                dtype_dict[col] = 'uint8'
                            elif max_val < 65536:
                                dtype_dict[col] = 'uint16'
                            else:
                                dtype_dict[col] = 'uint32'
                        else:
                            if min_val >= -128 and max_val < 128:
                                dtype_dict[col] = 'int8'
                            elif min_val >= -32768 and max_val < 32768:
                                dtype_dict[col] = 'int16'
                            else:
                                dtype_dict[col] = 'int32'
                    else:
                        dtype_dict[col] = 'float32'
                else:
                    # Check if it should be categorical
                    unique_ratio = sample_df[col].nunique() / len(sample_df[col])
                    if unique_ratio < 0.5:
                        dtype_dict[col] = 'category'
        
        # Second pass: read full file with optimized types
        optimized_df = pd.read_csv(filename, dtype=dtype_dict)
        
        print(f"   Optimized data types: {dict(optimized_df.dtypes)}")
        return optimized_df
    
    # Test with sample data
    if os.path.exists('sample_data.csv'):
        optimized_df = infer_and_optimize_dtypes('sample_data.csv')
        print(f"   Optimized DataFrame memory usage: {optimized_df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    # 2. Handling encoding issues
    print("\n2. Handling Encoding Issues:")
    
    def detect_and_read_encoding(filename: str) -> pd.DataFrame:
        """Detect file encoding and read CSV accordingly."""
        
        import chardet
        
        # Detect encoding
        with open(filename, 'rb') as f:
            raw_data = f.read(10000)  # Read first 10KB for detection
            encoding_result = chardet.detect(raw_data)
        
        detected_encoding = encoding_result['encoding']
        confidence = encoding_result['confidence']
        
        print(f"   Detected encoding: {detected_encoding} (confidence: {confidence:.2f})")
        
        # Read with detected encoding
        try:
            df = pd.read_csv(filename, encoding=detected_encoding)
            print(f"   Successfully read with detected encoding")
            return df
        except UnicodeDecodeError:
            print(f"   Failed with detected encoding, trying UTF-8 with error handling")
            df = pd.read_csv(filename, encoding='utf-8', encoding_errors='replace')
            return df
    
    # 3. Configuration-driven CSV processing
    print("\n3. Configuration-Driven Processing:")
    
    def process_csv_with_config(filename: str, config: Dict[str, Any]) -> pd.DataFrame:
        """Process CSV file based on configuration dictionary."""
        
        # Extract configuration parameters
        read_params = config.get('read_params', {})
        transformations = config.get('transformations', [])
        validations = config.get('validations', [])
        
        # Read CSV with specified parameters
        df = pd.read_csv(filename, **read_params)
        
        # Apply transformations
        for transform in transformations:
            transform_type = transform['type']
            
            if transform_type == 'rename_columns':
                df = df.rename(columns=transform['mapping'])
            elif transform_type == 'convert_dtypes':
                df = df.astype(transform['dtypes'])
            elif transform_type == 'fill_missing':
                df = df.fillna(transform['values'])
            elif transform_type == 'filter_rows':
                query_string = transform['query']
                df = df.query(query_string)
        
        # Apply validations
        validation_results = []
        for validation in validations:
            validation_type = validation['type']
            
            if validation_type == 'required_columns':
                missing_cols = set(validation['columns']) - set(df.columns)
                if missing_cols:
                    validation_results.append(f"Missing required columns: {missing_cols}")
            elif validation_type == 'data_range':
                col = validation['column']
                min_val, max_val = validation['min'], validation['max']
                out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
                if out_of_range > 0:
                    validation_results.append(f"Column {col}: {out_of_range} values out of range")
        
        if validation_results:
            print(f"   Validation issues: {validation_results}")
        else:
            print(f"   All validations passed")
        
        return df
    
    # Example configuration
    csv_config = {
        'read_params': {
            'dtype': {'age': 'int16', 'department': 'category'},
            'parse_dates': ['hire_date']
        },
        'transformations': [
            {
                'type': 'rename_columns',
                'mapping': {'name': 'employee_name'}
            },
            {
                'type': 'filter_rows',
                'query': 'age >= 25'
            }
        ],
        'validations': [
            {
                'type': 'required_columns',
                'columns': ['employee_name', 'age', 'department']
            },
            {
                'type': 'data_range',
                'column': 'age',
                'min': 18,
                'max': 70
            }
        ]
    }
    
    if os.path.exists('sample_data.csv'):
        processed_df = process_csv_with_config('sample_data.csv', csv_config)
        print(f"   Processed DataFrame shape: {processed_df.shape}")

# ======================== BEST PRACTICES ========================

def csv_best_practices_guide():
    """Comprehensive guide to CSV best practices."""
    
    print("\n" + "="*60)
    print("CSV HANDLING BEST PRACTICES")
    print("="*60)
    
    best_practices = {
        "Reading CSVs": [
            "• Always specify data types explicitly for better performance",
            "• Use parse_dates for datetime columns",
            "• Consider using chunksize for large files",
            "• Specify encoding to avoid Unicode errors",
            "• Use usecols to read only needed columns",
            "• Handle missing values appropriately with na_values parameter"
        ],
        "Writing CSVs": [
            "• Set index=False unless the index is meaningful",
            "• Use appropriate encoding (UTF-8 recommended)",
            "• Format dates and numbers consistently",
            "• Consider compression for large files",
            "• Use descriptive column names",
            "• Handle missing values explicitly with na_rep parameter"
        ],
        "Performance Optimization": [
            "• Specify dtypes to avoid inference overhead",
            "• Use categorical data types for repetitive strings",
            "• Read only necessary columns with usecols",
            "• Use chunking for memory-constrained environments",
            "• Consider using more efficient formats (Parquet, Feather) for frequent access"
        ],
        "Error Handling": [
            "• Always wrap CSV operations in try-catch blocks",
            "• Validate data after reading",
            "• Handle encoding issues gracefully",
            "• Check for empty files and malformed data",
            "• Log processing steps for debugging"
        ],
        "Data Quality": [
            "• Validate column names and data types",
            "• Check for unexpected missing values",
            "• Verify data ranges and constraints",
            "• Handle duplicate rows appropriately",
            "• Ensure consistent data formatting"
        ]
    }
    
    for category, practices in best_practices.items():
        print(f"\n{category}:")
        for practice in practices:
            print(f"  {practice}")

# ======================== MAIN DEMONSTRATION ========================

def run_comprehensive_csv_demo():
    """Run comprehensive CSV demonstration."""
    
    print("PANDAS CSV OPERATIONS COMPREHENSIVE GUIDE")
    print("="*45)
    
    # Run all demonstrations
    demonstrate_csv_reading()
    demonstrate_csv_writing()
    advanced_csv_techniques()
    csv_best_practices_guide()
    
    # Cleanup demo files
    demo_files = [
        'sample_data.csv', 'sample_semicolon.csv', 'output_basic.csv',
        'output_formatted.csv', 'high_earners.csv', 'summary.csv',
        'incremental.csv', 'large_output.csv', 'compressed_output.csv.gz'
    ]
    
    print(f"\n=== CLEANUP ===")
    for file in demo_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"Removed {file}")

# Execute demonstration
if __name__ == "__main__":
    run_comprehensive_csv_demo()
```

#### Explanation
1. **Comprehensive Reading**: Multiple approaches for reading CSV files with various parameters and error handling
2. **Flexible Writing**: Different writing scenarios including formatting, appending, and chunked processing
3. **Advanced Techniques**: Dynamic type inference, encoding detection, and configuration-driven processing
4. **Error Handling**: Robust error handling for common CSV issues
5. **Performance Optimization**: Memory-efficient techniques for large files

#### Use Cases
- **Data Import/Export**: Standard data exchange between systems
- **ETL Pipelines**: Reading source data and writing processed results  
- **Data Analysis**: Loading datasets for exploratory analysis
- **Report Generation**: Exporting analysis results for sharing
- **Data Backup**: Creating portable data archives

#### Best Practices
- **Type Specification**: Always specify dtypes for better performance and data integrity
- **Encoding Awareness**: Handle different encodings properly to avoid data corruption
- **Memory Management**: Use chunking for large files to avoid memory issues
- **Error Handling**: Implement robust error handling for production environments
- **Data Validation**: Validate data quality after reading

#### Pitfalls
- **Memory Overflow**: Large files can exhaust system memory without chunking
- **Encoding Issues**: Incorrect encoding can corrupt non-ASCII characters
- **Type Inference**: Automatic type inference can be slow and inaccurate
- **Data Loss**: Improper handling of missing values can lead to data loss
- **Performance Issues**: Reading all columns when only few are needed

#### Debugging
```python
def debug_csv_issues(filename: str):
    """Debug common CSV reading issues."""
    
    print(f"Debugging CSV file: {filename}")
    
    # Check if file exists
    if not os.path.exists(filename):
        print("❌ File does not exist")
        return
    
    # Check file size
    file_size = os.path.getsize(filename) / 1024**2  # MB
    print(f"📊 File size: {file_size:.2f} MB")
    
    # Check encoding
    import chardet
    with open(filename, 'rb') as f:
        raw_data = f.read(10000)
        encoding = chardet.detect(raw_data)
    print(f"🔤 Detected encoding: {encoding}")
    
    # Try to read first few lines
    try:
        sample = pd.read_csv(filename, nrows=5)
        print(f"✅ Successfully read first 5 rows")
        print(f"📋 Columns: {list(sample.columns)}")
        print(f"🔢 Data types: {dict(sample.dtypes)}")
    except Exception as e:
        print(f"❌ Error reading file: {e}")

def optimize_csv_performance(filename: str) -> Dict[str, Any]:
    """Analyze and provide optimization recommendations."""
    
    recommendations = {
        'dtype_optimization': {},
        'memory_savings': 0,
        'performance_tips': []
    }
    
    # Sample the data
    sample_df = pd.read_csv(filename, nrows=1000)
    
    # Analyze data types
    for col in sample_df.columns:
        current_dtype = sample_df[col].dtype
        
        if current_dtype == 'object':
            # Check if it can be categorical
            unique_ratio = sample_df[col].nunique() / len(sample_df[col])
            if unique_ratio < 0.5:
                recommendations['dtype_optimization'][col] = 'category'
                recommendations['performance_tips'].append(f"Convert {col} to categorical")
        
        elif current_dtype == 'int64':
            # Check if it can be downcasted
            min_val, max_val = sample_df[col].min(), sample_df[col].max()
            if min_val >= 0 and max_val < 256:
                recommendations['dtype_optimization'][col] = 'uint8'
    
    return recommendations
```

#### Optimization

**CSV Performance Quick Reference:**

| Scenario | Read Optimization | Write Optimization |
|----------|------------------|-------------------|
| **Large Files** | Use `chunksize`, specify `dtype` | Use `mode='a'` for appending |
| **Memory Limited** | Read specific columns with `usecols` | Write in chunks |
| **Repeated Access** | Cache with `pickle` or use `Parquet` | Use compression |
| **Mixed Data Types** | Specify `dtype` explicitly | Format columns appropriately |
| **International Data** | Specify `encoding` | Use `encoding='utf-8'` |

**Common Parameters Quick Guide:**
- `dtype`: Specify data types for better performance
- `parse_dates`: Automatically parse date columns  
- `usecols`: Read only needed columns
- `chunksize`: Process large files in chunks
- `encoding`: Handle international characters
- `na_values`: Define custom missing value indicators

---

## Question 2

**How do you handlemissing datain aDataFrame?**

### Answer

#### Theory
Missing data is a common challenge in data analysis that can significantly impact the quality of insights and model performance. Pandas provides comprehensive tools for detecting, handling, and analyzing missing data patterns. Understanding these methods is crucial for maintaining data integrity and making informed decisions about data imputation strategies. Missing values in Pandas are represented as `NaN` (Not a Number), `None`, or `NaT` (Not a Time) for datetime objects.

#### Code Example

```python
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ======================== MISSING DATA DETECTION ========================

def demonstrate_missing_data_detection():
    """Comprehensive demonstration of missing data detection techniques."""
    
    print("=== MISSING DATA DETECTION TECHNIQUES ===")
    
    # Create sample data with various missing patterns
    sample_data = create_missing_data_sample()
    print(f"Sample data shape: {sample_data.shape}")
    
    # 1. Basic missing data detection
    print("\n1. Basic Missing Data Detection:")
    
    # Check for any missing values
    has_missing = sample_data.isnull().any().any()
    print(f"   Dataset has missing values: {has_missing}")
    
    # Count missing values per column
    missing_counts = sample_data.isnull().sum()
    print(f"   Missing values per column:")
    for col, count in missing_counts.items():
        if count > 0:
            percentage = (count / len(sample_data)) * 100
            print(f"     {col}: {count} ({percentage:.1f}%)")
    
    # Check missing values per row
    missing_per_row = sample_data.isnull().sum(axis=1)
    rows_with_missing = (missing_per_row > 0).sum()
    print(f"   Rows with missing values: {rows_with_missing}")
    
    # 2. Advanced missing data analysis
    print("\n2. Advanced Missing Data Analysis:")
    
    def analyze_missing_patterns(df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive missing data pattern analysis."""
        
        analysis = {
            'total_missing': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / df.size) * 100,
            'columns_with_missing': df.columns[df.isnull().any()].tolist(),
            'complete_rows': len(df.dropna()),
            'missing_patterns': {},
            'correlation_with_missing': {}
        }
        
        # Analyze missing patterns
        missing_df = df.isnull()
        pattern_counts = missing_df.value_counts()
        analysis['missing_patterns'] = pattern_counts.head(10).to_dict()
        
        # Correlation between missing values
        if len(analysis['columns_with_missing']) > 1:
            missing_corr = missing_df[analysis['columns_with_missing']].astype(int).corr()
            analysis['correlation_with_missing'] = missing_corr.to_dict()
        
        return analysis
    
    missing_analysis = analyze_missing_patterns(sample_data)
    print(f"   Total missing values: {missing_analysis['total_missing']}")
    print(f"   Missing percentage: {missing_analysis['missing_percentage']:.2f}%")
    print(f"   Complete rows: {missing_analysis['complete_rows']}")
    
    # 3. Visualizing missing data patterns
    print("\n3. Missing Data Visualization:")
    
    def visualize_missing_data(df: pd.DataFrame):
        """Create visualizations for missing data patterns."""
        
        import matplotlib.pyplot as plt
        
        # Calculate missing percentages
        missing_pct = (df.isnull().sum() / len(df)) * 100
        missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
        
        if len(missing_pct) > 0:
            print(f"   Created missing data visualization for {len(missing_pct)} columns")
            
            # Print the missing percentages instead of plotting
            print("   Missing data percentages by column:")
            for col, pct in missing_pct.items():
                print(f"     {col}: {pct:.1f}%")
        else:
            print("   No missing data to visualize")
    
    visualize_missing_data(sample_data)

# ======================== MISSING DATA HANDLING STRATEGIES ========================

def demonstrate_missing_data_strategies():
    """Demonstrate various strategies for handling missing data."""
    
    print("\n=== MISSING DATA HANDLING STRATEGIES ===")
    
    # Create sample data for demonstrations
    df_sample = create_missing_data_sample()
    
    # 1. Removing missing data
    print("\n1. Removing Missing Data:")
    
    # Drop rows with any missing values
    df_dropna_any = df_sample.dropna()
    print(f"   Original shape: {df_sample.shape}")
    print(f"   After dropping rows with any NaN: {df_dropna_any.shape}")
    
    # Drop rows with all missing values
    df_dropna_all = df_sample.dropna(how='all')
    print(f"   After dropping rows with all NaN: {df_dropna_all.shape}")
    
    # Drop columns with high missing percentage
    threshold = 0.5  # 50% missing threshold
    df_drop_cols = df_sample.dropna(axis=1, thresh=int(threshold * len(df_sample)))
    print(f"   After dropping columns with >50% missing: {df_drop_cols.shape}")
    
    # Drop rows with missing values in specific columns
    critical_columns = ['age', 'salary']
    df_drop_subset = df_sample.dropna(subset=critical_columns)
    print(f"   After dropping rows missing critical columns: {df_drop_subset.shape}")
    
    # 2. Forward and backward filling
    print("\n2. Forward and Backward Filling:")
    
    # Create time series data for demonstration
    dates = pd.date_range('2023-01-01', periods=20, freq='D')
    ts_data = pd.DataFrame({
        'date': dates,
        'value': [1, 2, np.nan, 4, 5, np.nan, np.nan, 8, 9, 10,
                 np.nan, 12, 13, np.nan, 15, 16, np.nan, np.nan, 19, 20]
    })
    
    # Forward fill (propagate last valid observation)
    ts_ffill = ts_data.copy()
    ts_ffill['value_ffill'] = ts_data['value'].fillna(method='ffill')
    
    # Backward fill (use next valid observation)
    ts_bfill = ts_data.copy()
    ts_bfill['value_bfill'] = ts_data['value'].fillna(method='bfill')
    
    # Combined forward and backward fill
    ts_combined = ts_data.copy()
    ts_combined['value_combined'] = ts_data['value'].fillna(method='ffill').fillna(method='bfill')
    
    print(f"   Original missing count: {ts_data['value'].isnull().sum()}")
    print(f"   After forward fill: {ts_ffill['value_ffill'].isnull().sum()}")
    print(f"   After backward fill: {ts_bfill['value_bfill'].isnull().sum()}")
    print(f"   After combined fill: {ts_combined['value_combined'].isnull().sum()}")
    
    # 3. Statistical imputation methods
    print("\n3. Statistical Imputation Methods:")
    
    def statistical_imputation_demo(df: pd.DataFrame):
        """Demonstrate various statistical imputation methods."""
        
        df_stats = df.copy()
        
        # Mean imputation for numerical columns
        numerical_cols = df_stats.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df_stats[col].isnull().any():
                mean_value = df_stats[col].mean()
                df_stats[f'{col}_mean_imputed'] = df_stats[col].fillna(mean_value)
                print(f"   {col}: Mean imputation with value {mean_value:.2f}")
        
        # Median imputation (more robust to outliers)
        for col in numerical_cols:
            if df_stats[col].isnull().any():
                median_value = df_stats[col].median()
                df_stats[f'{col}_median_imputed'] = df_stats[col].fillna(median_value)
                print(f"   {col}: Median imputation with value {median_value:.2f}")
        
        # Mode imputation for categorical columns
        categorical_cols = df_stats.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if df_stats[col].isnull().any():
                mode_value = df_stats[col].mode()[0] if not df_stats[col].mode().empty else 'Unknown'
                df_stats[f'{col}_mode_imputed'] = df_stats[col].fillna(mode_value)
                print(f"   {col}: Mode imputation with value '{mode_value}'")
        
        return df_stats
    
    df_imputed = statistical_imputation_demo(df_sample)
    
    # 4. Advanced imputation techniques
    print("\n4. Advanced Imputation Techniques:")
    
    def advanced_imputation_methods(df: pd.DataFrame):
        """Demonstrate advanced imputation methods."""
        
        df_advanced = df.copy()
        
        # Interpolation for time series or ordered data
        if 'age' in df.columns:
            df_advanced['age_interpolated'] = df['age'].interpolate(method='linear')
            missing_before = df['age'].isnull().sum()
            missing_after = df_advanced['age_interpolated'].isnull().sum()
            print(f"   Linear interpolation - Age: {missing_before} → {missing_after} missing")
        
        # Group-based imputation
        if 'department' in df.columns and 'salary' in df.columns:
            # Impute salary based on department median
            dept_salary_median = df.groupby('department')['salary'].median()
            
            def impute_by_group(row):
                if pd.isna(row['salary']) and row['department'] in dept_salary_median:
                    return dept_salary_median[row['department']]
                return row['salary']
            
            df_advanced['salary_group_imputed'] = df_advanced.apply(impute_by_group, axis=1)
            print(f"   Group-based imputation completed for salary by department")
        
        # Regression-based imputation
        if len(df.select_dtypes(include=[np.number]).columns) >= 2:
            from sklearn.linear_model import LinearRegression
            from sklearn.impute import IterativeImputer
            
            # Use iterative imputation (MICE-like approach)
            numeric_data = df.select_dtypes(include=[np.number])
            
            if numeric_data.shape[1] >= 2:
                imputer = IterativeImputer(random_state=42, max_iter=10)
                imputed_values = imputer.fit_transform(numeric_data)
                
                for i, col in enumerate(numeric_data.columns):
                    df_advanced[f'{col}_iterative_imputed'] = imputed_values[:, i]
                
                print(f"   Iterative imputation completed for {numeric_data.shape[1]} numeric columns")
        
        return df_advanced
    
    df_advanced_imputed = advanced_imputation_methods(df_sample)

# ======================== MISSING DATA HANDLER CLASS ========================

class MissingDataHandler:
    """Comprehensive missing data analysis and handling toolkit."""
    
    def __init__(self, data: pd.DataFrame):
        """Initialize with dataset."""
        self.data = data.copy()
        self.original_shape = data.shape
        self.missing_summary = None
        
    def analyze_missing_patterns(self) -> Dict[str, Any]:
        """Comprehensive missing data pattern analysis."""
        
        missing_info = {
            'total_missing': self.data.isnull().sum().sum(),
            'missing_percentage': (self.data.isnull().sum().sum() / self.data.size) * 100,
            'columns_missing': {},
            'rows_missing': {},
            'missing_patterns': {},
            'recommendations': []
        }
        
        # Column-wise analysis
        for col in self.data.columns:
            col_missing = self.data[col].isnull().sum()
            if col_missing > 0:
                missing_info['columns_missing'][col] = {
                    'count': col_missing,
                    'percentage': (col_missing / len(self.data)) * 100,
                    'dtype': str(self.data[col].dtype)
                }
        
        # Row-wise analysis
        rows_with_missing = self.data.isnull().any(axis=1).sum()
        missing_info['rows_missing'] = {
            'count': rows_with_missing,
            'percentage': (rows_with_missing / len(self.data)) * 100
        }
        
        # Generate recommendations
        self._generate_recommendations(missing_info)
        
        self.missing_summary = missing_info
        return missing_info
    
    def _generate_recommendations(self, missing_info: Dict[str, Any]):
        """Generate recommendations for handling missing data."""
        
        recommendations = []
        
        # Overall missing data recommendations
        total_missing_pct = missing_info['missing_percentage']
        
        if total_missing_pct < 5:
            recommendations.append("Low missing data rate - consider listwise deletion")
        elif total_missing_pct < 15:
            recommendations.append("Moderate missing data - use imputation methods")
        else:
            recommendations.append("High missing data rate - investigate data collection process")
        
        # Column-specific recommendations
        for col, info in missing_info['columns_missing'].items():
            col_missing_pct = info['percentage']
            
            if col_missing_pct > 50:
                recommendations.append(f"Consider dropping column '{col}' (>{col_missing_pct:.1f}% missing)")
            elif col_missing_pct > 20:
                recommendations.append(f"Use advanced imputation for '{col}' ({col_missing_pct:.1f}% missing)")
            elif info['dtype'] in ['object', 'category']:
                recommendations.append(f"Use mode imputation for categorical '{col}'")
            else:
                recommendations.append(f"Use median/mean imputation for numeric '{col}'")
        
        missing_info['recommendations'] = recommendations
    
    def handle_missing_data(self, strategy: str = 'auto', **kwargs):
        """Apply missing data handling strategy."""
        
        if strategy == 'auto':
            if self.missing_summary is None:
                self.analyze_missing_patterns()
            
            # Auto-select strategy based on missing data patterns
            total_missing_pct = self.missing_summary['missing_percentage']
            
            if total_missing_pct < 5:
                return self._apply_deletion()
            else:
                return self._apply_imputation(**kwargs)
        
        elif strategy == 'deletion':
            return self._apply_deletion(**kwargs)
        elif strategy == 'imputation':
            return self._apply_imputation(**kwargs)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _apply_deletion(self, threshold: float = 0.5):
        """Apply deletion-based missing data handling."""
        
        df_cleaned = self.data.copy()
        
        # Drop columns with high missing percentage
        col_missing_pct = df_cleaned.isnull().sum() / len(df_cleaned)
        cols_to_drop = col_missing_pct[col_missing_pct > threshold].index
        df_cleaned = df_cleaned.drop(columns=cols_to_drop)
        
        # Drop rows with any missing values in remaining columns
        df_cleaned = df_cleaned.dropna()
        
        return df_cleaned
    
    def _apply_imputation(self, method: str = 'mixed'):
        """Apply imputation-based missing data handling."""
        
        df_imputed = self.data.copy()
        
        if method == 'mixed':
            # Use appropriate imputation for each column type
            
            # Numeric columns: median imputation
            numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df_imputed[col].isnull().any():
                    df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())
            
            # Categorical columns: mode imputation
            categorical_cols = df_imputed.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                if df_imputed[col].isnull().any():
                    mode_val = df_imputed[col].mode()
                    if not mode_val.empty:
                        df_imputed[col] = df_imputed[col].fillna(mode_val[0])
        
        return df_imputed

# ======================== HELPER FUNCTIONS ========================

def create_missing_data_sample():
    """Create sample dataset with various missing data patterns."""
    
    np.random.seed(42)
    
    n_samples = 200
    
    # Create base data
    data = {
        'id': range(1, n_samples + 1),
        'name': [f'Person_{i}' for i in range(1, n_samples + 1)],
        'age': np.random.randint(20, 70, n_samples),
        'salary': np.random.normal(70000, 20000, n_samples),
        'department': np.random.choice(['Engineering', 'Marketing', 'Sales', 'HR'], n_samples),
        'experience': np.random.randint(0, 30, n_samples),
        'performance_score': np.random.uniform(1, 5, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Introduce various missing patterns
    
    # Random missing values
    df.loc[np.random.choice(df.index, 20, replace=False), 'age'] = np.nan
    df.loc[np.random.choice(df.index, 15, replace=False), 'salary'] = np.nan
    
    # Missing values correlated with other variables
    # Higher chance of missing performance score for new employees
    new_employees = df[df['experience'] < 2].index
    missing_performance = np.random.choice(new_employees, min(10, len(new_employees)), replace=False)
    df.loc[missing_performance, 'performance_score'] = np.nan
    
    # Completely missing rows (rare but possible)
    completely_missing_rows = np.random.choice(df.index, 3, replace=False)
    df.loc[completely_missing_rows, ['age', 'salary', 'performance_score']] = np.nan
    
    # Missing values in groups
    hr_employees = df[df['department'] == 'HR'].index
    if len(hr_employees) > 0:
        missing_hr_salary = np.random.choice(hr_employees, min(5, len(hr_employees)), replace=False)
        df.loc[missing_hr_salary, 'salary'] = np.nan
    
    return df

# ======================== DEMONSTRATION EXAMPLES ========================

def run_missing_data_demo():
    """Run comprehensive missing data handling demonstration."""
    
    print("PANDAS MISSING DATA HANDLING GUIDE")
    print("="*35)
    
    # Create sample data
    df = create_missing_data_sample()
    print(f"Sample dataset created with shape: {df.shape}")
    
    # Basic detection
    print(f"\nMissing value summary:")
    missing_summary = df.isnull().sum()
    for col, count in missing_summary.items():
        if count > 0:
            print(f"  {col}: {count} missing values")
    
    # Demonstrate different strategies
    print(f"\n=== STRATEGY COMPARISON ===")
    
    # Strategy 1: Drop all rows with any missing values
    df_dropna = df.dropna()
    print(f"1. Drop all missing: {df.shape} → {df_dropna.shape}")
    
    # Strategy 2: Fill with median/mode
    df_filled = df.copy()
    
    # Fill numeric columns with median
    numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_filled[col].fillna(df_filled[col].median(), inplace=True)
    
    # Fill categorical columns with mode
    categorical_cols = df_filled.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        mode_val = df_filled[col].mode()
        if not mode_val.empty:
            df_filled[col].fillna(mode_val[0], inplace=True)
    
    print(f"2. Fill with median/mode: {df_filled.isnull().sum().sum()} missing values remaining")
    
    # Strategy 3: Forward fill for time series pattern
    df_ffill = df.copy()
    df_ffill = df_ffill.fillna(method='ffill')
    print(f"3. Forward fill: {df_ffill.isnull().sum().sum()} missing values remaining")
    
    # Strategy 4: Using the MissingDataHandler class
    handler = MissingDataHandler(df)
    analysis = handler.analyze_missing_patterns()
    
    print(f"\n=== AUTOMATED ANALYSIS ===")
    print(f"Total missing: {analysis['total_missing']} ({analysis['missing_percentage']:.1f}%)")
    print(f"Recommendations:")
    for rec in analysis['recommendations'][:3]:
        print(f"  • {rec}")
    
    # Apply automatic handling
    df_auto = handler.handle_missing_data(strategy='auto')
    print(f"4. Automatic handling: {df.shape} → {df_auto.shape}")

# Execute demonstration
if __name__ == "__main__":
    run_missing_data_demo()
```

#### Explanation
1. **Detection Methods**: Multiple approaches to identify missing data patterns and understand their distribution
2. **Removal Strategies**: Various deletion techniques based on rows, columns, and thresholds
3. **Filling Methods**: Forward/backward fill, statistical imputation, and interpolation
4. **Advanced Techniques**: Group-based, regression-based, and iterative imputation
5. **Automated Handling**: Intelligent strategy selection based on data characteristics

#### Use Cases
- **Data Cleaning**: Preparing datasets for analysis and modeling
- **Time Series Analysis**: Handling gaps in temporal data
- **Survey Analysis**: Managing incomplete responses and non-responses
- **Database Migration**: Cleaning data during ETL processes
- **ML Preprocessing**: Ensuring complete datasets for machine learning

#### Best Practices
- **Understand Patterns**: Analyze why data is missing before choosing strategy
- **Domain Knowledge**: Use business logic to guide imputation decisions  
- **Multiple Approaches**: Test different methods and compare results
- **Validate Results**: Assess imputation quality and impact on analysis
- **Document Process**: Record methods used for reproducibility

#### Pitfalls
- **Bias Introduction**: Improper imputation can distort relationships
- **Over-imputation**: Filling too many values creates artificial patterns
- **Wrong Method**: Using inappropriate technique for data type/context
- **Ignoring Mechanism**: Not considering MCAR, MAR, or MNAR patterns
- **No Validation**: Failing to assess imputation quality

#### Debugging
```python
def debug_missing_data(df: pd.DataFrame, column: str = None):
    """Debug missing data issues in DataFrame."""
    
    if column:
        print(f"Debugging column: {column}")
        if column not in df.columns:
            print(f"❌ Column '{column}' not found")
            return
        
        missing_count = df[column].isnull().sum()
        total_count = len(df[column])
        missing_pct = (missing_count / total_count) * 100
        
        print(f"📊 Missing: {missing_count}/{total_count} ({missing_pct:.1f}%)")
        print(f"🔢 Data type: {df[column].dtype}")
        
        if missing_count > 0:
            print(f"🔍 Missing positions: {df[df[column].isnull()].index[:5].tolist()}")
    else:
        print("Debugging entire DataFrame:")
        print(f"📊 Shape: {df.shape}")
        print(f"❓ Total missing: {df.isnull().sum().sum()}")
        print(f"📋 Missing by column:")
        
        for col in df.columns:
            missing = df[col].isnull().sum()
            if missing > 0:
                pct = (missing / len(df)) * 100
                print(f"  {col}: {missing} ({pct:.1f}%)")

def compare_imputation_methods(df: pd.DataFrame, column: str):
    """Compare different imputation methods for a column."""
    
    if column not in df.columns:
        print(f"Column '{column}' not found")
        return
    
    original_missing = df[column].isnull().sum()
    if original_missing == 0:
        print(f"No missing values in column '{column}'")
        return
    
    print(f"Comparing imputation methods for '{column}':")
    print(f"Original missing values: {original_missing}")
    
    methods = {}
    
    if df[column].dtype in ['int64', 'float64']:
        # Numeric methods
        methods['Mean'] = df[column].fillna(df[column].mean())
        methods['Median'] = df[column].fillna(df[column].median())
        methods['Forward Fill'] = df[column].fillna(method='ffill')
        methods['Interpolation'] = df[column].interpolate()
    else:
        # Categorical methods
        mode_val = df[column].mode()
        if not mode_val.empty:
            methods['Mode'] = df[column].fillna(mode_val[0])
        methods['Forward Fill'] = df[column].fillna(method='ffill')
        methods['Constant'] = df[column].fillna('Unknown')
    
    for method_name, imputed_series in methods.items():
        remaining_missing = imputed_series.isnull().sum()
        print(f"  {method_name}: {remaining_missing} missing remaining")
```

#### Optimization

**Missing Data Strategy Decision Matrix:**

| Missing % | Data Type | Recommended Strategy |
|-----------|-----------|---------------------|
| < 5% | Any | Listwise deletion |
| 5-15% | Numeric | Mean/Median imputation |
| 5-15% | Categorical | Mode imputation |
| 15-30% | Numeric | Regression/Iterative imputation |
| 15-30% | Categorical | Group-based imputation |
| > 30% | Any | Investigate data collection |

**Performance Tips:**
- Use vectorized operations for large datasets
- Consider chunked processing for memory efficiency
- Cache imputation parameters for consistency
- Validate imputation quality regularly

---

## Question 3

**How do youapplya function to all elements in aDataFramecolumn?**

### Answer

#### Theory
Applying functions to DataFrame columns is a fundamental operation in data manipulation and transformation. Pandas provides several methods to apply functions to DataFrame columns, each optimized for different scenarios. The most common methods include `apply()`, `map()`, `applymap()`, and vectorized operations. Understanding these methods and their performance characteristics is crucial for efficient data processing, especially when working with large datasets.

#### Code Example

```python
import pandas as pd
import numpy as np
from typing import Callable, Dict, List, Optional, Union, Any
import time
from functools import wraps
import warnings
warnings.filterwarnings('ignore')

# ======================== BASIC FUNCTION APPLICATION ========================

def demonstrate_basic_function_application():
    """Demonstrate basic function application methods in Pandas."""
    
    print("=== BASIC FUNCTION APPLICATION METHODS ===")
    
    # Create sample DataFrame
    df = create_sample_dataframe()
    print(f"Sample DataFrame shape: {df.shape}")
    print(f"Sample data:\n{df.head()}")
    
    # 1. Using apply() method
    print("\n1. Using apply() Method:")
    
    # Apply function to a single column
    def categorize_age(age):
        """Categorize age into groups."""
        if age < 30:
            return 'Young'
        elif age < 50:
            return 'Middle-aged'
        else:
            return 'Senior'
    
    df['age_category'] = df['age'].apply(categorize_age)
    print(f"   Age categorization applied:")
    print(f"   Categories: {df['age_category'].value_counts().to_dict()}")
    
    # Apply function with additional arguments
    def salary_adjustment(salary, adjustment_rate=0.1):
        """Apply salary adjustment."""
        return salary * (1 + adjustment_rate)
    
    df['adjusted_salary'] = df['salary'].apply(salary_adjustment, adjustment_rate=0.15)
    print(f"   Salary adjustment applied with 15% increase")
    print(f"   Average increase: ${(df['adjusted_salary'] - df['salary']).mean():.2f}")
    
    # 2. Using map() method
    print("\n2. Using map() Method:")
    
    # Map with dictionary
    department_codes = {
        'Engineering': 'ENG',
        'Marketing': 'MKT',
        'Sales': 'SAL',
        'HR': 'HR',
        'Finance': 'FIN'
    }
    
    df['dept_code'] = df['department'].map(department_codes)
    print(f"   Department codes mapped:")
    print(f"   Unique codes: {df['dept_code'].unique().tolist()}")
    
    # Map with function
    def get_name_length(name):
        """Get length of name."""
        return len(str(name))
    
    df['name_length'] = df['name'].map(get_name_length)
    print(f"   Name lengths calculated: min={df['name_length'].min()}, max={df['name_length'].max()}")
    
    # 3. Lambda functions
    print("\n3. Lambda Functions:")
    
    # Apply lambda function
    df['salary_thousands'] = df['salary'].apply(lambda x: round(x / 1000, 1))
    print(f"   Salary in thousands: {df['salary_thousands'].head().tolist()}")
    
    # Complex lambda with conditional logic
    df['performance_level'] = df['performance_rating'].apply(
        lambda x: 'Excellent' if x >= 4.5 else 'Good' if x >= 3.5 else 'Needs Improvement'
    )
    print(f"   Performance levels: {df['performance_level'].value_counts().to_dict()}")
    
    return df

# ======================== ADVANCED FUNCTION APPLICATION ========================

def demonstrate_advanced_function_application():
    """Demonstrate advanced function application techniques."""
    
    print("\n=== ADVANCED FUNCTION APPLICATION TECHNIQUES ===")
    
    df = create_sample_dataframe()
    
    # 1. Vectorized operations (fastest)
    print("\n1. Vectorized Operations:")
    
    # NumPy vectorized operations
    df['salary_log'] = np.log(df['salary'])
    df['age_squared'] = df['age'] ** 2
    df['salary_normalized'] = (df['salary'] - df['salary'].mean()) / df['salary'].std()
    
    print(f"   Vectorized operations applied:")
    print(f"   Log salary range: {df['salary_log'].min():.2f} to {df['salary_log'].max():.2f}")
    print(f"   Normalized salary mean: {df['salary_normalized'].mean():.6f}")
    
    # 2. Apply with multiple columns
    print("\n2. Apply with Multiple Columns:")
    
    def calculate_value_score(row):
        """Calculate employee value score based on multiple factors."""
        base_score = row['performance_rating'] * 20
        experience_bonus = min(row['experience'] * 2, 20)  # Cap at 20
        age_factor = 1.1 if 30 <= row['age'] <= 50 else 1.0
        
        return (base_score + experience_bonus) * age_factor
    
    df['value_score'] = df.apply(calculate_value_score, axis=1)
    print(f"   Value scores calculated:")
    print(f"   Score range: {df['value_score'].min():.1f} to {df['value_score'].max():.1f}")
    
    # 3. Conditional application
    print("\n3. Conditional Application:")
    
    def conditional_bonus(row):
        """Calculate bonus based on multiple conditions."""
        if row['department'] == 'Sales' and row['performance_rating'] > 4.0:
            return row['salary'] * 0.2  # 20% bonus for high-performing sales
        elif row['experience'] > 10 and row['performance_rating'] > 3.5:
            return row['salary'] * 0.15  # 15% bonus for experienced performers
        elif row['performance_rating'] > 4.5:
            return row['salary'] * 0.1   # 10% bonus for excellent performers
        else:
            return 0
    
    df['bonus'] = df.apply(conditional_bonus, axis=1)
    bonus_recipients = (df['bonus'] > 0).sum()
    total_bonus = df['bonus'].sum()
    print(f"   Conditional bonuses: {bonus_recipients} recipients, total ${total_bonus:,.2f}")
    
    # 4. String operations
    print("\n4. String Operations:")
    
    # Multiple string transformations
    df['name_upper'] = df['name'].str.upper()
    df['name_initials'] = df['name'].apply(lambda x: ''.join([word[0] for word in str(x).split()]))
    df['email'] = df['name'].apply(lambda x: str(x).lower().replace(' ', '.') + '@company.com')
    
    print(f"   String operations applied:")
    print(f"   Sample emails: {df['email'].head(3).tolist()}")
    print(f"   Sample initials: {df['name_initials'].head(5).tolist()}")
    
    return df

# ======================== PERFORMANCE OPTIMIZATION ========================

def demonstrate_performance_optimization():
    """Demonstrate performance optimization techniques for function application."""
    
    print("\n=== PERFORMANCE OPTIMIZATION TECHNIQUES ===")
    
    # Create larger dataset for performance testing
    large_df = create_large_dataframe(50000)
    print(f"Large dataset created: {large_df.shape}")
    
    # Performance decorator
    def time_function(func):
        """Decorator to time function execution."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"   {func.__name__}: {end_time - start_time:.4f} seconds")
            return result
        return wrapper
    
    # 1. Compare different application methods
    print("\n1. Performance Comparison:")
    
    @time_function
    def apply_method(df):
        """Use apply method."""
        return df['salary'].apply(lambda x: x * 1.1)
    
    @time_function
    def vectorized_method(df):
        """Use vectorized operation."""
        return df['salary'] * 1.1
    
    @time_function
    def map_method(df):
        """Use map method with lambda."""
        return df['salary'].map(lambda x: x * 1.1)
    
    # Test different methods
    result_apply = apply_method(large_df.copy())
    result_vectorized = vectorized_method(large_df.copy())
    result_map = map_method(large_df.copy())
    
    # Verify results are equivalent
    print(f"   Results equivalent: {np.allclose(result_apply, result_vectorized)}")
    
    # 2. Optimized categorical operations
    print("\n2. Optimized Categorical Operations:")
    
    @time_function
    def category_apply_slow(df):
        """Slow categorical operation."""
        return df['department'].apply(lambda x: x.upper() if x else 'UNKNOWN')
    
    @time_function
    def category_optimized(df):
        """Optimized categorical operation."""
        # Convert to categorical first, then use cat accessor
        cat_series = df['department'].astype('category')
        return cat_series.cat.rename_categories(lambda x: x.upper())
    
    @time_function
    def category_vectorized(df):
        """Vectorized string operation."""
        return df['department'].str.upper().fillna('UNKNOWN')
    
    # Test categorical operations
    df_copy = large_df.copy()
    result_slow = category_apply_slow(df_copy)
    result_optimized = category_optimized(df_copy)
    result_vec = category_vectorized(df_copy)
    
    # 3. Memory-efficient operations
    print("\n3. Memory-Efficient Operations:")
    
    def memory_efficient_apply(df, chunk_size=10000):
        """Apply function in chunks to manage memory."""
        
        def process_chunk(chunk):
            """Process a single chunk."""
            return chunk['salary'].apply(lambda x: np.log(x) if x > 0 else 0)
        
        results = []
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            chunk_result = process_chunk(chunk)
            results.append(chunk_result)
        
        return pd.concat(results)
    
    @time_function
    def chunked_processing(df):
        """Process in chunks."""
        return memory_efficient_apply(df, chunk_size=10000)
    
    chunked_result = chunked_processing(large_df.copy())
    print(f"   Chunked processing completed: {len(chunked_result)} results")

# ======================== SPECIALIZED APPLICATION METHODS ========================

class FunctionApplicator:
    """Advanced function application utility class."""
    
    def __init__(self, dataframe: pd.DataFrame):
        """Initialize with DataFrame."""
        self.df = dataframe.copy()
        self.transformations = {}
        
    def add_transformation(self, column: str, func: Callable, name: str = None, **kwargs):
        """Add a transformation to be applied later."""
        
        if name is None:
            name = f"{column}_{func.__name__}"
        
        self.transformations[name] = {
            'column': column,
            'function': func,
            'kwargs': kwargs,
            'applied': False
        }
        
        return self
    
    def apply_transformation(self, name: str) -> pd.Series:
        """Apply a specific transformation."""
        
        if name not in self.transformations:
            raise ValueError(f"Transformation '{name}' not found")
        
        transform = self.transformations[name]
        column = transform['column']
        func = transform['function']
        kwargs = transform['kwargs']
        
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        
        # Apply the transformation
        if kwargs:
            result = self.df[column].apply(func, **kwargs)
        else:
            result = self.df[column].apply(func)
        
        transform['applied'] = True
        return result
    
    def apply_all_transformations(self) -> pd.DataFrame:
        """Apply all registered transformations."""
        
        result_df = self.df.copy()
        
        for name, transform in self.transformations.items():
            try:
                result_df[name] = self.apply_transformation(name)
                print(f"✅ Applied transformation: {name}")
            except Exception as e:
                print(f"❌ Failed to apply {name}: {str(e)}")
        
        return result_df
    
    def batch_apply(self, column: str, functions: Dict[str, Callable]) -> pd.DataFrame:
        """Apply multiple functions to a single column."""
        
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found")
        
        result_df = self.df.copy()
        
        for func_name, func in functions.items():
            try:
                new_col_name = f"{column}_{func_name}"
                result_df[new_col_name] = self.df[column].apply(func)
                print(f"✅ Applied {func_name} to {column}")
            except Exception as e:
                print(f"❌ Failed to apply {func_name}: {str(e)}")
        
        return result_df
    
    def conditional_apply(self, column: str, condition_func: Callable, 
                         true_func: Callable, false_func: Callable = None) -> pd.Series:
        """Apply function conditionally based on a condition."""
        
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found")
        
        def conditional_logic(value):
            if condition_func(value):
                return true_func(value)
            elif false_func:
                return false_func(value)
            else:
                return value
        
        return self.df[column].apply(conditional_logic)

# ======================== HELPER FUNCTIONS ========================

def create_sample_dataframe():
    """Create sample DataFrame for demonstrations."""
    
    np.random.seed(42)
    
    n_samples = 1000
    
    departments = ['Engineering', 'Marketing', 'Sales', 'HR', 'Finance']
    
    data = {
        'id': range(1, n_samples + 1),
        'name': [f'Employee_{i}' for i in range(1, n_samples + 1)],
        'age': np.random.randint(22, 65, n_samples),
        'salary': np.random.normal(75000, 20000, n_samples).clip(min=30000),
        'department': np.random.choice(departments, n_samples),
        'experience': np.random.randint(0, 20, n_samples),
        'performance_rating': np.random.uniform(2.0, 5.0, n_samples).round(1)
    }
    
    return pd.DataFrame(data)

def create_large_dataframe(size):
    """Create large DataFrame for performance testing."""
    
    np.random.seed(123)
    
    departments = ['Engineering', 'Marketing', 'Sales', 'HR', 'Finance']
    
    data = {
        'id': range(1, size + 1),
        'salary': np.random.normal(75000, 20000, size).clip(min=30000),
        'department': np.random.choice(departments, size),
        'performance_rating': np.random.uniform(2.0, 5.0, size)
    }
    
    return pd.DataFrame(data)

# ======================== PRACTICAL EXAMPLES ========================

def practical_function_applications():
    """Demonstrate practical real-world function applications."""
    
    print("\n=== PRACTICAL FUNCTION APPLICATIONS ===")
    
    df = create_sample_dataframe()
    
    # 1. Data validation functions
    print("\n1. Data Validation Functions:")
    
    def validate_salary(salary):
        """Validate salary is within reasonable range."""
        if pd.isna(salary):
            return 'Missing'
        elif salary < 20000:
            return 'Too Low'
        elif salary > 200000:
            return 'Too High'
        else:
            return 'Valid'
    
    df['salary_validation'] = df['salary'].apply(validate_salary)
    validation_counts = df['salary_validation'].value_counts()
    print(f"   Salary validation: {validation_counts.to_dict()}")
    
    # 2. Complex business logic
    print("\n2. Complex Business Logic:")
    
    def calculate_promotion_eligibility(row):
        """Calculate promotion eligibility based on multiple factors."""
        score = 0
        
        # Performance factor
        if row['performance_rating'] >= 4.5:
            score += 40
        elif row['performance_rating'] >= 4.0:
            score += 30
        elif row['performance_rating'] >= 3.5:
            score += 20
        
        # Experience factor
        if row['experience'] >= 5:
            score += 30
        elif row['experience'] >= 3:
            score += 20
        elif row['experience'] >= 1:
            score += 10
        
        # Age factor (avoid discrimination, focus on experience)
        if row['experience'] > 10:
            score += 10
        
        # Department factor
        if row['department'] in ['Engineering', 'Sales']:
            score += 10
        
        return score
    
    df['promotion_score'] = df.apply(calculate_promotion_eligibility, axis=1)
    df['promotion_eligible'] = df['promotion_score'].apply(lambda x: x >= 70)
    
    eligible_count = df['promotion_eligible'].sum()
    avg_score = df['promotion_score'].mean()
    print(f"   Promotion eligible: {eligible_count}/{len(df)} employees")
    print(f"   Average promotion score: {avg_score:.1f}")
    
    # 3. Text processing functions
    print("\n3. Text Processing Functions:")
    
    def standardize_name(name):
        """Standardize name formatting."""
        return str(name).title().strip()
    
    def generate_username(name):
        """Generate username from name."""
        clean_name = str(name).lower().replace(' ', '.')
        return clean_name.replace('_', '.')
    
    df['name_standardized'] = df['name'].apply(standardize_name)
    df['username'] = df['name'].apply(generate_username)
    
    print(f"   Text processing applied:")
    print(f"   Sample usernames: {df['username'].head(3).tolist()}")
    
    # 4. Financial calculations
    print("\n4. Financial Calculations:")
    
    def calculate_tax_bracket(salary):
        """Calculate tax bracket based on salary."""
        if salary <= 40000:
            return '15%'
        elif salary <= 80000:
            return '25%'
        elif salary <= 120000:
            return '28%'
        else:
            return '33%'
    
    def estimate_net_salary(salary):
        """Estimate net salary after taxes and deductions."""
        # Simplified calculation
        if salary <= 40000:
            tax_rate = 0.15
        elif salary <= 80000:
            tax_rate = 0.25
        elif salary <= 120000:
            tax_rate = 0.28
        else:
            tax_rate = 0.33
        
        # Additional deductions (health insurance, retirement, etc.)
        deductions = min(salary * 0.1, 10000)  # Cap at $10k
        
        return salary * (1 - tax_rate) - deductions
    
    df['tax_bracket'] = df['salary'].apply(calculate_tax_bracket)
    df['estimated_net_salary'] = df['salary'].apply(estimate_net_salary)
    
    tax_distribution = df['tax_bracket'].value_counts()
    avg_net_salary = df['estimated_net_salary'].mean()
    print(f"   Tax brackets: {tax_distribution.to_dict()}")
    print(f"   Average net salary: ${avg_net_salary:,.2f}")

# ======================== MAIN DEMONSTRATION ========================

def run_comprehensive_function_application_demo():
    """Run comprehensive function application demonstration."""
    
    print("PANDAS FUNCTION APPLICATION COMPREHENSIVE GUIDE")
    print("="*50)
    
    # Run all demonstrations
    df1 = demonstrate_basic_function_application()
    df2 = demonstrate_advanced_function_application()
    demonstrate_performance_optimization()
    
    # Demonstrate the FunctionApplicator class
    print("\n=== FUNCTION APPLICATOR CLASS DEMONSTRATION ===")
    
    sample_df = create_sample_dataframe()
    applicator = FunctionApplicator(sample_df)
    
    # Add transformations
    applicator.add_transformation('salary', lambda x: x * 1.1, 'salary_increased')
    applicator.add_transformation('age', lambda x: 'Young' if x < 30 else 'Senior', 'age_group')
    applicator.add_transformation('performance_rating', lambda x: x * 2, 'performance_scaled')
    
    # Apply all transformations
    result_df = applicator.apply_all_transformations()
    print(f"Applied transformations: {list(applicator.transformations.keys())}")
    
    # Batch apply multiple functions
    math_functions = {
        'squared': lambda x: x ** 2,
        'sqrt': lambda x: np.sqrt(x),
        'log': lambda x: np.log(x) if x > 0 else 0
    }
    
    batch_result = applicator.batch_apply('salary', math_functions)
    print(f"Batch applied math functions to salary column")
    
    # Conditional apply
    conditional_result = applicator.conditional_apply(
        'performance_rating',
        condition_func=lambda x: x >= 4.0,
        true_func=lambda x: 'High Performer',
        false_func=lambda x: 'Regular Performer'
    )
    print(f"Conditional apply: {conditional_result.value_counts().to_dict()}")
    
    # Practical examples
    practical_function_applications()

# Execute demonstration
if __name__ == "__main__":
    run_comprehensive_function_application_demo()
```

#### Explanation
1. **Apply Method**: Most versatile method for applying functions to Series/DataFrame
2. **Map Method**: Efficient for dictionary mapping and simple transformations
3. **Vectorized Operations**: Fastest approach using NumPy operations
4. **Lambda Functions**: Concise inline functions for simple transformations
5. **Complex Logic**: Multi-column operations and conditional applications
6. **Performance Optimization**: Techniques for handling large datasets efficiently

#### Use Cases
- **Data Transformation**: Converting data formats and types
- **Feature Engineering**: Creating new features from existing columns
- **Data Validation**: Applying business rules and validation logic
- **Text Processing**: Standardizing and cleaning text data
- **Mathematical Operations**: Complex calculations across columns

#### Best Practices
- **Vectorization First**: Use vectorized operations when possible for best performance
- **Appropriate Method**: Choose apply(), map(), or vectorized based on use case
- **Function Reusability**: Create reusable functions for common transformations
- **Error Handling**: Include error handling in custom functions
- **Memory Management**: Use chunked processing for large datasets

#### Pitfalls
- **Performance Issues**: Using apply() when vectorized operations are available
- **Memory Consumption**: Applying memory-intensive functions to large datasets
- **Type Errors**: Not handling different data types properly in functions
- **Side Effects**: Functions that modify global state or have side effects
- **Null Values**: Not handling NaN values in custom functions

#### Debugging
```python
def debug_function_application(df: pd.DataFrame, column: str, func: Callable):
    """Debug function application issues."""
    
    print(f"Debugging function application on column: {column}")
    
    if column not in df.columns:
        print(f"❌ Column '{column}' not found")
        return
    
    # Check column info
    print(f"📊 Column info:")
    print(f"   Data type: {df[column].dtype}")
    print(f"   Non-null values: {df[column].count()}/{len(df[column])}")
    print(f"   Unique values: {df[column].nunique()}")
    
    # Test function on sample values
    print(f"🧪 Testing function on sample values:")
    sample_values = df[column].dropna().head(5)
    
    for i, value in enumerate(sample_values):
        try:
            result = func(value)
            print(f"   {i+1}. {value} → {result}")
        except Exception as e:
            print(f"   {i+1}. {value} → ERROR: {str(e)}")
    
    # Check for null values
    null_count = df[column].isnull().sum()
    if null_count > 0:
        print(f"⚠️ Warning: {null_count} null values in column")
        print("   Consider handling null values in your function")

def benchmark_function_methods(df: pd.DataFrame, column: str):
    """Benchmark different function application methods."""
    
    print(f"Benchmarking function application methods for column: {column}")
    
    # Simple transformation function
    def simple_transform(x):
        return x * 1.1
    
    methods = {
        'apply': lambda: df[column].apply(simple_transform),
        'map': lambda: df[column].map(simple_transform),
        'vectorized': lambda: df[column] * 1.1
    }
    
    results = {}
    
    for method_name, method_func in methods.items():
        start_time = time.time()
        try:
            result = method_func()
            end_time = time.time()
            results[method_name] = {
                'time': end_time - start_time,
                'success': True,
                'result_length': len(result)
            }
        except Exception as e:
            results[method_name] = {
                'time': None,
                'success': False,
                'error': str(e)
            }
    
    # Print results
    print("Performance comparison:")
    for method, info in results.items():
        if info['success']:
            print(f"  {method}: {info['time']:.4f} seconds")
        else:
            print(f"  {method}: FAILED - {info['error']}")
```

#### Optimization

**Function Application Method Selection:**

| Use Case | Recommended Method | Performance |
|----------|-------------------|-------------|
| **Simple Math** | Vectorized operations | Fastest |
| **Dictionary Mapping** | map() | Fast |
| **Complex Logic** | apply() | Moderate |
| **Multiple Columns** | apply(axis=1) | Slower |
| **Text Operations** | str accessor + vectorized | Fast |

**Performance Tips:**
- Use vectorized operations whenever possible
- Avoid apply() for simple mathematical operations
- Use categorical data types for repetitive string operations
- Consider chunked processing for very large datasets
- Cache expensive computations when applying to multiple columns

---

## Question 4

**Demonstrate how to handleduplicate rowsin aDataFrame.**

### Answer

#### Theory
Duplicate rows are a common data quality issue that can skew analysis results and impact model performance. Pandas provides robust tools for detecting, analyzing, and handling duplicates through methods like `duplicated()`, `drop_duplicates()`, and `value_counts()`. Understanding duplicate patterns helps maintain data integrity and ensures accurate analysis.

#### Code Example

```python
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union

# ======================== DUPLICATE DETECTION & HANDLING ========================

def comprehensive_duplicate_handling():
    """Comprehensive guide to handling duplicate rows in DataFrames."""
    
    print("=== DUPLICATE ROW HANDLING GUIDE ===")
    
    # Create sample data with duplicates
    df = create_sample_with_duplicates()
    print(f"Sample data shape: {df.shape}")
    
    # 1. Detecting duplicates
    print("\n1. DETECTING DUPLICATES:")
    
    # Check for any duplicates
    has_duplicates = df.duplicated().any()
    print(f"   Dataset has duplicates: {has_duplicates}")
    
    # Count total duplicates
    duplicate_count = df.duplicated().sum()
    print(f"   Total duplicate rows: {duplicate_count}")
    
    # Identify duplicate rows
    duplicate_rows = df[df.duplicated()]
    print(f"   Duplicate rows shape: {duplicate_rows.shape}")
    
    # Check duplicates based on specific columns
    name_duplicates = df.duplicated(subset=['name']).sum()
    email_duplicates = df.duplicated(subset=['email']).sum()
    print(f"   Name duplicates: {name_duplicates}")
    print(f"   Email duplicates: {email_duplicates}")
    
    # 2. Removing duplicates
    print("\n2. REMOVING DUPLICATES:")
    
    # Remove all duplicates (keep first occurrence)
    df_no_dups = df.drop_duplicates()
    print(f"   After removing duplicates: {df.shape} → {df_no_dups.shape}")
    
    # Keep last occurrence instead
    df_keep_last = df.drop_duplicates(keep='last')
    print(f"   Keeping last occurrence: {df_keep_last.shape}")
    
    # Remove duplicates based on specific columns
    df_unique_names = df.drop_duplicates(subset=['name'])
    print(f"   Unique names only: {df_unique_names.shape}")
    
    # 3. Advanced duplicate analysis
    print("\n3. ADVANCED DUPLICATE ANALYSIS:")
    
    def analyze_duplicate_patterns(df):
        """Analyze patterns in duplicate data."""
        
        # Group by all columns to find exact duplicates
        exact_duplicates = df.groupby(list(df.columns)).size()
        exact_dups_multiple = exact_duplicates[exact_duplicates > 1]
        
        print(f"   Exact duplicate groups: {len(exact_dups_multiple)}")
        
        # Analyze duplicates by key columns
        key_columns = ['name', 'email']
        for col in key_columns:
            if col in df.columns:
                col_value_counts = df[col].value_counts()
                duplicated_values = col_value_counts[col_value_counts > 1]
                print(f"   {col} with duplicates: {len(duplicated_values)}")
        
        return exact_dups_multiple
    
    duplicate_analysis = analyze_duplicate_patterns(df)
    
    # 4. Smart duplicate removal strategies
    print("\n4. SMART DUPLICATE REMOVAL:")
    
    def smart_duplicate_removal(df):
        """Remove duplicates with intelligent priority."""
        
        # Strategy 1: Keep most complete record (fewer NaN values)
        def keep_most_complete(group):
            """Keep record with fewest missing values."""
            missing_counts = group.isnull().sum(axis=1)
            return group.loc[missing_counts.idxmin()]
        
        # Strategy 2: Keep most recent record (if date column exists)
        if 'date_created' in df.columns:
            df_smart = df.sort_values('date_created').drop_duplicates(
                subset=['name', 'email'], keep='last'
            )
        else:
            # Use completeness strategy
            df_smart = df.groupby(['name', 'email']).apply(keep_most_complete).reset_index(drop=True)
        
        return df_smart
    
    df_smart_cleaned = smart_duplicate_removal(df)
    print(f"   Smart duplicate removal: {df.shape} → {df_smart_cleaned.shape}")
    
    return df_no_dups

def create_sample_with_duplicates():
    """Create sample DataFrame with various duplicate patterns."""
    
    np.random.seed(42)
    
    # Base data
    names = ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie Wilson']
    departments = ['Engineering', 'Marketing', 'Sales']
    
    data = []
    
    # Add original records
    for i in range(20):
        record = {
            'id': i + 1,
            'name': np.random.choice(names),
            'email': f"user{i+1}@company.com",
            'department': np.random.choice(departments),
            'salary': np.random.randint(50000, 100000),
            'date_created': pd.date_range('2023-01-01', periods=20)[i]
        }
        data.append(record)
    
    # Add exact duplicates
    data.append(data[0].copy())  # Exact duplicate of first record
    data.append(data[1].copy())  # Exact duplicate of second record
    
    # Add partial duplicates (same name/email but different other fields)
    partial_dup = data[2].copy()
    partial_dup['salary'] = 75000  # Different salary
    data.append(partial_dup)
    
    # Add records with same name but different email
    name_dup = {
        'id': 100,
        'name': 'John Doe',  # Same name as existing
        'email': 'john.doe.new@company.com',  # Different email
        'department': 'HR',
        'salary': 80000,
        'date_created': pd.Timestamp('2023-12-01')
    }
    data.append(name_dup)
    
    return pd.DataFrame(data)

# ======================== DUPLICATE HANDLING UTILITIES ========================

class DuplicateHandler:
    """Advanced duplicate detection and handling utilities."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.duplicate_analysis = {}
    
    def comprehensive_duplicate_report(self) -> Dict:
        """Generate comprehensive duplicate analysis report."""
        
        report = {
            'total_rows': len(self.df),
            'exact_duplicates': self.df.duplicated().sum(),
            'unique_rows': len(self.df.drop_duplicates()),
            'column_duplicates': {},
            'duplicate_patterns': {}
        }
        
        # Analyze each column for duplicates
        for col in self.df.columns:
            col_value_counts = self.df[col].value_counts()
            duplicated_values = col_value_counts[col_value_counts > 1]
            report['column_duplicates'][col] = {
                'unique_values': len(col_value_counts),
                'duplicated_values': len(duplicated_values),
                'max_occurrences': col_value_counts.max() if len(col_value_counts) > 0 else 0
            }
        
        # Analyze patterns
        if len(self.df) > 0:
            exact_duplicates_grouped = self.df.groupby(list(self.df.columns)).size()
            multiple_occurrences = exact_duplicates_grouped[exact_duplicates_grouped > 1]
            report['duplicate_patterns']['exact_duplicate_groups'] = len(multiple_occurrences)
        
        self.duplicate_analysis = report
        return report
    
    def remove_duplicates_with_priority(self, 
                                      priority_columns: List[str] = None,
                                      strategy: str = 'first') -> pd.DataFrame:
        """Remove duplicates with configurable priority strategy."""
        
        if priority_columns is None:
            # Use all columns
            return self.df.drop_duplicates(keep=strategy)
        
        if strategy == 'most_complete':
            # Keep record with fewest missing values
            def keep_most_complete(group):
                missing_counts = group.isnull().sum(axis=1)
                return group.loc[missing_counts.idxmin()]
            
            return self.df.groupby(priority_columns).apply(keep_most_complete).reset_index(drop=True)
        
        elif strategy == 'newest' and 'date_created' in self.df.columns:
            # Keep most recent record
            return self.df.sort_values('date_created').drop_duplicates(
                subset=priority_columns, keep='last'
            )
        
        else:
            # Standard strategy
            return self.df.drop_duplicates(subset=priority_columns, keep=strategy)

# ======================== DEMONSTRATION ========================

def run_duplicate_handling_demo():
    """Run comprehensive duplicate handling demonstration."""
    
    print("PANDAS DUPLICATE HANDLING COMPREHENSIVE GUIDE")
    print("="*45)
    
    # Main demonstration
    cleaned_df = comprehensive_duplicate_handling()
    
    # Advanced utilities demonstration
    print("\n=== DUPLICATE HANDLER CLASS DEMO ===")
    
    sample_df = create_sample_with_duplicates()
    handler = DuplicateHandler(sample_df)
    
    # Generate report
    report = handler.comprehensive_duplicate_report()
    print(f"\nDuplicate Analysis Report:")
    print(f"  Total rows: {report['total_rows']}")
    print(f"  Exact duplicates: {report['exact_duplicates']}")
    print(f"  Unique rows: {report['unique_rows']}")
    
    # Test different removal strategies
    strategies = ['first', 'last', 'most_complete']
    
    for strategy in strategies:
        try:
            cleaned = handler.remove_duplicates_with_priority(
                priority_columns=['name', 'email'], 
                strategy=strategy
            )
            print(f"  {strategy} strategy: {len(sample_df)} → {len(cleaned)} rows")
        except Exception as e:
            print(f"  {strategy} strategy failed: {str(e)}")

# Execute demonstration
if __name__ == "__main__":
    run_duplicate_handling_demo()
```

#### Explanation
1. **Detection Methods**: Multiple approaches to identify duplicates using `duplicated()` and analysis
2. **Removal Strategies**: Different techniques using `drop_duplicates()` with various parameters
3. **Smart Removal**: Intelligent strategies based on data completeness and recency
4. **Pattern Analysis**: Understanding duplicate patterns for better handling decisions

#### Use Cases
- **Data Cleaning**: Removing duplicate records before analysis
- **Data Integration**: Handling duplicates when merging datasets
- **Database Maintenance**: Ensuring data integrity in database operations
- **Survey Data**: Managing duplicate responses

#### Best Practices
- **Analyze First**: Understand duplicate patterns before removal
- **Define Strategy**: Choose appropriate keep strategy (first, last, most complete)
- **Validate Results**: Verify that removed duplicates were correct
- **Document Process**: Record duplicate handling decisions

#### Pitfalls
- **Incorrect Removal**: Removing legitimate records that appear similar
- **Key Column Selection**: Using wrong columns for duplicate identification
- **Data Loss**: Accidentally removing important variations in data
- **Performance Issues**: Not optimizing for large datasets

---

## Question 5

**How can youpivotdata in aDataFrame?**

### Answer

#### Theory
Data pivoting is a fundamental data reshaping operation that transforms data from long format to wide format, reorganizing rows into columns. Pandas provides `pivot()`, `pivot_table()`, and `unstack()` methods for different pivoting scenarios. Understanding pivoting is crucial for data analysis, reporting, and creating summary tables.

#### Code Example

```python
import pandas as pd
import numpy as np

# ======================== BASIC PIVOTING ========================

def demonstrate_basic_pivoting():
    """Demonstrate basic pivoting operations."""
    
    print("=== BASIC PIVOTING OPERATIONS ===")
    
    # Create sample data
    df = create_sales_data()
    print(f"Original data shape: {df.shape}")
    print(f"Sample data:\n{df.head()}")
    
    # 1. Basic pivot
    print("\n1. Basic Pivot:")
    pivot_basic = df.pivot(index='date', columns='product', values='sales')
    print(f"   Pivot table shape: {pivot_basic.shape}")
    print(f"   Pivot result:\n{pivot_basic.head()}")
    
    # 2. Pivot table with aggregation
    print("\n2. Pivot Table with Aggregation:")
    pivot_agg = df.pivot_table(
        index='date', 
        columns='product', 
        values='sales', 
        aggfunc='sum'
    )
    print(f"   Aggregated pivot:\n{pivot_agg.head()}")
    
    # 3. Multi-level pivoting
    print("\n3. Multi-level Pivoting:")
    pivot_multi = df.pivot_table(
        index=['date', 'region'], 
        columns='product', 
        values='sales', 
        aggfunc='mean'
    )
    print(f"   Multi-level pivot shape: {pivot_multi.shape}")
    
    return pivot_basic, pivot_agg

def create_sales_data():
    """Create sample sales data for pivoting demonstrations."""
    
    np.random.seed(42)
    
    dates = pd.date_range('2023-01-01', periods=30, freq='D')
    products = ['Laptop', 'Mouse', 'Keyboard', 'Monitor']
    regions = ['North', 'South', 'East', 'West']
    
    data = []
    for date in dates:
        for product in products:
            for region in regions:
                if np.random.random() > 0.3:  # Some missing combinations
                    data.append({
                        'date': date,
                        'product': product,
                        'region': region,
                        'sales': np.random.randint(10, 100),
                        'quantity': np.random.randint(1, 10)
                    })
    
    return pd.DataFrame(data)

# ======================== ADVANCED PIVOTING ========================

def advanced_pivoting_techniques():
    """Demonstrate advanced pivoting techniques."""
    
    print("\n=== ADVANCED PIVOTING TECHNIQUES ===")
    
    df = create_sales_data()
    
    # 1. Multiple value columns
    print("\n1. Multiple Value Columns:")
    pivot_multi_values = df.pivot_table(
        index='date',
        columns='product',
        values=['sales', 'quantity'],
        aggfunc={'sales': 'sum', 'quantity': 'mean'}
    )
    print(f"   Multi-value pivot shape: {pivot_multi_values.shape}")
    
    # 2. Custom aggregation functions
    print("\n2. Custom Aggregation:")
    pivot_custom = df.pivot_table(
        index='region',
        columns='product',
        values='sales',
        aggfunc=[np.sum, np.mean, np.std],
        fill_value=0
    )
    print(f"   Custom aggregation shape: {pivot_custom.shape}")
    
    # 3. Pivot with margins (totals)
    print("\n3. Pivot with Margins:")
    pivot_margins = df.pivot_table(
        index='region',
        columns='product',
        values='sales',
        aggfunc='sum',
        margins=True,
        margins_name='Total'
    )
    print(f"   Pivot with totals:\n{pivot_margins}")

# Run demonstrations
if __name__ == "__main__":
    demonstrate_basic_pivoting()
    advanced_pivoting_techniques()
```

#### Explanation
1. **Basic Pivot**: Simple transformation using `pivot()` method
2. **Pivot Table**: Advanced pivoting with aggregation using `pivot_table()`
3. **Multi-level**: Complex pivoting with multiple index/column levels
4. **Custom Aggregation**: Using custom functions for data aggregation

#### Use Cases
- **Financial Reporting**: Creating summary tables from transaction data
- **Sales Analysis**: Transforming sales data for regional/product analysis
- **Survey Data**: Converting responses to cross-tabulation format
- **Time Series**: Reshaping temporal data for analysis

#### Best Practices
- **Handle Missing Values**: Use `fill_value` parameter for missing combinations
- **Choose Appropriate Aggregation**: Select correct function for data type
- **Consider Performance**: Large datasets may require optimization
- **Validate Results**: Ensure pivot maintains data integrity

#### Pitfalls
- **Duplicate Indices**: Multiple values for same index/column combination
- **Memory Issues**: Large pivots can consume significant memory
- **Data Loss**: Inappropriate aggregation functions can lose information

---

## Question 6

**Show how to applyconditional logicto columns using thewhere()method.**

### Answer

#### Theory
The `where()` method in Pandas provides a powerful way to apply conditional logic to DataFrame columns. It allows you to conditionally replace values based on boolean conditions, offering more flexibility than simple filtering. The method follows the pattern: keep values where condition is True, replace with specified value where condition is False. This is particularly useful for data cleaning, feature engineering, and implementing complex business rules.

#### Code Example

```python
import pandas as pd
import numpy as np
from typing import Union, Optional, Dict, List, Any

# ======================== BASIC WHERE() OPERATIONS ========================

def demonstrate_basic_where():
    """Demonstrate basic where() method usage."""
    
    print("=== BASIC WHERE() OPERATIONS ===")
    
    # Create sample data
    df = create_sample_data()
    print(f"Original data:\n{df.head()}")
    
    # 1. Basic where() - replace negative values with 0
    print("\n1. Basic Conditional Replacement:")
    df['score_positive'] = df['score'].where(df['score'] >= 0, 0)
    print(f"Replace negative scores with 0:")
    print(df[['score', 'score_positive']].head(10))
    
    # 2. Where with different replacement values
    print("\n2. Different Replacement Values:")
    df['category_filtered'] = df['category'].where(df['category'] != 'C', 'OTHER')
    print(f"Replace category 'C' with 'OTHER':")
    print(df[['category', 'category_filtered']].value_counts())
    
    # 3. Where with NaN replacement
    print("\n3. Replace with NaN:")
    df['salary_high_only'] = df['salary'].where(df['salary'] > 70000)
    non_null_count = df['salary_high_only'].notna().sum()
    print(f"Keep only salaries > 70000, others become NaN: {non_null_count} non-null values")
    
    return df

def demonstrate_advanced_where():
    """Demonstrate advanced where() operations."""
    
    print("\n=== ADVANCED WHERE() OPERATIONS ===")
    
    df = create_sample_data()
    
    # 1. Multiple conditions with compound boolean logic
    print("\n1. Multiple Conditions:")
    
    # Complex condition: keep values only if score > 70 AND salary > 60000
    complex_condition = (df['score'] > 70) & (df['salary'] > 60000)
    df['high_performer'] = df['name'].where(complex_condition, 'Not High Performer')
    
    high_performers = (df['high_performer'] != 'Not High Performer').sum()
    print(f"High performers (score > 70 AND salary > 60000): {high_performers}")
    
    # 2. Chained where() operations
    print("\n2. Chained Where Operations:")
    df['score_categorized'] = (df['score']
                              .where(df['score'] >= 80, 'Low')
                              .where(df['score'] < 80, 'High'))
    
    print(f"Score categorization:")
    print(df['score_categorized'].value_counts())
    
    # 3. Where with lambda functions
    print("\n3. Where with Lambda Functions:")
    df['age_group'] = df['age'].where(
        df['age'].apply(lambda x: x >= 30), 
        'Young'
    ).where(
        df['age'].apply(lambda x: x < 50),
        'Senior'
    )
    
    print(f"Age groups:")
    print(df['age_group'].value_counts())
    
    # 4. Where with other column values
    print("\n4. Replace with Other Column Values:")
    # If score is low, use age as backup score
    df['adjusted_score'] = df['score'].where(df['score'] >= 50, df['age'])
    
    print(f"Original vs Adjusted scores (first 10 rows):")
    print(df[['score', 'age', 'adjusted_score']].head(10))

def demonstrate_conditional_operations():
    """Demonstrate various conditional operations using where()."""
    
    print("\n=== CONDITIONAL OPERATIONS ===")
    
    df = create_sample_data()
    
    # 1. Percentile-based conditions
    print("\n1. Percentile-based Conditions:")
    
    # Keep only top 25% salaries, replace others with median
    salary_75th = df['salary'].quantile(0.75)
    salary_median = df['salary'].median()
    
    df['salary_top_quartile'] = df['salary'].where(
        df['salary'] >= salary_75th, 
        salary_median
    )
    
    print(f"75th percentile salary: ${salary_75th:,.2f}")
    print(f"Top quartile salaries kept, others set to median: ${salary_median:,.2f}")
    
    # 2. String-based conditions
    print("\n2. String-based Conditions:")
    
    # Keep names only if they contain certain letters
    df['name_filtered'] = df['name'].where(
        df['name'].str.contains('e', case=False, na=False),
        'Name without E'
    )
    
    names_with_e = (df['name_filtered'] != 'Name without E').sum()
    print(f"Names containing 'e': {names_with_e}")
    
    # 3. Date-based conditions (if we had date columns)
    print("\n3. Index-based Conditions:")
    
    # Use row index for conditions
    df['position_based'] = df['score'].where(df.index < 50, 'Second Half')
    
    first_half_count = (df['position_based'] != 'Second Half').sum()
    print(f"First half records kept: {first_half_count}")

# ======================== PRACTICAL WHERE() UTILITIES ========================

class ConditionalProcessor:
    """Utility class for advanced conditional processing with where()."""
    
    def __init__(self, dataframe: pd.DataFrame):
        """Initialize with DataFrame."""
        self.df = dataframe.copy()
        self.operations_log = []
    
    def apply_business_rules(self, rules: Dict[str, Dict]) -> pd.DataFrame:
        """Apply multiple business rules using where() method."""
        
        result_df = self.df.copy()
        
        for rule_name, rule_config in rules.items():
            column = rule_config['column']
            condition = rule_config['condition']
            replacement = rule_config['replacement']
            new_column = rule_config.get('new_column', f"{column}_processed")
            
            try:
                # Apply the where() operation
                if callable(condition):
                    # If condition is a function
                    mask = condition(result_df[column])
                else:
                    # If condition is a boolean Series
                    mask = condition
                
                result_df[new_column] = result_df[column].where(mask, replacement)
                
                self.operations_log.append({
                    'rule': rule_name,
                    'column': column,
                    'new_column': new_column,
                    'status': 'success'
                })
                
            except Exception as e:
                self.operations_log.append({
                    'rule': rule_name,
                    'column': column,
                    'status': 'error',
                    'error': str(e)
                })
        
        return result_df
    
    def conditional_aggregation(self, group_column: str, target_column: str, 
                              conditions: Dict[str, Any]) -> pd.DataFrame:
        """Apply conditional aggregation using where()."""
        
        result_df = self.df.copy()
        
        for condition_name, condition_config in conditions.items():
            condition_func = condition_config['condition']
            agg_func = condition_config['aggregation']
            
            # Create conditional column
            conditional_values = result_df[target_column].where(
                condition_func(result_df), 
                np.nan
            )
            
            # Group and aggregate
            grouped_result = conditional_values.groupby(result_df[group_column]).agg(agg_func)
            
            # Add to result
            new_column = f"{target_column}_{condition_name}_{agg_func.__name__}"
            result_df[new_column] = result_df[group_column].map(grouped_result)
        
        return result_df
    
    def cascade_conditions(self, column: str, condition_chain: List[Dict]) -> pd.Series:
        """Apply cascading conditions using multiple where() operations."""
        
        result = self.df[column].copy()
        
        for step in condition_chain:
            condition = step['condition']
            replacement = step['replacement']
            
            if callable(condition):
                mask = condition(result)
            else:
                mask = condition
            
            result = result.where(mask, replacement)
        
        return result

def create_sample_data():
    """Create sample data for where() demonstrations."""
    
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'name': [f'Employee_{i}' for i in range(1, n_samples + 1)],
        'age': np.random.randint(22, 65, n_samples),
        'salary': np.random.normal(65000, 20000, n_samples).clip(min=30000, max=150000),
        'score': np.random.normal(75, 15, n_samples).clip(min=0, max=100),
        'category': np.random.choice(['A', 'B', 'C'], n_samples),
        'department': np.random.choice(['Engineering', 'Marketing', 'Sales', 'HR'], n_samples)
    }
    
    # Introduce some negative scores for demonstration
    data['score'][np.random.choice(n_samples, 10, replace=False)] *= -1
    
    return pd.DataFrame(data)

# ======================== COMPARISON WITH OTHER METHODS ========================

def compare_conditional_methods():
    """Compare where() with other conditional methods."""
    
    print("\n=== COMPARISON WITH OTHER CONDITIONAL METHODS ===")
    
    df = create_sample_data()
    
    # Scenario: Replace low scores with average score
    avg_score = df['score'].mean()
    condition = df['score'] < 50
    
    print(f"Scenario: Replace scores < 50 with average ({avg_score:.2f})")
    
    # Method 1: where()
    method1 = df['score'].where(~condition, avg_score)
    
    # Method 2: loc indexing
    method2 = df['score'].copy()
    method2.loc[condition] = avg_score
    
    # Method 3: numpy.where()
    method3 = pd.Series(np.where(condition, avg_score, df['score']), index=df.index)
    
    # Method 4: apply with lambda
    method4 = df['score'].apply(lambda x: avg_score if x < 50 else x)
    
    # Verify all methods produce same result
    print(f"All methods produce same result: {method1.equals(method2) and method2.equals(method3) and method3.equals(method4)}")
    
    # Performance comparison would go here in real implementation
    replaced_count = condition.sum()
    print(f"Values replaced: {replaced_count}")

# ======================== REAL-WORLD EXAMPLES ========================

def real_world_where_examples():
    """Demonstrate real-world applications of where() method."""
    
    print("\n=== REAL-WORLD WHERE() APPLICATIONS ===")
    
    # Example 1: Data Quality - Cap outliers
    print("\n1. Data Quality - Outlier Capping:")
    
    df = create_sample_data()
    
    # Cap salaries at 95th percentile
    salary_cap = df['salary'].quantile(0.95)
    df['salary_capped'] = df['salary'].where(df['salary'] <= salary_cap, salary_cap)
    
    outliers_capped = (df['salary'] != df['salary_capped']).sum()
    print(f"Salaries capped at ${salary_cap:,.2f}: {outliers_capped} outliers capped")
    
    # Example 2: Business Logic - Performance ratings
    print("\n2. Business Logic - Performance Categories:")
    
    def categorize_performance(score):
        if score >= 90:
            return 'Excellent'
        elif score >= 75:
            return 'Good'
        elif score >= 60:
            return 'Satisfactory'
        else:
            return 'Needs Improvement'
    
    # Using where() for performance categorization
    df['performance_rating'] = 'Needs Improvement'
    df['performance_rating'] = df['performance_rating'].where(
        df['score'] < 60, 'Satisfactory'
    ).where(
        df['score'] < 75, 'Good'
    ).where(
        df['score'] < 90, 'Excellent'
    )
    
    rating_counts = df['performance_rating'].value_counts()
    print(f"Performance ratings distribution:")
    for rating, count in rating_counts.items():
        print(f"  {rating}: {count}")
    
    # Example 3: Feature Engineering - Risk scoring
    print("\n3. Feature Engineering - Risk Scoring:")
    
    # Create risk flags based on multiple conditions
    df['high_risk'] = False
    
    # High risk if low score AND high salary (overpaid underperformer)
    risk_condition = (df['score'] < 60) & (df['salary'] > df['salary'].quantile(0.8))
    df['risk_flag'] = df['high_risk'].where(~risk_condition, True)
    
    high_risk_count = df['risk_flag'].sum()
    print(f"High-risk employees identified: {high_risk_count}")

# ======================== MAIN DEMONSTRATION ========================

def run_comprehensive_where_demo():
    """Run comprehensive where() method demonstration."""
    
    print("PANDAS WHERE() METHOD COMPREHENSIVE GUIDE")
    print("="*45)
    
    # Run all demonstrations
    df1 = demonstrate_basic_where()
    demonstrate_advanced_where()
    demonstrate_conditional_operations()
    compare_conditional_methods()
    real_world_where_examples()
    
    # Demonstrate utility class
    print("\n=== CONDITIONAL PROCESSOR DEMONSTRATION ===")
    
    sample_df = create_sample_data()
    processor = ConditionalProcessor(sample_df)
    
    # Define business rules
    business_rules = {
        'salary_adjustment': {
            'column': 'salary',
            'condition': lambda x: x < 50000,
            'replacement': 50000,
            'new_column': 'salary_adjusted'
        },
        'score_normalization': {
            'column': 'score',
            'condition': lambda x: x >= 0,
            'replacement': 0,
            'new_column': 'score_normalized'
        }
    }
    
    # Apply business rules
    processed_df = processor.apply_business_rules(business_rules)
    
    print(f"Business rules applied:")
    for log_entry in processor.operations_log:
        status_symbol = "✅" if log_entry['status'] == 'success' else "❌"
        print(f"  {status_symbol} {log_entry['rule']}: {log_entry['column']} → {log_entry.get('new_column', 'N/A')}")
    
    # Show results
    comparison_cols = ['salary', 'salary_adjusted', 'score', 'score_normalized']
    print(f"\nSample results:")
    print(processed_df[comparison_cols].head())

# Execute demonstration
if __name__ == "__main__":
    run_comprehensive_where_demo()
```

#### Explanation
1. **Basic Where Logic**: Simple condition-based value replacement using boolean masks
2. **Multiple Conditions**: Combining conditions with `&`, `|`, and `~` operators
3. **Chained Operations**: Sequential where() calls for complex logic
4. **Business Rules**: Applying domain-specific logic with configurable rules
5. **Performance Comparison**: Comparing where() with alternative conditional methods

#### Use Cases
- **Data Cleaning**: Replace invalid or outlier values with acceptable alternatives
- **Feature Engineering**: Create conditional features based on business logic
- **Data Validation**: Flag or correct data quality issues
- **Business Rules**: Implement complex conditional business logic
- **Risk Assessment**: Create risk flags based on multiple criteria

#### Best Practices
- **Readable Conditions**: Use clear, well-documented boolean conditions
- **Chaining Strategy**: Use method chaining for sequential conditional logic
- **Performance**: Prefer vectorized operations over apply() when possible
- **Documentation**: Document business logic for maintainability
- **Validation**: Test edge cases and boundary conditions

#### Pitfalls
- **Condition Complexity**: Overly complex conditions can be hard to debug
- **Data Types**: Ensure replacement values match column data types
- **Missing Values**: Handle NaN values appropriately in conditions
- **Performance**: Large datasets with complex conditions may be slow
- **Logic Errors**: Incorrect boolean logic can produce unexpected results

#### Debugging
```python
def debug_where_operations(df: pd.DataFrame, column: str, condition, replacement):
    """Debug where() operations step by step."""
    
    print(f"Debugging where() operation on column: {column}")
    
    # Check condition evaluation
    if callable(condition):
        mask = condition(df[column])
    else:
        mask = condition
    
    print(f"Condition evaluation:")
    print(f"  True values: {mask.sum()}")
    print(f"  False values: {(~mask).sum()}")
    print(f"  NaN in condition: {mask.isna().sum()}")
    
    # Check replacement type compatibility
    original_dtype = df[column].dtype
    replacement_dtype = type(replacement).__name__
    print(f"Data type compatibility:")
    print(f"  Original column type: {original_dtype}")
    print(f"  Replacement type: {replacement_dtype}")
    
    # Preview result
    result = df[column].where(mask, replacement)
    changes = (df[column] != result).sum() if original_dtype == result.dtype else "Type changed"
    print(f"Changes made: {changes}")

def validate_conditional_logic(df: pd.DataFrame, conditions: Dict):
    """Validate conditional logic before applying."""
    
    validation_results = {}
    
    for name, condition_func in conditions.items():
        try:
            mask = condition_func(df)
            validation_results[name] = {
                'valid': True,
                'true_count': mask.sum(),
                'false_count': (~mask).sum(),
                'na_count': mask.isna().sum()
            }
        except Exception as e:
            validation_results[name] = {
                'valid': False,
                'error': str(e)
            }
    
    return validation_results
```

#### Optimization

**Where() Performance Tips:**

| Scenario | Optimization Strategy |
|----------|----------------------|
| **Simple Conditions** | Use vectorized boolean operations |
| **Multiple Conditions** | Combine with `&`, `|` operators |
| **Complex Logic** | Break into smaller, testable conditions |
| **Large DataFrames** | Consider chunked processing |
| **Repeated Operations** | Cache condition results |

**Performance Comparison:**
- `where()`: Best for conditional replacement
- `loc[]`: Best for in-place modifications  
- `np.where()`: Best for simple numeric conditions
- `apply()`: Most flexible but slowest

---

## Question 7

**How do youreshapeaDataFrameusingstackandunstackmethods?**

### Answer

#### Theory
Data reshaping with `stack()` and `unstack()` is fundamental for transforming DataFrame structure between wide and long formats. These methods pivot data by moving between column and index levels, essential for data analysis, visualization preparation, and statistical operations. Stack converts columns to rows (wide to long), while unstack converts index levels to columns (long to wide).

#### Code Example

```python
import pandas as pd
import numpy as np
from typing import Union, Optional, List, Dict

# ======================== BASIC STACK AND UNSTACK ========================

def demonstrate_basic_stack_unstack():
    """Demonstrate basic stack and unstack operations."""
    
    print("=== BASIC STACK AND UNSTACK OPERATIONS ===")
    
    # Create sample wide-format data
    df_wide = create_wide_format_data()
    print(f"Original wide format data:")
    print(df_wide)
    print(f"Shape: {df_wide.shape}")
    
    # 1. Stack - convert columns to rows (wide to long)
    print(f"\n1. STACK Operation (Wide to Long):")
    df_stacked = df_wide.stack()
    print(f"Stacked data:")
    print(df_stacked)
    print(f"Type: {type(df_stacked)}")
    print(f"Shape: {df_stacked.shape}")
    
    # 2. Unstack - convert rows to columns (long to wide)
    print(f"\n2. UNSTACK Operation (Long to Wide):")
    df_unstacked = df_stacked.unstack()
    print(f"Unstacked back to wide:")
    print(df_unstacked)
    print(f"Shape: {df_unstacked.shape}")
    
    # Verify round-trip consistency
    is_consistent = df_wide.equals(df_unstacked)
    print(f"Round-trip consistency: {is_consistent}")
    
    return df_wide, df_stacked

def demonstrate_multilevel_operations():
    """Demonstrate stack/unstack with multi-level indices."""
    
    print(f"\n=== MULTI-LEVEL INDEX OPERATIONS ===")
    
    # Create multi-level DataFrame
    df_multi = create_multilevel_data()
    print(f"Multi-level DataFrame:")
    print(df_multi)
    print(f"Index levels: {df_multi.index.names}")
    print(f"Column levels: {df_multi.columns.names}")
    
    # 1. Stack specific level
    print(f"\n1. Stack Specific Level:")
    
    # Stack the innermost column level
    stacked_inner = df_multi.stack(level=-1)
    print(f"Stacked inner level:")
    print(stacked_inner.head(10))
    print(f"New index levels: {stacked_inner.index.names}")
    
    # Stack the outermost column level
    stacked_outer = df_multi.stack(level=0)
    print(f"\nStacked outer level:")
    print(stacked_outer.head(10))
    
    # 2. Unstack specific level
    print(f"\n2. Unstack Specific Level:")
    
    # Create a Series with multi-level index
    series_multi = df_multi.stack()
    print(f"Multi-index Series:")
    print(series_multi.head())
    
    # Unstack different levels
    unstacked_last = series_multi.unstack(level=-1)
    print(f"\nUnstacked last level:")
    print(unstacked_last.head())
    
    unstacked_first = series_multi.unstack(level=0)
    print(f"\nUnstacked first level:")
    print(unstacked_first.head())
    
    return df_multi, series_multi

def demonstrate_advanced_reshaping():
    """Demonstrate advanced reshaping techniques."""
    
    print(f"\n=== ADVANCED RESHAPING TECHNIQUES ===")
    
    # Create complex dataset
    df_complex = create_complex_dataset()
    print(f"Complex dataset:")
    print(df_complex.head())
    
    # 1. Stack with dropna parameter
    print(f"\n1. Handling Missing Values in Stack:")
    
    # Stack without dropping NaN
    stacked_with_nan = df_complex.stack(dropna=False)
    print(f"Stacked with NaN preserved: {stacked_with_nan.shape[0]} rows")
    
    # Stack dropping NaN (default)
    stacked_drop_nan = df_complex.stack(dropna=True)
    print(f"Stacked with NaN dropped: {stacked_drop_nan.shape[0]} rows")
    
    # 2. Multiple level stacking
    print(f"\n2. Multiple Level Operations:")
    
    # Create DataFrame with multiple column levels
    df_multilevel_cols = create_multilevel_columns_data()
    print(f"DataFrame with multi-level columns:")
    print(df_multilevel_cols.head())
    
    # Stack all levels
    fully_stacked = df_multilevel_cols.stack(level=[0, 1])
    print(f"Fully stacked (all levels): {fully_stacked.shape}")
    
    # Stack only specific levels
    partial_stack = df_multilevel_cols.stack(level=1)
    print(f"Partial stack (level 1 only): {partial_stack.shape}")
    
    # 3. Fill value in unstack
    print(f"\n3. Fill Values in Unstack:")
    
    # Create data with missing combinations
    sparse_data = create_sparse_data()
    print(f"Sparse data:")
    print(sparse_data)
    
    # Unstack with default (NaN fill)
    unstacked_default = sparse_data.unstack()
    print(f"Unstacked with NaN fill:")
    print(unstacked_default)
    
    # Unstack with custom fill value
    unstacked_filled = sparse_data.unstack(fill_value=0)
    print(f"Unstacked with 0 fill:")
    print(unstacked_filled)

# ======================== PRACTICAL APPLICATIONS ========================

def practical_reshaping_examples():
    """Demonstrate practical applications of stack/unstack."""
    
    print(f"\n=== PRACTICAL RESHAPING APPLICATIONS ===")
    
    # 1. Time series analysis
    print(f"\n1. Time Series Reshaping:")
    
    ts_data = create_timeseries_data()
    print(f"Time series data (wide format):")
    print(ts_data.head())
    
    # Stack for analysis
    ts_long = ts_data.stack().reset_index()
    ts_long.columns = ['date', 'metric', 'value']
    print(f"Time series in long format:")
    print(ts_long.head())
    
    # Calculate rolling statistics in long format
    ts_long['rolling_mean'] = ts_long.groupby('metric')['value'].rolling(7).mean().reset_index(0, drop=True)
    print(f"Added rolling statistics:")
    print(ts_long.head())
    
    # 2. Survey data transformation
    print(f"\n2. Survey Data Transformation:")
    
    survey_data = create_survey_data()
    print(f"Survey responses (wide format):")
    print(survey_data.head())
    
    # Convert to long format for analysis
    survey_long = survey_data.set_index('respondent_id').stack().reset_index()
    survey_long.columns = ['respondent_id', 'question', 'response']
    print(f"Survey in long format:")
    print(survey_long.head())
    
    # Calculate question statistics
    question_stats = survey_long.groupby('question')['response'].agg(['mean', 'std', 'count'])
    print(f"Question statistics:")
    print(question_stats)
    
    # 3. Financial data analysis
    print(f"\n3. Financial Data Analysis:")
    
    financial_data = create_financial_data()
    print(f"Financial data (multi-currency, wide):")
    print(financial_data.head())
    
    # Stack currencies for comparison
    financial_long = financial_data.stack(level='Currency').reset_index()
    print(f"Financial data stacked by currency:")
    print(financial_long.head())
    
    # Calculate currency correlations
    financial_pivot = financial_long.pivot(index='Date', columns='Currency', values='Price')
    correlations = financial_pivot.corr()
    print(f"Currency correlations:")
    print(correlations)

# ======================== UTILITY CLASS FOR RESHAPING ========================

class DataReshaper:
    """Utility class for advanced data reshaping operations."""
    
    def __init__(self, dataframe: pd.DataFrame):
        """Initialize with DataFrame."""
        self.df = dataframe.copy()
        self.reshaping_history = []
    
    def smart_stack(self, target_format: str = 'long', handle_missing: str = 'drop') -> pd.DataFrame:
        """Intelligently stack data based on target format."""
        
        if target_format == 'long':
            # Convert to long format
            if isinstance(self.df.columns, pd.MultiIndex):
                # Multi-level columns
                result = self.df.stack(level=list(range(self.df.columns.nlevels)))
            else:
                # Single level columns
                result = self.df.stack(dropna=(handle_missing == 'drop'))
            
            self.reshaping_history.append({
                'operation': 'stack',
                'target_format': target_format,
                'original_shape': self.df.shape,
                'result_shape': result.shape
            })
            
            return result
        
        else:
            raise ValueError(f"Unsupported target format: {target_format}")
    
    def smart_unstack(self, level: Union[int, str, List] = -1, 
                     fill_value=None, sparse: bool = False) -> pd.DataFrame:
        """Intelligently unstack data with optimization."""
        
        if not isinstance(self.df.index, pd.MultiIndex):
            raise ValueError("DataFrame must have MultiIndex for unstacking")
        
        # Determine optimal fill value if not specified
        if fill_value is None:
            if self.df.dtype in ['int64', 'float64']:
                fill_value = 0
            else:
                fill_value = 'Unknown'
        
        result = self.df.unstack(level=level, fill_value=fill_value)
        
        # Use sparse arrays for memory efficiency if requested
        if sparse and hasattr(pd, 'SparseDtype'):
            for col in result.columns:
                if result[col].dtype in ['int64', 'float64']:
                    result[col] = result[col].astype(pd.SparseDtype(result[col].dtype, fill_value))
        
        self.reshaping_history.append({
            'operation': 'unstack',
            'level': level,
            'fill_value': fill_value,
            'original_shape': self.df.shape,
            'result_shape': result.shape
        })
        
        return result
    
    def optimize_memory_usage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize memory usage after reshaping."""
        
        result_df = df.copy()
        
        for col in result_df.columns:
            if result_df[col].dtype == 'object':
                # Try to convert to category
                if result_df[col].nunique() / len(result_df[col]) < 0.5:
                    result_df[col] = result_df[col].astype('category')
            
            elif result_df[col].dtype == 'float64':
                # Downcast float if possible
                if result_df[col].min() >= np.finfo(np.float32).min and \
                   result_df[col].max() <= np.finfo(np.float32).max:
                    result_df[col] = result_df[col].astype('float32')
        
        return result_df

# ======================== HELPER FUNCTIONS ========================

def create_wide_format_data():
    """Create sample wide-format data."""
    
    np.random.seed(42)
    
    data = {
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'C': [100, 200, 300, 400, 500],
        'D': [1000, 2000, 3000, 4000, 5000]
    }
    
    return pd.DataFrame(data, index=['Row1', 'Row2', 'Row3', 'Row4', 'Row5'])

def create_multilevel_data():
    """Create sample data with multi-level index and columns."""
    
    # Create multi-level columns
    columns = pd.MultiIndex.from_product([['Sales', 'Marketing'], ['Q1', 'Q2', 'Q3', 'Q4']], 
                                       names=['Department', 'Quarter'])
    
    # Create multi-level index
    index = pd.MultiIndex.from_product([['North', 'South'], ['Product_A', 'Product_B']], 
                                     names=['Region', 'Product'])
    
    # Generate data
    np.random.seed(42)
    data = np.random.randint(100, 1000, size=(4, 8))
    
    return pd.DataFrame(data, index=index, columns=columns)

def create_complex_dataset():
    """Create complex dataset with missing values."""
    
    np.random.seed(42)
    
    data = pd.DataFrame({
        'Metric1': [1, 2, np.nan, 4, 5],
        'Metric2': [10, np.nan, 30, 40, 50],
        'Metric3': [100, 200, 300, np.nan, 500],
        'Metric4': [np.nan, 2000, 3000, 4000, 5000]
    }, index=['A', 'B', 'C', 'D', 'E'])
    
    return data

def create_multilevel_columns_data():
    """Create data with multi-level columns."""
    
    # Create hierarchical columns
    columns = pd.MultiIndex.from_product([['Revenue', 'Costs'], ['2022', '2023']], 
                                       names=['Metric', 'Year'])
    
    np.random.seed(42)
    data = np.random.randint(1000, 10000, size=(5, 4))
    
    return pd.DataFrame(data, 
                       index=['Jan', 'Feb', 'Mar', 'Apr', 'May'],
                       columns=columns)

def create_sparse_data():
    """Create sparse data for unstack demonstration."""
    
    index = pd.MultiIndex.from_tuples([
        ('A', 'X'), ('A', 'Y'), ('B', 'X'), ('B', 'Z'), ('C', 'Y')
    ], names=['Group', 'Item'])
    
    return pd.Series([1, 2, 3, 4, 5], index=index, name='Value')

def create_timeseries_data():
    """Create time series data."""
    
    dates = pd.date_range('2023-01-01', periods=30, freq='D')
    
    np.random.seed(42)
    data = {
        'Temperature': np.random.normal(20, 5, 30),
        'Humidity': np.random.normal(60, 10, 30),
        'Pressure': np.random.normal(1013, 20, 30)
    }
    
    return pd.DataFrame(data, index=dates)

def create_survey_data():
    """Create survey response data."""
    
    np.random.seed(42)
    n_respondents = 100
    
    data = {
        'respondent_id': range(1, n_respondents + 1),
        'satisfaction_q1': np.random.randint(1, 6, n_respondents),
        'satisfaction_q2': np.random.randint(1, 6, n_respondents),
        'satisfaction_q3': np.random.randint(1, 6, n_respondents),
        'likelihood_q1': np.random.randint(1, 11, n_respondents),
        'likelihood_q2': np.random.randint(1, 11, n_respondents)
    }
    
    return pd.DataFrame(data)

def create_financial_data():
    """Create financial data with multi-currency."""
    
    dates = pd.date_range('2023-01-01', periods=30, freq='D')
    
    # Multi-level columns for currencies
    columns = pd.MultiIndex.from_product([['USD', 'EUR', 'GBP'], ['Price']], 
                                       names=['Currency', 'Metric'])
    
    np.random.seed(42)
    data = np.random.uniform(0.8, 1.2, size=(30, 3))
    
    return pd.DataFrame(data, index=dates, columns=columns)

# ======================== MAIN DEMONSTRATION ========================

def run_comprehensive_stack_unstack_demo():
    """Run comprehensive stack/unstack demonstration."""
    
    print("PANDAS STACK/UNSTACK COMPREHENSIVE GUIDE")
    print("="*45)
    
    # Basic operations
    df_wide, df_stacked = demonstrate_basic_stack_unstack()
    
    # Multi-level operations
    df_multi, series_multi = demonstrate_multilevel_operations()
    
    # Advanced techniques
    demonstrate_advanced_reshaping()
    
    # Practical applications
    practical_reshaping_examples()
    
    # Demonstrate utility class
    print(f"\n=== DATA RESHAPER CLASS DEMONSTRATION ===")
    
    sample_df = create_wide_format_data()
    reshaper = DataReshaper(sample_df)
    
    # Smart stack
    stacked_result = reshaper.smart_stack(target_format='long')
    print(f"Smart stack result shape: {stacked_result.shape}")
    
    # Smart unstack (need MultiIndex)
    reshaper_multi = DataReshaper(df_multi)
    unstacked_result = reshaper_multi.smart_unstack(level=0, fill_value=0)
    print(f"Smart unstack result shape: {unstacked_result.shape}")
    
    # Show reshaping history
    print(f"\nReshaping history:")
    for i, operation in enumerate(reshaper_multi.reshaping_history, 1):
        print(f"  {i}. {operation['operation']}: {operation['original_shape']} → {operation['result_shape']}")

# Execute demonstration
if __name__ == "__main__":
    run_comprehensive_stack_unstack_demo()
```

#### Explanation
1. **Stack Operation**: Converts columns to index levels, creating long format from wide format
2. **Unstack Operation**: Converts index levels to columns, creating wide format from long format
3. **Multi-level Handling**: Managing complex hierarchical data structures
4. **Missing Value Control**: Options for handling NaN values during reshaping
5. **Level Specification**: Targeting specific index/column levels for transformation

#### Use Cases
- **Data Analysis**: Converting between wide and long formats for different analytical needs
- **Visualization Preparation**: Reshaping data for plotting libraries that expect specific formats
- **Statistical Operations**: Preparing data for statistical functions that require specific structures
- **Database Operations**: Converting between normalized and denormalized formats
- **Time Series Analysis**: Reshaping temporal data for different analytical approaches

#### Best Practices
- **Understand Data Structure**: Know your index and column hierarchy before reshaping
- **Handle Missing Values**: Explicitly decide how to handle NaN values
- **Memory Considerations**: Be aware of memory usage with large DataFrames
- **Naming Conventions**: Use clear names for multi-level indices and columns
- **Reversibility**: Ensure operations can be reversed if needed

#### Pitfalls
- **Memory Explosion**: Unstacking sparse data can create many NaN values
- **Index Confusion**: Losing track of which level to stack/unstack
- **Data Type Changes**: Reshaping may change data types unexpectedly
- **Performance Issues**: Large multi-level operations can be slow
- **Missing Data**: Unexpected NaN introduction during operations

#### Debugging
```python
def debug_stack_unstack(df: pd.DataFrame, operation: str):
    """Debug stack/unstack operations."""
    
    print(f"Debugging {operation} operation:")
    print(f"Original shape: {df.shape}")
    print(f"Index type: {type(df.index)}")
    print(f"Columns type: {type(df.columns)}")
    
    if isinstance(df.index, pd.MultiIndex):
        print(f"Index levels: {df.index.nlevels}")
        print(f"Index names: {df.index.names}")
    
    if isinstance(df.columns, pd.MultiIndex):
        print(f"Column levels: {df.columns.nlevels}")
        print(f"Column names: {df.columns.names}")
    
    # Check for missing values
    missing_count = df.isnull().sum().sum()
    print(f"Missing values: {missing_count}")

def analyze_reshaping_impact(original: pd.DataFrame, reshaped):
    """Analyze the impact of reshaping operations."""
    
    print("Reshaping Impact Analysis:")
    print(f"Shape change: {original.shape} → {reshaped.shape}")
    
    # Memory usage comparison
    orig_memory = original.memory_usage(deep=True).sum()
    
    if hasattr(reshaped, 'memory_usage'):
        new_memory = reshaped.memory_usage(deep=True).sum()
        memory_change = ((new_memory - orig_memory) / orig_memory) * 100
        print(f"Memory change: {memory_change:.1f}%")
    
    # Data completeness
    if hasattr(reshaped, 'isnull'):
        orig_missing = original.isnull().sum().sum()
        new_missing = reshaped.isnull().sum().sum()
        print(f"Missing values: {orig_missing} → {new_missing}")
```

#### Optimization

**Stack/Unstack Performance Tips:**

| Operation | Optimization Strategy |
|-----------|----------------------|
| **Large DataFrames** | Use chunking or consider alternatives like `melt()`/`pivot()` |
| **Memory Usage** | Use `sparse=True` in unstack when possible |
| **Multiple Levels** | Stack/unstack specific levels rather than all |
| **Missing Data** | Use `dropna=True` to reduce result size |
| **Frequent Operations** | Cache intermediate results |

**Memory Optimization:**
- Use categorical data types for repetitive string data
- Consider sparse arrays for data with many zeros/NaN
- Process in chunks for very large datasets
- Use appropriate data types (int32 vs int64, etc.)

---

## Question 8

**How can you performstatistical aggregationonDataFramegroups?**

### Answer

#### Theory
Statistical aggregation on DataFrame groups is a core data analysis technique that applies statistical functions to subsets of data grouped by one or more criteria. Using `groupby()` operations, you can calculate summary statistics, perform transformations, and apply custom functions to grouped data. This enables powerful analytical workflows for understanding patterns across different segments of your dataset.

#### Code Example

```python
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Callable, Optional
import warnings
from functools import reduce

# ======================== BASIC STATISTICAL AGGREGATION ========================

def demonstrate_basic_groupby_stats():
    """Demonstrate basic statistical aggregation on grouped data."""
    
    print("=== BASIC STATISTICAL AGGREGATION ===")
    
    # Create sample dataset
    df = create_sales_dataset()
    print("Sales Dataset:")
    print(df.head(10))
    print(f"Dataset shape: {df.shape}")
    
    # 1. Basic aggregation functions
    print(f"\n1. BASIC AGGREGATION FUNCTIONS:")
    
    # Group by single column
    region_stats = df.groupby('region')['sales'].agg([
        'count', 'sum', 'mean', 'median', 'std', 'min', 'max'
    ])
    print(f"Sales statistics by region:")
    print(region_stats)
    
    # Group by multiple columns
    product_region_stats = df.groupby(['region', 'product'])['sales'].agg([
        'mean', 'sum', 'count'
    ]).round(2)
    print(f"\nSales statistics by region and product:")
    print(product_region_stats.head(10))
    
    # 2. Multiple column aggregation
    print(f"\n2. MULTIPLE COLUMN AGGREGATION:")
    
    multi_col_agg = df.groupby('region').agg({
        'sales': ['sum', 'mean', 'count'],
        'quantity': ['sum', 'mean'],
        'price': ['mean', 'std']
    }).round(2)
    
    print(f"Multi-column aggregation:")
    print(multi_col_agg)
    
    # Flatten column names for easier access
    multi_col_agg.columns = ['_'.join(col).strip() for col in multi_col_agg.columns]
    print(f"\nFlattened column names:")
    print(multi_col_agg.columns.tolist())
    
    return df, region_stats

def demonstrate_advanced_aggregation():
    """Demonstrate advanced aggregation techniques."""
    
    print(f"\n=== ADVANCED AGGREGATION TECHNIQUES ===")
    
    df = create_sales_dataset()
    
    # 1. Custom aggregation functions
    print(f"\n1. CUSTOM AGGREGATION FUNCTIONS:")
    
    def coefficient_of_variation(series):
        """Calculate coefficient of variation."""
        return series.std() / series.mean() if series.mean() != 0 else np.nan
    
    def percentile_90(series):
        """Calculate 90th percentile."""
        return series.quantile(0.9)
    
    def sales_range(series):
        """Calculate sales range."""
        return series.max() - series.min()
    
    custom_agg = df.groupby('region')['sales'].agg([
        'mean',
        coefficient_of_variation,
        percentile_90,
        sales_range,
        lambda x: x.skew()  # Skewness
    ]).round(3)
    
    custom_agg.columns = ['mean', 'cv', 'p90', 'range', 'skewness']
    print(f"Custom aggregation functions:")
    print(custom_agg)
    
    # 2. Named aggregation (pandas 0.25+)
    print(f"\n2. NAMED AGGREGATION:")
    
    named_agg = df.groupby('region').agg(
        total_sales=('sales', 'sum'),
        avg_sales=('sales', 'mean'),
        median_sales=('sales', 'median'),
        sales_std=('sales', 'std'),
        max_quantity=('quantity', 'max'),
        avg_price=('price', 'mean'),
        transaction_count=('sales', 'count')
    ).round(2)
    
    print(f"Named aggregation:")
    print(named_agg)
    
    # 3. Multiple functions per column
    print(f"\n3. MULTIPLE FUNCTIONS PER COLUMN:")
    
    detailed_agg = df.groupby(['region', 'product']).agg({
        'sales': ['count', 'sum', 'mean', 'std', 'min', 'max'],
        'quantity': ['sum', 'mean', 'max'],
        'price': ['mean', 'std']
    })
    
    print(f"Detailed aggregation (first 10 rows):")
    print(detailed_agg.head(10))
    
    return custom_agg, named_agg

def demonstrate_time_based_aggregation():
    """Demonstrate time-based statistical aggregation."""
    
    print(f"\n=== TIME-BASED AGGREGATION ===")
    
    # Create time series data
    ts_df = create_timeseries_sales_data()
    print(f"Time series sales data:")
    print(ts_df.head())
    
    # 1. Temporal aggregation
    print(f"\n1. TEMPORAL AGGREGATION:")
    
    # Daily statistics
    daily_stats = ts_df.groupby('date').agg({
        'sales': ['sum', 'mean', 'count'],
        'customers': 'sum'
    })
    print(f"Daily statistics (first 5 days):")
    print(daily_stats.head())
    
    # Monthly aggregation
    ts_df['month'] = ts_df['date'].dt.to_period('M')
    monthly_stats = ts_df.groupby('month').agg({
        'sales': ['sum', 'mean', 'count', 'std'],
        'customers': ['sum', 'mean']
    }).round(2)
    print(f"\nMonthly statistics:")
    print(monthly_stats)
    
    # 2. Rolling window aggregation within groups
    print(f"\n2. ROLLING WINDOW AGGREGATION:")
    
    # Add rolling statistics
    ts_df = ts_df.sort_values(['store_id', 'date'])
    ts_df['rolling_7d_sales'] = ts_df.groupby('store_id')['sales'].rolling(
        window=7, min_periods=1
    ).mean().reset_index(0, drop=True)
    
    ts_df['rolling_7d_customers'] = ts_df.groupby('store_id')['customers'].rolling(
        window=7, min_periods=1
    ).sum().reset_index(0, drop=True)
    
    print(f"Data with rolling statistics:")
    print(ts_df[['date', 'store_id', 'sales', 'rolling_7d_sales', 
                 'customers', 'rolling_7d_customers']].head(10))
    
    return ts_df, monthly_stats

# ======================== ADVANCED GROUPBY TECHNIQUES ========================

def demonstrate_transform_and_filter():
    """Demonstrate transform and filter operations with groups."""
    
    print(f"\n=== TRANSFORM AND FILTER OPERATIONS ===")
    
    df = create_sales_dataset()
    
    # 1. Transform operations
    print(f"\n1. TRANSFORM OPERATIONS:")
    
    # Add group statistics as new columns
    df['region_avg_sales'] = df.groupby('region')['sales'].transform('mean')
    df['region_total_sales'] = df.groupby('region')['sales'].transform('sum')
    df['sales_vs_region_avg'] = df['sales'] - df['region_avg_sales']
    df['sales_pct_of_region'] = (df['sales'] / df['region_total_sales'] * 100).round(2)
    
    print(f"Data with transformed columns:")
    print(df[['region', 'sales', 'region_avg_sales', 'sales_vs_region_avg', 
              'sales_pct_of_region']].head(10))
    
    # 2. Ranking within groups
    print(f"\n2. RANKING WITHIN GROUPS:")
    
    df['sales_rank_in_region'] = df.groupby('region')['sales'].rank(method='dense', ascending=False)
    df['quantity_rank_in_region'] = df.groupby('region')['quantity'].rank(method='dense', ascending=False)
    
    top_performers = df[df['sales_rank_in_region'] <= 3].sort_values(['region', 'sales_rank_in_region'])
    print(f"Top 3 performers by region:")
    print(top_performers[['region', 'product', 'sales', 'sales_rank_in_region']].head(15))
    
    # 3. Filter operations
    print(f"\n3. FILTER OPERATIONS:")
    
    # Filter groups by group properties
    high_volume_regions = df.groupby('region').filter(lambda x: x['sales'].sum() > 100000)
    print(f"Data from high-volume regions (sales > 100k):")
    print(f"Regions included: {high_volume_regions['region'].unique()}")
    print(f"Filtered data shape: {high_volume_regions.shape}")
    
    # Filter by group size
    large_groups = df.groupby('product').filter(lambda x: len(x) >= 50)
    print(f"\nData from products with >= 50 transactions:")
    print(f"Products included: {large_groups['product'].unique()}")
    print(f"Filtered data shape: {large_groups.shape}")
    
    return df, high_volume_regions

# ======================== STATISTICAL AGGREGATION CLASS ========================

class StatisticalAggregator:
    """Advanced statistical aggregation utility for DataFrames."""
    
    def __init__(self, dataframe: pd.DataFrame):
        """Initialize with DataFrame."""
        self.df = dataframe.copy()
        self.aggregation_results = {}
    
    def comprehensive_group_stats(self, groupby_cols: Union[str, List[str]], 
                                 numeric_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """Generate comprehensive statistics for grouped data."""
        
        if numeric_cols is None:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove groupby columns from numeric columns
        if isinstance(groupby_cols, str):
            groupby_cols = [groupby_cols]
        
        numeric_cols = [col for col in numeric_cols if col not in groupby_cols]
        
        # Define comprehensive statistics
        stats_dict = {}
        for col in numeric_cols:
            stats_dict[col] = [
                'count', 'sum', 'mean', 'median', 'std', 'var',
                'min', 'max', 'skew', 
                lambda x: x.quantile(0.25),  # Q1
                lambda x: x.quantile(0.75),  # Q3
                lambda x: x.quantile(0.9),   # P90
                lambda x: x.max() - x.min()  # Range
            ]
        
        # Perform aggregation
        result = self.df.groupby(groupby_cols).agg(stats_dict)
        
        # Flatten column names
        result.columns = ['_'.join(col).strip() for col in result.columns]
        
        # Add coefficient of variation
        for col in numeric_cols:
            mean_col = f"{col}_mean"
            std_col = f"{col}_std"
            if mean_col in result.columns and std_col in result.columns:
                result[f"{col}_cv"] = result[std_col] / result[mean_col]
        
        self.aggregation_results['comprehensive'] = result
        return result
    
    def rolling_group_stats(self, groupby_cols: Union[str, List[str]], 
                           value_col: str, window: int, 
                           sort_col: Optional[str] = None) -> pd.DataFrame:
        """Calculate rolling statistics within groups."""
        
        df_sorted = self.df.copy()
        
        if sort_col:
            df_sorted = df_sorted.sort_values(groupby_cols + [sort_col])
        
        # Calculate rolling statistics
        rolling_stats = df_sorted.groupby(groupby_cols)[value_col].rolling(
            window=window, min_periods=1
        ).agg(['mean', 'std', 'min', 'max', 'sum']).reset_index()
        
        # Merge back with original data
        result = df_sorted.reset_index().merge(
            rolling_stats, 
            left_on=['index'] + groupby_cols, 
            right_on=['level_0'] + groupby_cols,
            how='left'
        ).drop(['level_0'], axis=1)
        
        # Rename columns
        rename_dict = {
            'mean': f'rolling_{window}_mean',
            'std': f'rolling_{window}_std',
            'min': f'rolling_{window}_min',
            'max': f'rolling_{window}_max',
            'sum': f'rolling_{window}_sum'
        }
        result = result.rename(columns=rename_dict)
        
        self.aggregation_results['rolling'] = result
        return result
    
    def outlier_detection_by_group(self, groupby_cols: Union[str, List[str]], 
                                  value_col: str, method: str = 'iqr') -> pd.DataFrame:
        """Detect outliers within each group."""
        
        def detect_outliers_iqr(group):
            """Detect outliers using IQR method."""
            Q1 = group[value_col].quantile(0.25)
            Q3 = group[value_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            group['is_outlier'] = (group[value_col] < lower_bound) | (group[value_col] > upper_bound)
            group['outlier_score'] = np.where(
                group[value_col] < lower_bound,
                (lower_bound - group[value_col]) / IQR,
                np.where(
                    group[value_col] > upper_bound,
                    (group[value_col] - upper_bound) / IQR,
                    0
                )
            )
            return group
        
        def detect_outliers_zscore(group):
            """Detect outliers using Z-score method."""
            mean_val = group[value_col].mean()
            std_val = group[value_col].std()
            
            if std_val == 0:
                group['is_outlier'] = False
                group['outlier_score'] = 0
            else:
                group['z_score'] = np.abs((group[value_col] - mean_val) / std_val)
                group['is_outlier'] = group['z_score'] > 3
                group['outlier_score'] = group['z_score']
            
            return group
        
        if method == 'iqr':
            result = self.df.groupby(groupby_cols).apply(detect_outliers_iqr).reset_index(drop=True)
        elif method == 'zscore':
            result = self.df.groupby(groupby_cols).apply(detect_outliers_zscore).reset_index(drop=True)
        else:
            raise ValueError("Method must be 'iqr' or 'zscore'")
        
        self.aggregation_results['outliers'] = result
        return result
    
    def generate_group_report(self, groupby_cols: Union[str, List[str]]) -> Dict:
        """Generate comprehensive group analysis report."""
        
        report = {}
        
        # Basic group information
        group_sizes = self.df.groupby(groupby_cols).size()
        report['group_sizes'] = {
            'total_groups': len(group_sizes),
            'avg_group_size': group_sizes.mean(),
            'min_group_size': group_sizes.min(),
            'max_group_size': group_sizes.max(),
            'group_size_std': group_sizes.std()
        }
        
        # Numeric column analysis
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if isinstance(groupby_cols, str):
            groupby_cols_list = [groupby_cols]
        else:
            groupby_cols_list = groupby_cols
        
        numeric_cols = [col for col in numeric_cols if col not in groupby_cols_list]
        
        if numeric_cols:
            comprehensive_stats = self.comprehensive_group_stats(groupby_cols, numeric_cols)
            report['comprehensive_stats'] = comprehensive_stats
            
            # Group comparison
            for col in numeric_cols:
                mean_col = f"{col}_mean"
                if mean_col in comprehensive_stats.columns:
                    group_means = comprehensive_stats[mean_col]
                    report[f'{col}_group_comparison'] = {
                        'highest_group': group_means.idxmax(),
                        'lowest_group': group_means.idxmin(),
                        'mean_difference': group_means.max() - group_means.min(),
                        'cv_across_groups': group_means.std() / group_means.mean()
                    }
        
        return report

# ======================== HELPER FUNCTIONS ========================

def create_sales_dataset():
    """Create comprehensive sales dataset for demonstration."""
    
    np.random.seed(42)
    
    regions = ['North', 'South', 'East', 'West']
    products = ['Product_A', 'Product_B', 'Product_C', 'Product_D', 'Product_E']
    salespeople = [f'Sales_{i}' for i in range(1, 21)]
    
    n_records = 1000
    
    data = {
        'region': np.random.choice(regions, n_records),
        'product': np.random.choice(products, n_records),
        'salesperson': np.random.choice(salespeople, n_records),
        'sales': np.random.exponential(5000, n_records).round(2),
        'quantity': np.random.poisson(10, n_records),
        'price': np.random.normal(100, 20, n_records).round(2)
    }
    
    # Ensure positive prices
    data['price'] = np.where(data['price'] < 0, np.abs(data['price']), data['price'])
    
    return pd.DataFrame(data)

def create_timeseries_sales_data():
    """Create time series sales dataset."""
    
    np.random.seed(42)
    
    # Create date range
    date_range = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    stores = ['Store_A', 'Store_B', 'Store_C', 'Store_D']
    
    data = []
    for date in date_range:
        for store in stores:
            # Add some seasonality and randomness
            base_sales = 1000 + 200 * np.sin(2 * np.pi * date.dayofyear / 365)
            daily_sales = base_sales + np.random.normal(0, 100)
            daily_customers = max(1, int(daily_sales / 50 + np.random.normal(0, 5)))
            
            data.append({
                'date': date,
                'store_id': store,
                'sales': max(0, daily_sales),
                'customers': daily_customers
            })
    
    return pd.DataFrame(data)

# ======================== MAIN DEMONSTRATION ========================

def run_comprehensive_aggregation_demo():
    """Run comprehensive statistical aggregation demonstration."""
    
    print("PANDAS STATISTICAL AGGREGATION COMPREHENSIVE GUIDE")
    print("="*50)
    
    # Basic statistical aggregation
    df, region_stats = demonstrate_basic_groupby_stats()
    
    # Advanced aggregation techniques
    custom_agg, named_agg = demonstrate_advanced_aggregation()
    
    # Time-based aggregation
    ts_df, monthly_stats = demonstrate_time_based_aggregation()
    
    # Transform and filter operations
    transformed_df, filtered_df = demonstrate_transform_and_filter()
    
    # Demonstrate StatisticalAggregator class
    print(f"\n=== STATISTICAL AGGREGATOR CLASS ===")
    
    aggregator = StatisticalAggregator(df)
    
    # Comprehensive statistics
    comprehensive = aggregator.comprehensive_group_stats(['region'])
    print(f"Comprehensive statistics by region:")
    print(comprehensive.round(2))
    
    # Generate report
    report = aggregator.generate_group_report(['region'])
    print(f"\nGroup Analysis Report:")
    print(f"Total groups: {report['group_sizes']['total_groups']}")
    print(f"Average group size: {report['group_sizes']['avg_group_size']:.1f}")
    
    if 'sales_group_comparison' in report:
        sales_comparison = report['sales_group_comparison']
        print(f"Sales comparison:")
        print(f"  Highest: {sales_comparison['highest_group']}")
        print(f"  Lowest: {sales_comparison['lowest_group']}")
        print(f"  Difference: {sales_comparison['mean_difference']:.2f}")

# Execute demonstration
if __name__ == "__main__":
    run_comprehensive_aggregation_demo()
```

#### Explanation
1. **Basic Aggregation**: Standard statistical functions (sum, mean, count, etc.) applied to grouped data
2. **Custom Functions**: User-defined aggregation functions for specialized calculations
3. **Named Aggregation**: Clean column naming for multiple aggregation functions
4. **Transform Operations**: Adding group-level statistics back to original DataFrame
5. **Filter Operations**: Selecting groups based on group-level conditions
6. **Time-based Aggregation**: Temporal grouping and rolling window calculations

#### Use Cases
- **Business Analytics**: Regional performance analysis, product comparisons
- **Financial Analysis**: Portfolio statistics, risk metrics by category
- **Scientific Research**: Experimental group comparisons, statistical testing
- **Operations Research**: Performance metrics by department, efficiency analysis
- **Marketing Analytics**: Customer segmentation analysis, campaign effectiveness

#### Best Practices
- **Choose Appropriate Functions**: Select aggregation functions that match your analytical goals
- **Handle Missing Data**: Decide how to treat NaN values in calculations
- **Use Named Aggregation**: Employ named aggregation for cleaner column names
- **Optimize Performance**: Use built-in functions when possible for better performance
- **Validate Results**: Check aggregation results for reasonableness and accuracy

#### Pitfalls
- **Memory Usage**: Large groupby operations can consume significant memory
- **Mixed Data Types**: Aggregating mixed types may produce unexpected results
- **Hierarchical Columns**: Multi-level column names can be confusing
- **Empty Groups**: Some operations may create empty groups
- **Performance Degradation**: Complex custom functions can be slow

#### Debugging
```python
def debug_groupby_operation(df: pd.DataFrame, groupby_cols: Union[str, List[str]]):
    """Debug groupby operations."""
    
    print("Groupby Operation Debug Info:")
    print(f"DataFrame shape: {df.shape}")
    print(f"Groupby columns: {groupby_cols}")
    
    # Check group information
    groups = df.groupby(groupby_cols)
    print(f"Number of groups: {groups.ngroups}")
    
    group_sizes = groups.size()
    print(f"Group sizes - Min: {group_sizes.min()}, Max: {group_sizes.max()}, Mean: {group_sizes.mean():.1f}")
    
    # Check for empty groups
    empty_groups = group_sizes[group_sizes == 0]
    if len(empty_groups) > 0:
        print(f"Warning: {len(empty_groups)} empty groups found")
    
    # Check data types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Numeric columns available for aggregation: {numeric_cols}")

def validate_aggregation_results(original_df: pd.DataFrame, aggregated_df: pd.DataFrame, 
                                groupby_cols: List[str], value_col: str):
    """Validate aggregation results."""
    
    print("Aggregation Validation:")
    
    # Check total preservation for sum operations
    original_total = original_df[value_col].sum()
    if 'sum' in aggregated_df.columns:
        aggregated_total = aggregated_df['sum'].sum()
        print(f"Sum preservation: Original={original_total:.2f}, Aggregated={aggregated_total:.2f}")
        
    # Check group coverage
    original_groups = set(original_df[groupby_cols[0]].unique())
    aggregated_groups = set(aggregated_df.index.get_level_values(0).unique())
    missing_groups = original_groups - aggregated_groups
    if missing_groups:
        print(f"Warning: Missing groups in aggregation: {missing_groups}")
```

#### Optimization

**Performance Optimization Tips:**

| Scenario | Optimization Strategy |
|----------|----------------------|
| **Large DataFrames** | Use categorical data types for groupby columns |
| **Many Groups** | Consider chunking or parallel processing |
| **Complex Aggregations** | Use built-in functions over custom lambdas |
| **Memory Constraints** | Process in batches or use out-of-core solutions |
| **Repeated Operations** | Cache intermediate groupby objects |

**Memory Efficiency:**
- Convert groupby columns to categorical type before grouping
- Use appropriate numeric data types (int32 vs int64)
- Drop unnecessary columns before aggregation
- Consider using sparse data structures for sparse results

---

## Question 9

**How do you usewindow functionsinPandasfor running calculations?**

### Answer

#### Theory
Window functions in Pandas provide powerful capabilities for performing calculations over sliding windows of data. These functions enable rolling, expanding, and exponentially weighted calculations that are essential for time series analysis, trend detection, smoothing data, and statistical modeling. Window functions maintain the original DataFrame structure while adding computed values based on neighboring data points.

#### Code Example

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Optional, List, Dict, Callable
import warnings

# ======================== BASIC WINDOW FUNCTIONS ========================

def demonstrate_rolling_windows():
    """Demonstrate basic rolling window calculations."""
    
    print("=== BASIC ROLLING WINDOW FUNCTIONS ===")
    
    # Create time series data
    df = create_stock_price_data()
    print("Stock Price Data:")
    print(df.head(10))
    
    # 1. Basic rolling calculations
    print(f"\n1. BASIC ROLLING CALCULATIONS:")
    
    # Rolling mean (simple moving average)
    df['sma_5'] = df['price'].rolling(window=5).mean()
    df['sma_20'] = df['price'].rolling(window=20).mean()
    
    # Rolling standard deviation
    df['rolling_std_5'] = df['price'].rolling(window=5).std()
    
    # Rolling min/max
    df['rolling_min_10'] = df['price'].rolling(window=10).min()
    df['rolling_max_10'] = df['price'].rolling(window=10).max()
    
    # Rolling sum
    df['volume_sum_5'] = df['volume'].rolling(window=5).sum()
    
    print("Data with rolling calculations:")
    print(df[['date', 'price', 'sma_5', 'sma_20', 'rolling_std_5']].head(25))
    
    # 2. Rolling quantiles
    print(f"\n2. ROLLING QUANTILES:")
    
    df['rolling_q25'] = df['price'].rolling(window=20).quantile(0.25)
    df['rolling_q75'] = df['price'].rolling(window=20).quantile(0.75)
    df['rolling_median'] = df['price'].rolling(window=20).median()
    
    print("Rolling quantiles:")
    print(df[['date', 'price', 'rolling_q25', 'rolling_median', 'rolling_q75']].head(25))
    
    # 3. Custom rolling functions
    print(f"\n3. CUSTOM ROLLING FUNCTIONS:")
    
    def rolling_volatility(prices):
        """Calculate rolling volatility (coefficient of variation)."""
        if len(prices) < 2:
            return np.nan
        return prices.std() / prices.mean() if prices.mean() != 0 else np.nan
    
    def rolling_trend_strength(prices):
        """Calculate trend strength using linear regression slope."""
        if len(prices) < 3:
            return np.nan
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        return slope / prices.mean() if prices.mean() != 0 else np.nan
    
    df['rolling_volatility'] = df['price'].rolling(window=10).apply(rolling_volatility)
    df['trend_strength'] = df['price'].rolling(window=10).apply(rolling_trend_strength)
    
    print("Custom rolling functions:")
    print(df[['date', 'price', 'rolling_volatility', 'trend_strength']].head(15))
    
    return df

def demonstrate_expanding_windows():
    """Demonstrate expanding window calculations."""
    
    print(f"\n=== EXPANDING WINDOW FUNCTIONS ===")
    
    df = create_stock_price_data()
    
    # 1. Basic expanding calculations
    print(f"\n1. BASIC EXPANDING CALCULATIONS:")
    
    # Expanding mean (cumulative average)
    df['expanding_mean'] = df['price'].expanding().mean()
    
    # Expanding standard deviation
    df['expanding_std'] = df['price'].expanding().std()
    
    # Expanding min/max
    df['expanding_min'] = df['price'].expanding().min()
    df['expanding_max'] = df['price'].expanding().max()
    
    print("Expanding calculations:")
    print(df[['date', 'price', 'expanding_mean', 'expanding_std', 
              'expanding_min', 'expanding_max']].head(15))
    
    # 2. Expanding with minimum periods
    print(f"\n2. EXPANDING WITH MINIMUM PERIODS:")
    
    # Require minimum 5 observations
    df['expanding_mean_min5'] = df['price'].expanding(min_periods=5).mean()
    df['expanding_std_min5'] = df['price'].expanding(min_periods=5).std()
    
    print("Expanding with minimum periods:")
    print(df[['date', 'price', 'expanding_mean_min5', 'expanding_std_min5']].head(10))
    
    # 3. Expanding custom functions
    print(f"\n3. EXPANDING CUSTOM FUNCTIONS:")
    
    def expanding_sharpe_ratio(returns, risk_free_rate=0.02):
        """Calculate expanding Sharpe ratio."""
        if len(returns) < 10:
            return np.nan
        excess_returns = returns - risk_free_rate/252  # Daily risk-free rate
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() != 0 else np.nan
    
    # Calculate returns first
    df['returns'] = df['price'].pct_change()
    df['expanding_sharpe'] = df['returns'].expanding(min_periods=10).apply(expanding_sharpe_ratio)
    
    print("Expanding Sharpe ratio:")
    print(df[['date', 'price', 'returns', 'expanding_sharpe']].head(15))
    
    return df

def demonstrate_ewm_windows():
    """Demonstrate exponentially weighted moving calculations."""
    
    print(f"\n=== EXPONENTIALLY WEIGHTED MOVING FUNCTIONS ===")
    
    df = create_stock_price_data()
    
    # 1. Basic EWM calculations
    print(f"\n1. BASIC EWM CALCULATIONS:")
    
    # EWM with span parameter
    df['ewm_span_10'] = df['price'].ewm(span=10).mean()
    df['ewm_span_20'] = df['price'].ewm(span=20).mean()
    
    # EWM with alpha parameter
    df['ewm_alpha_01'] = df['price'].ewm(alpha=0.1).mean()
    df['ewm_alpha_03'] = df['price'].ewm(alpha=0.3).mean()
    
    # EWM standard deviation
    df['ewm_std'] = df['price'].ewm(span=10).std()
    
    print("EWM calculations:")
    print(df[['date', 'price', 'ewm_span_10', 'ewm_span_20', 
              'ewm_alpha_01', 'ewm_std']].head(15))
    
    # 2. EWM with adjust parameter
    print(f"\n2. EWM WITH ADJUST PARAMETER:")
    
    # Without adjustment (gives more weight to recent observations)
    df['ewm_no_adjust'] = df['price'].ewm(span=10, adjust=False).mean()
    
    # With adjustment (default, more conservative)
    df['ewm_adjust'] = df['price'].ewm(span=10, adjust=True).mean()
    
    print("EWM with/without adjustment:")
    print(df[['date', 'price', 'ewm_no_adjust', 'ewm_adjust']].head(15))
    
    # 3. EWM for volatility estimation
    print(f"\n3. EWM VOLATILITY ESTIMATION:")
    
    df['returns'] = df['price'].pct_change()
    
    # EWMA volatility
    df['ewm_volatility'] = df['returns'].ewm(span=30).std() * np.sqrt(252)  # Annualized
    
    # GARCH-like volatility (exponentially weighted squared returns)
    df['squared_returns'] = df['returns'] ** 2
    df['ewm_var'] = df['squared_returns'].ewm(span=30).mean()
    df['ewm_vol_garch'] = np.sqrt(df['ewm_var'] * 252)  # Annualized volatility
    
    print("EWM volatility estimation:")
    print(df[['date', 'returns', 'ewm_volatility', 'ewm_vol_garch']].head(15))
    
    return df

def demonstrate_advanced_window_techniques():
    """Demonstrate advanced window function techniques."""
    
    print(f"\n=== ADVANCED WINDOW TECHNIQUES ===")
    
    df = create_stock_price_data()
    
    # 1. Grouped window functions
    print(f"\n1. GROUPED WINDOW FUNCTIONS:")
    
    # Add sector information
    df['sector'] = np.random.choice(['Tech', 'Finance', 'Healthcare'], len(df))
    
    # Rolling means by sector
    df['sector_rolling_mean'] = df.groupby('sector')['price'].rolling(window=10).mean().reset_index(0, drop=True)
    df['sector_rolling_std'] = df.groupby('sector')['price'].rolling(window=10).std().reset_index(0, drop=True)
    
    print("Grouped rolling calculations:")
    print(df[['date', 'sector', 'price', 'sector_rolling_mean', 'sector_rolling_std']].head(15))
    
    # 2. Multiple window sizes
    print(f"\n2. MULTIPLE WINDOW SIZES:")
    
    window_sizes = [5, 10, 20, 50]
    for window in window_sizes:
        df[f'sma_{window}'] = df['price'].rolling(window=window).mean()
        df[f'std_{window}'] = df['price'].rolling(window=window).std()
    
    # Calculate relative position within bands
    df['price_position'] = (df['price'] - df['sma_20']) / df['std_20']
    
    print("Multiple window sizes:")
    print(df[['date', 'price', 'sma_5', 'sma_10', 'sma_20', 'sma_50', 'price_position']].head(15))
    
    # 3. Window-based signals
    print(f"\n3. WINDOW-BASED SIGNALS:")
    
    # Golden Cross signal (SMA 50 > SMA 200)
    df['sma_200'] = df['price'].rolling(window=200).mean()
    df['golden_cross'] = (df['sma_50'] > df['sma_200']) & (df['sma_50'].shift(1) <= df['sma_200'].shift(1))
    
    # Bollinger Bands
    df['bb_upper'] = df['sma_20'] + (2 * df['std_20'])
    df['bb_lower'] = df['sma_20'] - (2 * df['std_20'])
    df['bb_signal'] = np.where(df['price'] > df['bb_upper'], 'Overbought',
                              np.where(df['price'] < df['bb_lower'], 'Oversold', 'Normal'))
    
    # RSI approximation using rolling calculations
    df['returns'] = df['price'].pct_change()
    df['gains'] = np.where(df['returns'] > 0, df['returns'], 0)
    df['losses'] = np.where(df['returns'] < 0, -df['returns'], 0)
    df['avg_gains'] = df['gains'].rolling(window=14).mean()
    df['avg_losses'] = df['losses'].rolling(window=14).mean()
    df['rs'] = df['avg_gains'] / df['avg_losses']
    df['rsi'] = 100 - (100 / (1 + df['rs']))
    
    print("Window-based signals:")
    print(df[['date', 'price', 'bb_signal', 'rsi']].head(15))
    
    return df

# ======================== WINDOW FUNCTIONS UTILITY CLASS ========================

class WindowCalculator:
    """Advanced window functions utility for time series analysis."""
    
    def __init__(self, dataframe: pd.DataFrame, value_col: str, date_col: Optional[str] = None):
        """Initialize with DataFrame and value column."""
        self.df = dataframe.copy()
        self.value_col = value_col
        self.date_col = date_col
        
        if date_col and date_col in self.df.columns:
            self.df[date_col] = pd.to_datetime(self.df[date_col])
            self.df = self.df.sort_values(date_col)
    
    def add_technical_indicators(self, short_window: int = 12, long_window: int = 26, 
                               signal_window: int = 9) -> pd.DataFrame:
        """Add common technical indicators using window functions."""
        
        # Simple Moving Averages
        self.df[f'sma_{short_window}'] = self.df[self.value_col].rolling(window=short_window).mean()
        self.df[f'sma_{long_window}'] = self.df[self.value_col].rolling(window=long_window).mean()
        
        # Exponential Moving Averages
        self.df[f'ema_{short_window}'] = self.df[self.value_col].ewm(span=short_window).mean()
        self.df[f'ema_{long_window}'] = self.df[self.value_col].ewm(span=long_window).mean()
        
        # MACD
        self.df['macd'] = self.df[f'ema_{short_window}'] - self.df[f'ema_{long_window}']
        self.df['macd_signal'] = self.df['macd'].ewm(span=signal_window).mean()
        self.df['macd_histogram'] = self.df['macd'] - self.df['macd_signal']
        
        # Bollinger Bands
        bb_window = 20
        self.df[f'bb_middle'] = self.df[self.value_col].rolling(window=bb_window).mean()
        bb_std = self.df[self.value_col].rolling(window=bb_window).std()
        self.df['bb_upper'] = self.df[f'bb_middle'] + (2 * bb_std)
        self.df['bb_lower'] = self.df[f'bb_middle'] - (2 * bb_std)
        self.df['bb_width'] = self.df['bb_upper'] - self.df['bb_lower']
        self.df['bb_position'] = (self.df[self.value_col] - self.df['bb_lower']) / self.df['bb_width']
        
        return self.df
    
    def add_volatility_measures(self, windows: List[int] = [10, 20, 30]) -> pd.DataFrame:
        """Add various volatility measures."""
        
        # Calculate returns
        self.df['returns'] = self.df[self.value_col].pct_change()
        
        for window in windows:
            # Historical volatility (rolling standard deviation)
            self.df[f'hist_vol_{window}'] = self.df['returns'].rolling(window=window).std() * np.sqrt(252)
            
            # Parkinson volatility (high-low estimator)
            if 'high' in self.df.columns and 'low' in self.df.columns:
                hl_ratio = np.log(self.df['high'] / self.df['low'])
                self.df[f'parkinson_vol_{window}'] = np.sqrt(
                    hl_ratio.rolling(window=window).apply(lambda x: (x**2).sum() / (4 * np.log(2) * len(x)))
                ) * np.sqrt(252)
            
            # EWMA volatility
            self.df[f'ewma_vol_{window}'] = self.df['returns'].ewm(span=window).std() * np.sqrt(252)
            
            # Yang-Zhang volatility (advanced estimator)
            if all(col in self.df.columns for col in ['open', 'high', 'low', 'close']):
                self._add_yang_zhang_volatility(window)
        
        return self.df
    
    def add_momentum_indicators(self, periods: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """Add momentum-based indicators."""
        
        for period in periods:
            # Rate of Change (ROC)
            self.df[f'roc_{period}'] = ((self.df[self.value_col] / self.df[self.value_col].shift(period)) - 1) * 100
            
            # Momentum
            self.df[f'momentum_{period}'] = self.df[self.value_col] - self.df[self.value_col].shift(period)
            
            # Relative Strength Index (RSI)
            self._add_rsi(period)
            
            # Williams %R
            if 'high' in self.df.columns and 'low' in self.df.columns:
                highest_high = self.df['high'].rolling(window=period).max()
                lowest_low = self.df['low'].rolling(window=period).min()
                self.df[f'williams_r_{period}'] = -100 * (
                    (highest_high - self.df[self.value_col]) / (highest_high - lowest_low)
                )
        
        return self.df
    
    def add_trend_indicators(self, window: int = 14) -> pd.DataFrame:
        """Add trend-based indicators."""
        
        # Average Directional Index (ADX) approximation
        if all(col in self.df.columns for col in ['high', 'low']):
            self._add_adx_approximation(window)
        
        # Parabolic SAR approximation
        self._add_parabolic_sar()
        
        # Commodity Channel Index (CCI)
        if all(col in self.df.columns for col in ['high', 'low']):
            typical_price = (self.df['high'] + self.df['low'] + self.df[self.value_col]) / 3
            sma_tp = typical_price.rolling(window=window).mean()
            mad = typical_price.rolling(window=window).apply(lambda x: np.mean(np.abs(x - x.mean())))
            self.df[f'cci_{window}'] = (typical_price - sma_tp) / (0.015 * mad)
        
        return self.df
    
    def _add_rsi(self, period: int):
        """Add RSI calculation."""
        delta = self.df[self.value_col].diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()
        
        rs = avg_gains / avg_losses
        self.df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    def _add_yang_zhang_volatility(self, window: int):
        """Add Yang-Zhang volatility estimator."""
        # This is a simplified version
        o = np.log(self.df['open'] / self.df['close'].shift(1))
        c = np.log(self.df['close'] / self.df['open'])
        h = np.log(self.df['high'] / self.df['close'])
        l = np.log(self.df['low'] / self.df['close'])
        
        yz = o**2 + 0.5*(h + l)**2 - (2*np.log(2) - 1)*c**2
        self.df[f'yz_vol_{window}'] = np.sqrt(yz.rolling(window=window).mean() * 252)
    
    def _add_adx_approximation(self, window: int):
        """Add simplified ADX calculation."""
        # Simplified ADX using rolling calculations
        high_diff = self.df['high'] - self.df['high'].shift(1)
        low_diff = self.df['low'].shift(1) - self.df['low']
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        tr = np.maximum(
            self.df['high'] - self.df['low'],
            np.maximum(
                np.abs(self.df['high'] - self.df[self.value_col].shift(1)),
                np.abs(self.df['low'] - self.df[self.value_col].shift(1))
            )
        )
        
        plus_di = 100 * (pd.Series(plus_dm).rolling(window=window).sum() / 
                        pd.Series(tr).rolling(window=window).sum())
        minus_di = 100 * (pd.Series(minus_dm).rolling(window=window).sum() / 
                         pd.Series(tr).rolling(window=window).sum())
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        self.df[f'adx_{window}'] = dx.rolling(window=window).mean()
    
    def _add_parabolic_sar(self, af_start: float = 0.02, af_max: float = 0.2):
        """Add simplified Parabolic SAR."""
        # Simplified Parabolic SAR
        high = self.df['high'] if 'high' in self.df.columns else self.df[self.value_col]
        low = self.df['low'] if 'low' in self.df.columns else self.df[self.value_col]
        
        sar = [low.iloc[0]]
        trend = [1]  # 1 for uptrend, -1 for downtrend
        af = [af_start]
        
        for i in range(1, len(self.df)):
            if trend[i-1] == 1:  # Uptrend
                sar_val = sar[i-1] + af[i-1] * (high.iloc[i-1] - sar[i-1])
                if low.iloc[i] <= sar_val:
                    trend.append(-1)
                    sar.append(high.iloc[i-1])
                    af.append(af_start)
                else:
                    trend.append(1)
                    sar.append(max(sar_val, low.iloc[i-1], low.iloc[i-2] if i > 1 else low.iloc[i-1]))
                    af.append(min(af_max, af[i-1] + af_start) if high.iloc[i] > high.iloc[i-1] else af[i-1])
            else:  # Downtrend
                sar_val = sar[i-1] - af[i-1] * (sar[i-1] - low.iloc[i-1])
                if high.iloc[i] >= sar_val:
                    trend.append(1)
                    sar.append(low.iloc[i-1])
                    af.append(af_start)
                else:
                    trend.append(-1)
                    sar.append(min(sar_val, high.iloc[i-1], high.iloc[i-2] if i > 1 else high.iloc[i-1]))
                    af.append(min(af_max, af[i-1] + af_start) if low.iloc[i] < low.iloc[i-1] else af[i-1])
        
        self.df['parabolic_sar'] = sar

# ======================== HELPER FUNCTIONS ========================

def create_stock_price_data():
    """Create realistic stock price time series data."""
    
    np.random.seed(42)
    
    # Create date range
    dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
    
    # Generate realistic price data with trends and volatility
    n_days = len(dates)
    returns = np.random.normal(0.0008, 0.02, n_days)  # Daily returns with small positive drift
    
    # Add some volatility clustering
    volatility = np.ones(n_days) * 0.02
    for i in range(1, n_days):
        volatility[i] = 0.95 * volatility[i-1] + 0.05 * np.abs(returns[i-1])
        returns[i] = np.random.normal(0.0008, volatility[i])
    
    # Calculate prices
    prices = [100]  # Starting price
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Generate volume data
    volumes = np.random.lognormal(12, 0.5, n_days).astype(int)
    
    # Generate OHLC data
    highs = [p * (1 + np.random.uniform(0, 0.03)) for p in prices]
    lows = [p * (1 - np.random.uniform(0, 0.03)) for p in prices]
    opens = [prices[0]] + [prices[i-1] * (1 + np.random.uniform(-0.01, 0.01)) for i in range(1, n_days)]
    
    data = {
        'date': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': prices,
        'price': prices,  # Using close as price
        'volume': volumes
    }
    
    return pd.DataFrame(data)

# ======================== MAIN DEMONSTRATION ========================

def run_comprehensive_window_demo():
    """Run comprehensive window functions demonstration."""
    
    print("PANDAS WINDOW FUNCTIONS COMPREHENSIVE GUIDE")
    print("="*45)
    
    # Basic rolling windows
    df_rolling = demonstrate_rolling_windows()
    
    # Expanding windows
    df_expanding = demonstrate_expanding_windows()
    
    # Exponentially weighted moving functions
    df_ewm = demonstrate_ewm_windows()
    
    # Advanced window techniques
    df_advanced = demonstrate_advanced_window_techniques()
    
    # Demonstrate WindowCalculator class
    print(f"\n=== WINDOW CALCULATOR CLASS ===")
    
    stock_data = create_stock_price_data()
    calculator = WindowCalculator(stock_data, 'close', 'date')
    
    # Add technical indicators
    df_with_indicators = calculator.add_technical_indicators()
    print(f"Technical indicators added:")
    print(df_with_indicators[['date', 'close', 'sma_12', 'ema_12', 'macd', 'bb_position']].head(30))
    
    # Add volatility measures
    df_with_vol = calculator.add_volatility_measures()
    print(f"\nVolatility measures:")
    vol_cols = [col for col in df_with_vol.columns if 'vol' in col.lower()]
    print(df_with_vol[['date'] + vol_cols[:3]].head(15))

# Execute demonstration
if __name__ == "__main__":
    run_comprehensive_window_demo()
```

#### Explanation
1. **Rolling Windows**: Fixed-size sliding windows for moving averages, standard deviations, and custom calculations
2. **Expanding Windows**: Growing windows that include all previous observations
3. **Exponentially Weighted**: Recent observations get higher weight, useful for trend following
4. **Grouped Windows**: Apply window functions within groups of data
5. **Custom Functions**: User-defined calculations applied to windows

#### Use Cases
- **Financial Analysis**: Moving averages, volatility estimation, technical indicators
- **Time Series Smoothing**: Noise reduction, trend identification
- **Quality Control**: Rolling statistics for process monitoring
- **Weather Data**: Climate trend analysis, seasonal adjustments
- **IoT Analytics**: Sensor data smoothing, anomaly detection

#### Best Practices
- **Window Size Selection**: Choose appropriate window sizes based on data frequency and analysis goals
- **Minimum Periods**: Set minimum periods to handle edge cases
- **Performance Optimization**: Use built-in functions when possible
- **Missing Data Handling**: Decide how to handle NaN values in calculations
- **Memory Management**: Be aware of memory usage with large datasets

#### Pitfalls
- **Look-ahead Bias**: Ensure calculations only use past data
- **Window Size Impact**: Small windows are noisy, large windows are lagged
- **Edge Effects**: First few observations may have unreliable calculations
- **Computational Cost**: Complex window functions can be slow
- **Parameter Sensitivity**: Results may be sensitive to window parameters

#### Debugging
```python
def debug_window_function(df: pd.DataFrame, column: str, window_size: int):
    """Debug window function calculations."""
    
    print(f"Window Function Debug - Column: {column}, Window: {window_size}")
    print(f"Data shape: {df.shape}")
    print(f"Data type: {df[column].dtype}")
    
    # Check for missing values
    missing_count = df[column].isnull().sum()
    print(f"Missing values: {missing_count}")
    
    # Calculate rolling mean for debugging
    rolling_mean = df[column].rolling(window=window_size)
    print(f"Rolling window object: {rolling_mean}")
    
    # Show first few calculations
    result = rolling_mean.mean()
    print(f"First non-null rolling mean index: {result.first_valid_index()}")
    print(f"First few rolling means:")
    print(result.head(window_size + 5))

def validate_window_calculations(df: pd.DataFrame, column: str, window_size: int):
    """Validate window calculation results."""
    
    print("Window Calculation Validation:")
    
    # Manual calculation for first complete window
    manual_mean = df[column].iloc[:window_size].mean()
    rolling_mean = df[column].rolling(window=window_size).mean()
    
    print(f"Manual mean of first {window_size} values: {manual_mean}")
    print(f"Rolling mean at position {window_size-1}: {rolling_mean.iloc[window_size-1]}")
    print(f"Match: {np.isclose(manual_mean, rolling_mean.iloc[window_size-1], equal_nan=True)}")
```

#### Optimization

**Window Function Performance Tips:**

| Function Type | Optimization Strategy |
|---------------|----------------------|
| **Rolling** | Use built-in methods over apply() when possible |
| **Expanding** | Consider memory usage for very long series |
| **EWM** | Choose span vs alpha based on use case |
| **Custom Functions** | Vectorize operations when possible |
| **Grouped Windows** | Sort data by groups for better performance |

**Memory and Performance:**
- Use appropriate data types (float32 vs float64)
- Process large datasets in chunks
- Consider using numba for custom window functions
- Pre-sort data when using grouped window functions

---

## Question 10

**Provide an example of how tonormalize datawithin aDataFramecolumn.**

### Answer

#### Theory
Data normalization is a crucial preprocessing step that transforms data to a standard scale, making it suitable for machine learning algorithms, statistical analysis, and data comparison. Different normalization techniques serve different purposes: Min-Max scaling maps data to a fixed range, Z-score standardization centers data around zero with unit variance, and robust scaling uses quartiles to handle outliers. Proper normalization ensures features contribute equally to analysis and prevents scale-dependent algorithms from being biased.

#### Code Example

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer
from scipy import stats
import matplotlib.pyplot as plt
from typing import Union, List, Optional, Dict, Tuple

# ======================== BASIC NORMALIZATION TECHNIQUES ========================

def demonstrate_basic_normalization():
    """Demonstrate basic normalization techniques."""
    
    print("=== BASIC NORMALIZATION TECHNIQUES ===")
    
    # Create sample dataset with different scales
    df = create_sample_dataset()
    print("Original Dataset:")
    print(df.describe())
    
    # 1. Min-Max Normalization (0-1 scaling)
    print(f"\n1. MIN-MAX NORMALIZATION (0-1 Scaling):")
    
    df_minmax = df.copy()
    
    # Method 1: Manual Min-Max scaling
    for col in ['sales', 'price', 'quantity']:
        col_min = df[col].min()
        col_max = df[col].max()
        df_minmax[f'{col}_minmax'] = (df[col] - col_min) / (col_max - col_min)
    
    # Method 2: Using built-in function
    df_minmax['sales_minmax_builtin'] = (df['sales'] - df['sales'].min()) / (df['sales'].max() - df['sales'].min())
    
    print("Min-Max normalized data:")
    print(df_minmax[['sales', 'sales_minmax', 'sales_minmax_builtin']].head())
    print(f"Min-Max range check - Min: {df_minmax['sales_minmax'].min():.3f}, Max: {df_minmax['sales_minmax'].max():.3f}")
    
    # 2. Z-Score Normalization (Standardization)
    print(f"\n2. Z-SCORE NORMALIZATION (Standardization):")
    
    df_zscore = df.copy()
    
    # Method 1: Manual Z-score
    for col in ['sales', 'price', 'quantity']:
        col_mean = df[col].mean()
        col_std = df[col].std()
        df_zscore[f'{col}_zscore'] = (df[col] - col_mean) / col_std
    
    # Method 2: Using built-in function
    df_zscore['sales_zscore_builtin'] = (df['sales'] - df['sales'].mean()) / df['sales'].std()
    
    print("Z-score normalized data:")
    print(df_zscore[['sales', 'sales_zscore', 'sales_zscore_builtin']].head())
    print(f"Z-score stats - Mean: {df_zscore['sales_zscore'].mean():.6f}, Std: {df_zscore['sales_zscore'].std():.3f}")
    
    # 3. Custom Range Normalization
    print(f"\n3. CUSTOM RANGE NORMALIZATION:")
    
    def normalize_to_range(series, new_min, new_max):
        """Normalize series to custom range."""
        old_min, old_max = series.min(), series.max()
        old_range = old_max - old_min
        new_range = new_max - new_min
        
        return (series - old_min) * new_range / old_range + new_min
    
    df['sales_custom'] = normalize_to_range(df['sales'], -1, 1)  # Normalize to [-1, 1]
    df['price_custom'] = normalize_to_range(df['price'], 0, 100)  # Normalize to [0, 100]
    
    print("Custom range normalization:")
    print(df[['sales', 'sales_custom', 'price', 'price_custom']].head())
    
    return df, df_minmax, df_zscore

def demonstrate_advanced_normalization():
    """Demonstrate advanced normalization techniques."""
    
    print(f"\n=== ADVANCED NORMALIZATION TECHNIQUES ===")
    
    df = create_sample_dataset()
    
    # 1. Robust Scaling (using median and IQR)
    print(f"\n1. ROBUST SCALING (Median-IQR):")
    
    def robust_scale(series):
        """Robust scaling using median and IQR."""
        median = series.median()
        q75 = series.quantile(0.75)
        q25 = series.quantile(0.25)
        iqr = q75 - q25
        
        if iqr == 0:
            return pd.Series(np.zeros(len(series)), index=series.index)
        
        return (series - median) / iqr
    
    df['sales_robust'] = robust_scale(df['sales'])
    df['price_robust'] = robust_scale(df['price'])
    
    print("Robust scaling results:")
    print(df[['sales', 'sales_robust', 'price', 'price_robust']].head())
    
    # Compare with outliers
    df_with_outliers = df.copy()
    df_with_outliers.loc[0, 'sales'] = 1000000  # Add extreme outlier
    
    df_with_outliers['sales_zscore'] = (df_with_outliers['sales'] - df_with_outliers['sales'].mean()) / df_with_outliers['sales'].std()
    df_with_outliers['sales_robust'] = robust_scale(df_with_outliers['sales'])
    
    print(f"\nRobust vs Z-score with outliers:")
    print(f"Z-score range: {df_with_outliers['sales_zscore'].min():.2f} to {df_with_outliers['sales_zscore'].max():.2f}")
    print(f"Robust range: {df_with_outliers['sales_robust'].min():.2f} to {df_with_outliers['sales_robust'].max():.2f}")
    
    # 2. Unit Vector Scaling (L2 normalization)
    print(f"\n2. UNIT VECTOR SCALING (L2 Normalization):")
    
    def unit_vector_scale(series):
        """Scale to unit vector (L2 norm = 1)."""
        l2_norm = np.sqrt((series ** 2).sum())
        if l2_norm == 0:
            return series
        return series / l2_norm
    
    df['sales_unit'] = unit_vector_scale(df['sales'])
    df['price_unit'] = unit_vector_scale(df['price'])
    
    print("Unit vector scaling:")
    print(df[['sales', 'sales_unit', 'price', 'price_unit']].head())
    print(f"L2 norm check - Sales: {np.sqrt((df['sales_unit'] ** 2).sum()):.6f}")
    
    # 3. Decimal Scaling
    print(f"\n3. DECIMAL SCALING:")
    
    def decimal_scale(series):
        """Decimal scaling normalization."""
        max_abs = series.abs().max()
        if max_abs == 0:
            return series
        
        # Find appropriate power of 10
        power = int(np.ceil(np.log10(max_abs)))
        return series / (10 ** power)
    
    df['sales_decimal'] = decimal_scale(df['sales'])
    df['price_decimal'] = decimal_scale(df['price'])
    
    print("Decimal scaling:")
    print(df[['sales', 'sales_decimal', 'price', 'price_decimal']].head())
    
    return df

def demonstrate_distribution_normalization():
    """Demonstrate normalization for distribution transformation."""
    
    print(f"\n=== DISTRIBUTION NORMALIZATION ===")
    
    # Create skewed data
    df_skewed = create_skewed_dataset()
    print("Skewed dataset statistics:")
    print(df_skewed.describe())
    
    # 1. Log Transformation
    print(f"\n1. LOG TRANSFORMATION:")
    
    # Add small constant to handle zeros
    df_skewed['log_sales'] = np.log1p(df_skewed['sales'])  # log(1 + x)
    df_skewed['log10_sales'] = np.log10(df_skewed['sales'] + 1)
    
    print("Log transformation:")
    print(df_skewed[['sales', 'log_sales', 'log10_sales']].head())
    print(f"Original skewness: {df_skewed['sales'].skew():.3f}")
    print(f"Log-transformed skewness: {df_skewed['log_sales'].skew():.3f}")
    
    # 2. Square Root Transformation
    print(f"\n2. SQUARE ROOT TRANSFORMATION:")
    
    df_skewed['sqrt_sales'] = np.sqrt(df_skewed['sales'])
    
    print("Square root transformation:")
    print(df_skewed[['sales', 'sqrt_sales']].head())
    print(f"Sqrt-transformed skewness: {df_skewed['sqrt_sales'].skew():.3f}")
    
    # 3. Box-Cox Transformation
    print(f"\n3. BOX-COX TRANSFORMATION:")
    
    # Box-Cox requires positive values
    positive_sales = df_skewed['sales'] + 1  # Ensure positive
    
    # Find optimal lambda
    transformed_data, optimal_lambda = stats.boxcox(positive_sales)
    df_skewed['boxcox_sales'] = transformed_data
    
    print(f"Box-Cox transformation (lambda = {optimal_lambda:.3f}):")
    print(df_skewed[['sales', 'boxcox_sales']].head())
    print(f"Box-Cox skewness: {df_skewed['boxcox_sales'].skew():.3f}")
    
    # 4. Yeo-Johnson Transformation (can handle negative values)
    print(f"\n4. YEO-JOHNSON TRANSFORMATION:")
    
    # Add some negative values for demonstration
    df_mixed = df_skewed.copy()
    df_mixed['mixed_values'] = df_skewed['sales'] - df_skewed['sales'].median()
    
    # Yeo-Johnson transformation
    transformed_yj, optimal_lambda_yj = stats.yeojohnson(df_mixed['mixed_values'])
    df_mixed['yeojohnson_values'] = transformed_yj
    
    print(f"Yeo-Johnson transformation (lambda = {optimal_lambda_yj:.3f}):")
    print(df_mixed[['mixed_values', 'yeojohnson_values']].head())
    print(f"Original skewness: {df_mixed['mixed_values'].skew():.3f}")
    print(f"Yeo-Johnson skewness: {df_mixed['yeojohnson_values'].skew():.3f}")
    
    return df_skewed, df_mixed

def demonstrate_sklearn_normalization():
    """Demonstrate normalization using scikit-learn."""
    
    print(f"\n=== SCIKIT-LEARN NORMALIZATION ===")
    
    df = create_sample_dataset()
    
    # 1. MinMaxScaler
    print(f"\n1. SKLEARN MINMAXSCALER:")
    
    scaler_minmax = MinMaxScaler()
    df_scaled = df.copy()
    
    # Scale specific columns
    columns_to_scale = ['sales', 'price', 'quantity']
    df_scaled[columns_to_scale] = scaler_minmax.fit_transform(df[columns_to_scale])
    
    print("MinMaxScaler results:")
    print(df_scaled[columns_to_scale].describe())
    
    # 2. StandardScaler
    print(f"\n2. SKLEARN STANDARDSCALER:")
    
    scaler_standard = StandardScaler()
    df_standard = df.copy()
    df_standard[columns_to_scale] = scaler_standard.fit_transform(df[columns_to_scale])
    
    print("StandardScaler results:")
    print(df_standard[columns_to_scale].describe())
    
    # 3. RobustScaler
    print(f"\n3. SKLEARN ROBUSTSCALER:")
    
    scaler_robust = RobustScaler()
    df_robust = df.copy()
    df_robust[columns_to_scale] = scaler_robust.fit_transform(df[columns_to_scale])
    
    print("RobustScaler results:")
    print(df_robust[columns_to_scale].describe())
    
    # 4. PowerTransformer (Yeo-Johnson)
    print(f"\n4. SKLEARN POWERTRANSFORMER:")
    
    power_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
    df_power = df.copy()
    df_power[columns_to_scale] = power_transformer.fit_transform(df[columns_to_scale])
    
    print("PowerTransformer results:")
    print(df_power[columns_to_scale].describe())
    
    return df_scaled, df_standard, df_robust, df_power

# ======================== NORMALIZATION UTILITY CLASS ========================

class DataNormalizer:
    """Comprehensive data normalization utility."""
    
    def __init__(self, dataframe: pd.DataFrame):
        """Initialize with DataFrame."""
        self.df = dataframe.copy()
        self.scaling_info = {}
    
    def normalize_columns(self, columns: List[str], method: str = 'minmax', 
                         custom_range: Optional[Tuple[float, float]] = None) -> pd.DataFrame:
        """Normalize specified columns using chosen method."""
        
        result_df = self.df.copy()
        
        for col in columns:
            if col not in self.df.columns:
                print(f"Warning: Column '{col}' not found in DataFrame")
                continue
            
            original_values = self.df[col]
            
            if method == 'minmax':
                col_min, col_max = original_values.min(), original_values.max()
                if col_max == col_min:
                    normalized = pd.Series(np.zeros(len(original_values)), index=original_values.index)
                else:
                    if custom_range:
                        new_min, new_max = custom_range
                        normalized = (original_values - col_min) / (col_max - col_min) * (new_max - new_min) + new_min
                    else:
                        normalized = (original_values - col_min) / (col_max - col_min)
                
                self.scaling_info[col] = {'method': 'minmax', 'min': col_min, 'max': col_max}
            
            elif method == 'zscore':
                col_mean, col_std = original_values.mean(), original_values.std()
                if col_std == 0:
                    normalized = pd.Series(np.zeros(len(original_values)), index=original_values.index)
                else:
                    normalized = (original_values - col_mean) / col_std
                
                self.scaling_info[col] = {'method': 'zscore', 'mean': col_mean, 'std': col_std}
            
            elif method == 'robust':
                col_median = original_values.median()
                q75, q25 = original_values.quantile(0.75), original_values.quantile(0.25)
                iqr = q75 - q25
                
                if iqr == 0:
                    normalized = pd.Series(np.zeros(len(original_values)), index=original_values.index)
                else:
                    normalized = (original_values - col_median) / iqr
                
                self.scaling_info[col] = {'method': 'robust', 'median': col_median, 'iqr': iqr}
            
            elif method == 'unit_vector':
                l2_norm = np.sqrt((original_values ** 2).sum())
                if l2_norm == 0:
                    normalized = original_values
                else:
                    normalized = original_values / l2_norm
                
                self.scaling_info[col] = {'method': 'unit_vector', 'l2_norm': l2_norm}
            
            else:
                raise ValueError(f"Unknown normalization method: {method}")
            
            result_df[col] = normalized
        
        return result_df
    
    def apply_transformation(self, columns: List[str], transformation: str) -> pd.DataFrame:
        """Apply distribution transformation to columns."""
        
        result_df = self.df.copy()
        
        for col in columns:
            if col not in self.df.columns:
                continue
            
            original_values = self.df[col]
            
            if transformation == 'log':
                # Use log1p to handle zeros
                transformed = np.log1p(original_values)
                
            elif transformation == 'sqrt':
                # Handle negative values
                transformed = np.sqrt(np.abs(original_values)) * np.sign(original_values)
                
            elif transformation == 'boxcox':
                # Box-Cox requires positive values
                shifted_values = original_values - original_values.min() + 1
                transformed, lambda_param = stats.boxcox(shifted_values)
                self.scaling_info[col] = {'transformation': 'boxcox', 'lambda': lambda_param, 'shift': original_values.min() - 1}
                
            elif transformation == 'yeojohnson':
                transformed, lambda_param = stats.yeojohnson(original_values)
                self.scaling_info[col] = {'transformation': 'yeojohnson', 'lambda': lambda_param}
                
            else:
                raise ValueError(f"Unknown transformation: {transformation}")
            
            result_df[col] = transformed
        
        return result_df
    
    def reverse_normalization(self, normalized_df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Reverse normalization to get original scale."""
        
        result_df = normalized_df.copy()
        
        for col in columns:
            if col not in self.scaling_info:
                print(f"Warning: No scaling info found for column '{col}'")
                continue
            
            info = self.scaling_info[col]
            normalized_values = normalized_df[col]
            
            if info['method'] == 'minmax':
                original = normalized_values * (info['max'] - info['min']) + info['min']
                
            elif info['method'] == 'zscore':
                original = normalized_values * info['std'] + info['mean']
                
            elif info['method'] == 'robust':
                original = normalized_values * info['iqr'] + info['median']
                
            elif info['method'] == 'unit_vector':
                original = normalized_values * info['l2_norm']
                
            else:
                continue
            
            result_df[col] = original
        
        return result_df
    
    def get_normalization_summary(self) -> pd.DataFrame:
        """Get summary of applied normalizations."""
        
        summary_data = []
        for col, info in self.scaling_info.items():
            summary_data.append({
                'column': col,
                'method': info['method'],
                'parameters': {k: v for k, v in info.items() if k != 'method'}
            })
        
        return pd.DataFrame(summary_data)
    
    def compare_normalization_methods(self, column: str) -> pd.DataFrame:
        """Compare different normalization methods for a column."""
        
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        
        original = self.df[column]
        comparisons = {}
        
        # Original statistics
        comparisons['original'] = {
            'mean': original.mean(),
            'std': original.std(),
            'min': original.min(),
            'max': original.max(),
            'range': original.max() - original.min(),
            'skewness': original.skew()
        }
        
        # Test different methods
        methods = ['minmax', 'zscore', 'robust', 'unit_vector']
        
        for method in methods:
            temp_normalizer = DataNormalizer(pd.DataFrame({column: original}))
            normalized_df = temp_normalizer.normalize_columns([column], method)
            normalized = normalized_df[column]
            
            comparisons[method] = {
                'mean': normalized.mean(),
                'std': normalized.std(),
                'min': normalized.min(),
                'max': normalized.max(),
                'range': normalized.max() - normalized.min(),
                'skewness': normalized.skew()
            }
        
        return pd.DataFrame(comparisons).T

# ======================== HELPER FUNCTIONS ========================

def create_sample_dataset():
    """Create sample dataset with different scales."""
    
    np.random.seed(42)
    
    data = {
        'sales': np.random.exponential(5000, 1000),  # Exponential distribution
        'price': np.random.normal(100, 25, 1000),    # Normal distribution
        'quantity': np.random.poisson(10, 1000),     # Poisson distribution
        'region': np.random.choice(['North', 'South', 'East', 'West'], 1000),
        'product': np.random.choice(['A', 'B', 'C', 'D'], 1000)
    }
    
    # Ensure positive prices
    data['price'] = np.abs(data['price'])
    
    return pd.DataFrame(data)

def create_skewed_dataset():
    """Create dataset with skewed distributions."""
    
    np.random.seed(42)
    
    # Create right-skewed data
    skewed_data = np.random.lognormal(mean=5, sigma=1, size=1000)
    
    data = {
        'sales': skewed_data,
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    return pd.DataFrame(data)

# ======================== MAIN DEMONSTRATION ========================

def run_comprehensive_normalization_demo():
    """Run comprehensive normalization demonstration."""
    
    print("PANDAS DATA NORMALIZATION COMPREHENSIVE GUIDE")
    print("="*47)
    
    # Basic normalization techniques
    df_orig, df_minmax, df_zscore = demonstrate_basic_normalization()
    
    # Advanced normalization techniques
    df_advanced = demonstrate_advanced_normalization()
    
    # Distribution normalization
    df_skewed, df_mixed = demonstrate_distribution_normalization()
    
    # Scikit-learn normalization
    df_scaled, df_standard, df_robust, df_power = demonstrate_sklearn_normalization()
    
    # Demonstrate DataNormalizer class
    print(f"\n=== DATA NORMALIZER CLASS ===")
    
    sample_df = create_sample_dataset()
    normalizer = DataNormalizer(sample_df)
    
    # Normalize with different methods
    normalized_minmax = normalizer.normalize_columns(['sales', 'price'], method='minmax')
    print(f"MinMax normalization:")
    print(normalized_minmax[['sales', 'price']].describe())
    
    # Compare normalization methods
    comparison = normalizer.compare_normalization_methods('sales')
    print(f"\nNormalization methods comparison for 'sales' column:")
    print(comparison.round(3))
    
    # Get normalization summary
    summary = normalizer.get_normalization_summary()
    print(f"\nNormalization summary:")
    print(summary)

# Execute demonstration
if __name__ == "__main__":
    run_comprehensive_normalization_demo()
```

#### Explanation
1. **Min-Max Normalization**: Scales data to a fixed range [0,1] or custom range
2. **Z-Score Standardization**: Centers data around zero with unit variance
3. **Robust Scaling**: Uses median and IQR, less sensitive to outliers
4. **Unit Vector Scaling**: Scales to unit L2 norm, useful for cosine similarity
5. **Distribution Transformation**: Log, Box-Cox, Yeo-Johnson for skewed data

#### Use Cases
- **Machine Learning**: Feature scaling for algorithms sensitive to scale (SVM, neural networks, k-means)
- **Statistical Analysis**: Comparing variables with different units and scales
- **Data Visualization**: Creating meaningful plots when variables have vastly different ranges
- **Distance Calculations**: Ensuring equal contribution of features in distance metrics
- **Financial Analysis**: Comparing financial metrics across different companies or time periods

#### Best Practices
- **Choose Appropriate Method**: Min-Max for bounded ranges, Z-score for normal distributions, robust for outliers
- **Preserve Information**: Store scaling parameters to reverse transformation
- **Handle Edge Cases**: Check for zero variance, infinite values, and missing data
- **Validate Results**: Verify normalization produces expected ranges and distributions
- **Consider Domain Knowledge**: Some domains may require specific normalization approaches

#### Pitfalls
- **Data Leakage**: Don't use test set statistics for normalization
- **Outlier Impact**: Min-Max scaling is sensitive to outliers
- **Zero Variance**: Columns with constant values cause division by zero
- **Distribution Assumptions**: Z-score assumes normal distribution
- **Lost Interpretability**: Normalized values lose original meaning

#### Debugging
```python
def debug_normalization(original: pd.Series, normalized: pd.Series, method: str):
    """Debug normalization results."""
    
    print(f"Normalization Debug - Method: {method}")
    print(f"Original stats:")
    print(f"  Range: {original.min():.3f} to {original.max():.3f}")
    print(f"  Mean: {original.mean():.3f}, Std: {original.std():.3f}")
    
    print(f"Normalized stats:")
    print(f"  Range: {normalized.min():.3f} to {normalized.max():.3f}")
    print(f"  Mean: {normalized.mean():.3f}, Std: {normalized.std():.3f}")
    
    # Check for issues
    if normalized.isnull().sum() > 0:
        print(f"Warning: {normalized.isnull().sum()} NaN values introduced")
    
    if np.isinf(normalized).sum() > 0:
        print(f"Warning: {np.isinf(normalized).sum()} infinite values introduced")

def validate_normalization_bounds(normalized: pd.Series, expected_min: float, expected_max: float):
    """Validate normalization bounds."""
    
    actual_min, actual_max = normalized.min(), normalized.max()
    
    print(f"Bounds validation:")
    print(f"Expected: [{expected_min}, {expected_max}]")
    print(f"Actual: [{actual_min:.6f}, {actual_max:.6f}]")
    
    if not (np.isclose(actual_min, expected_min) and np.isclose(actual_max, expected_max)):
        print("Warning: Normalization bounds don't match expected values")
    else:
        print("✓ Bounds validation passed")
```

#### Optimization

**Normalization Performance Tips:**

| Method | Optimization Strategy |
|--------|----------------------|
| **Large DataFrames** | Use vectorized operations, avoid loops |
| **Memory Constraints** | Process in chunks, use appropriate data types |
| **Repeated Operations** | Cache statistics, use fitted scalers |
| **Multiple Columns** | Batch process columns together |
| **Real-time Applications** | Pre-compute scaling parameters |

**Memory Efficiency:**
- Use appropriate numeric data types (float32 vs float64)
- Apply normalization in-place when possible
- Clear intermediate variables after use
- Consider sparse data structures for sparse datasets

---

## Question 11

**Show how to create simpleplotsfrom aDataFrameusingPandas’ visualization tools.**

**Answer:**

#### Theory
Pandas provides integrated plotting capabilities through the `.plot()` method, which serves as a convenient wrapper around Matplotlib. This integration allows for quick data visualization directly from DataFrames and Series without needing to explicitly import matplotlib in simple cases. The plotting functionality supports various chart types including line plots, bar charts, histograms, scatter plots, and more, making it ideal for exploratory data analysis and rapid prototyping of visualizations.

#### Code Example

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)

# ======================== DATA PREPARATION ========================

def create_plotting_dataset():
    """Create sample dataset for plotting demonstrations."""
    
    np.random.seed(42)
    
    # Generate date range
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    # Create sample data
    n_days = len(dates)
    
    # Sales data with trend and seasonality
    trend = np.linspace(1000, 2000, n_days)
    seasonality = 300 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)
    noise = np.random.normal(0, 100, n_days)
    sales = trend + seasonality + noise
    
    # Additional metrics
    profit = sales * np.random.uniform(0.1, 0.3, n_days)
    marketing_spend = np.random.exponential(200, n_days)
    
    # Categorical data
    regions = np.random.choice(['North', 'South', 'East', 'West'], n_days)
    products = np.random.choice(['Product_A', 'Product_B', 'Product_C', 'Product_D'], n_days)
    
    df = pd.DataFrame({
        'date': dates,
        'sales': sales,
        'profit': profit,
        'marketing_spend': marketing_spend,
        'region': regions,
        'product': products
    })
    
    # Ensure positive values
    df['sales'] = np.abs(df['sales'])
    df['profit'] = np.abs(df['profit'])
    df['marketing_spend'] = np.abs(df['marketing_spend'])
    
    return df

def demonstrate_basic_plots():
    """Demonstrate basic plotting capabilities."""
    
    print("=== BASIC PANDAS PLOTS ===")
    
    df = create_plotting_dataset()
    
    # Prepare monthly data for cleaner plots
    monthly_data = df.set_index('date').resample('M').agg({
        'sales': 'sum',
        'profit': 'sum',
        'marketing_spend': 'sum'
    })
    
    # 1. Line plots
    print(f"\n1. LINE PLOTS:")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Simple line plot
    monthly_data['sales'].plot(kind='line', ax=axes[0,0], title='Monthly Sales', color='blue')
    axes[0,0].set_ylabel('Sales ($)')
    axes[0,0].grid(True, alpha=0.3)
    
    # Multiple series line plot
    monthly_data[['sales', 'profit']].plot(kind='line', ax=axes[0,1], title='Sales vs Profit')
    axes[0,1].set_ylabel('Amount ($)')
    axes[0,1].grid(True, alpha=0.3)
    
    # 2. Bar plots
    region_sales = df.groupby('region')['sales'].sum()
    region_sales.plot(kind='bar', ax=axes[1,0], title='Sales by Region', color='skyblue')
    axes[1,0].set_ylabel('Total Sales ($)')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # 3. Scatter plot
    df.plot(kind='scatter', x='marketing_spend', y='sales', ax=axes[1,1], 
            title='Sales vs Marketing Spend', alpha=0.6, color='green')
    axes[1,1].set_xlabel('Marketing Spend ($)')
    axes[1,1].set_ylabel('Sales ($)')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return df
```

#### Explanation
1. **Built-in Integration**: Pandas `.plot()` method provides seamless integration with Matplotlib
2. **Multiple Chart Types**: Support for line, bar, histogram, scatter, box plots and more
3. **Quick Visualization**: Rapid prototyping of visualizations for exploratory data analysis

#### Use Cases
- **Exploratory Data Analysis**: Quick visualization during data investigation
- **Business Reporting**: Simple charts for presentations and reports
- **Data Quality Assessment**: Visual inspection of distributions and outliers

#### Best Practices
- **Choose Appropriate Plot Types**: Match visualization to data type and analysis goal
- **Add Meaningful Titles and Labels**: Always include descriptive titles and axis labels
- **Use Consistent Styling**: Apply consistent colors and formatting across plots

---

## Question 12

**What techniques can you use to improve theperformanceofPandasoperations?**

**Answer:**

#### Theory
Pandas provides integrated plotting capabilities through the `.plot()` method, which serves as a convenient wrapper around Matplotlib. This integration allows for quick data visualization directly from DataFrames and Series without needing to explicitly import matplotlib in simple cases. The plotting functionality supports various chart types including line plots, bar charts, histograms, scatter plots, and more, making it ideal for exploratory data analysis and rapid prototyping of visualizations.

#### Code Example

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)

# ======================== DATA PREPARATION ========================

def create_plotting_dataset():
    """Create sample dataset for plotting demonstrations."""
    
    np.random.seed(42)
    
    # Generate date range
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    # Create sample data
    n_days = len(dates)
    
    # Sales data with trend and seasonality
    trend = np.linspace(1000, 2000, n_days)
    seasonality = 300 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)
    noise = np.random.normal(0, 100, n_days)
    sales = trend + seasonality + noise
    
    # Additional metrics
    profit = sales * np.random.uniform(0.1, 0.3, n_days)
    marketing_spend = np.random.exponential(200, n_days)
    
    # Categorical data
    regions = np.random.choice(['North', 'South', 'East', 'West'], n_days)
    products = np.random.choice(['Product_A', 'Product_B', 'Product_C', 'Product_D'], n_days)
    
    df = pd.DataFrame({
        'date': dates,
        'sales': sales,
        'profit': profit,
        'marketing_spend': marketing_spend,
        'region': regions,
        'product': products
    })
    
    # Ensure positive values
    df['sales'] = np.abs(df['sales'])
    df['profit'] = np.abs(df['profit'])
    df['marketing_spend'] = np.abs(df['marketing_spend'])
    
    return df

def demonstrate_basic_plots():
    """Demonstrate basic plotting capabilities."""
    
    print("=== BASIC PANDAS PLOTS ===")
    
    df = create_plotting_dataset()
    
    # Prepare monthly data for cleaner plots
    monthly_data = df.set_index('date').resample('M').agg({
        'sales': 'sum',
        'profit': 'sum',
        'marketing_spend': 'sum'
    })
    
    # 1. Line plots
    print(f"\n1. LINE PLOTS:")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Simple line plot
    monthly_data['sales'].plot(kind='line', ax=axes[0,0], title='Monthly Sales', color='blue')
    axes[0,0].set_ylabel('Sales ($)')
    axes[0,0].grid(True, alpha=0.3)
    
    # Multiple series line plot
    monthly_data[['sales', 'profit']].plot(kind='line', ax=axes[0,1], title='Sales vs Profit')
    axes[0,1].set_ylabel('Amount ($)')
    axes[0,1].grid(True, alpha=0.3)
    
    # 2. Bar plots
    region_sales = df.groupby('region')['sales'].sum()
    region_sales.plot(kind='bar', ax=axes[1,0], title='Sales by Region', color='skyblue')
    axes[1,0].set_ylabel('Total Sales ($)')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # 3. Scatter plot
    df.plot(kind='scatter', x='marketing_spend', y='sales', ax=axes[1,1], 
            title='Sales vs Marketing Spend', alpha=0.6, color='green')
    axes[1,1].set_xlabel('Marketing Spend ($)')
    axes[1,1].set_ylabel('Sales ($)')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return df
```

#### Explanation
1. **Built-in Integration**: Pandas `.plot()` method provides seamless integration with Matplotlib
2. **Multiple Chart Types**: Support for line, bar, histogram, scatter, box plots and more
3. **Quick Visualization**: Rapid prototyping of visualizations for exploratory data analysis

#### Use Cases
- **Exploratory Data Analysis**: Quick visualization during data investigation
- **Business Reporting**: Simple charts for presentations and reports
- **Data Quality Assessment**: Visual inspection of distributions and outliers

#### Best Practices
- **Choose Appropriate Plot Types**: Match visualization to data type and analysis goal
- **Add Meaningful Titles and Labels**: Always include descriptive titles and axis labels
- **Use Consistent Styling**: Apply consistent colors and formatting across plots

---

## Question 13

**Compare and contrast thememory usageinPandasforcategories vs. objects.**

**Answer:**

#### Theory
Pandas provides integrated plotting capabilities through the `.plot()` method, which serves as a convenient wrapper around Matplotlib. This integration allows for quick data visualization directly from DataFrames and Series without needing to explicitly import matplotlib in simple cases. The plotting functionality supports various chart types including line plots, bar charts, histograms, scatter plots, and more, making it ideal for exploratory data analysis and rapid prototyping of visualizations.

#### Code Example

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)

# ======================== DATA PREPARATION ========================

def create_plotting_dataset():
    """Create sample dataset for plotting demonstrations."""
    
    np.random.seed(42)
    
    # Generate date range
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    # Create sample data
    n_days = len(dates)
    
    # Sales data with trend and seasonality
    trend = np.linspace(1000, 2000, n_days)
    seasonality = 300 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)
    noise = np.random.normal(0, 100, n_days)
    sales = trend + seasonality + noise
    
    # Additional metrics
    profit = sales * np.random.uniform(0.1, 0.3, n_days)
    marketing_spend = np.random.exponential(200, n_days)
    
    # Categorical data
    regions = np.random.choice(['North', 'South', 'East', 'West'], n_days)
    products = np.random.choice(['Product_A', 'Product_B', 'Product_C', 'Product_D'], n_days)
    
    df = pd.DataFrame({
        'date': dates,
        'sales': sales,
        'profit': profit,
        'marketing_spend': marketing_spend,
        'region': regions,
        'product': products
    })
    
    # Ensure positive values
    df['sales'] = np.abs(df['sales'])
    df['profit'] = np.abs(df['profit'])
    df['marketing_spend'] = np.abs(df['marketing_spend'])
    
    return df

def demonstrate_basic_plots():
    """Demonstrate basic plotting capabilities."""
    
    print("=== BASIC PANDAS PLOTS ===")
    
    df = create_plotting_dataset()
    
    # Prepare monthly data for cleaner plots
    monthly_data = df.set_index('date').resample('M').agg({
        'sales': 'sum',
        'profit': 'sum',
        'marketing_spend': 'sum'
    })
    
    # 1. Line plots
    print(f"\n1. LINE PLOTS:")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Simple line plot
    monthly_data['sales'].plot(kind='line', ax=axes[0,0], title='Monthly Sales', color='blue')
    axes[0,0].set_ylabel('Sales ($)')
    axes[0,0].grid(True, alpha=0.3)
    
    # Multiple series line plot
    monthly_data[['sales', 'profit']].plot(kind='line', ax=axes[0,1], title='Sales vs Profit')
    axes[0,1].set_ylabel('Amount ($)')
    axes[0,1].grid(True, alpha=0.3)
    
    # 2. Bar plots
    region_sales = df.groupby('region')['sales'].sum()
    region_sales.plot(kind='bar', ax=axes[1,0], title='Sales by Region', color='skyblue')
    axes[1,0].set_ylabel('Total Sales ($)')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # 3. Scatter plot
    df.plot(kind='scatter', x='marketing_spend', y='sales', ax=axes[1,1], 
            title='Sales vs Marketing Spend', alpha=0.6, color='green')
    axes[1,1].set_xlabel('Marketing Spend ($)')
    axes[1,1].set_ylabel('Sales ($)')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return df
```

#### Explanation
1. **Built-in Integration**: Pandas `.plot()` method provides seamless integration with Matplotlib
2. **Multiple Chart Types**: Support for line, bar, histogram, scatter, box plots and more
3. **Quick Visualization**: Rapid prototyping of visualizations for exploratory data analysis

#### Use Cases
- **Exploratory Data Analysis**: Quick visualization during data investigation
- **Business Reporting**: Simple charts for presentations and reports
- **Data Quality Assessment**: Visual inspection of distributions and outliers

#### Best Practices
- **Choose Appropriate Plot Types**: Match visualization to data type and analysis goal
- **Add Meaningful Titles and Labels**: Always include descriptive titles and axis labels
- **Use Consistent Styling**: Apply consistent colors and formatting across plots

---

## Question 14

**How do you managememory usagewhen working with largeDataFrames?**

**Answer:**

#### Theory
Pandas provides integrated plotting capabilities through the `.plot()` method, which serves as a convenient wrapper around Matplotlib. This integration allows for quick data visualization directly from DataFrames and Series without needing to explicitly import matplotlib in simple cases. The plotting functionality supports various chart types including line plots, bar charts, histograms, scatter plots, and more, making it ideal for exploratory data analysis and rapid prototyping of visualizations.

#### Code Example

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)

# ======================== DATA PREPARATION ========================

def create_plotting_dataset():
    """Create sample dataset for plotting demonstrations."""
    
    np.random.seed(42)
    
    # Generate date range
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    # Create sample data
    n_days = len(dates)
    
    # Sales data with trend and seasonality
    trend = np.linspace(1000, 2000, n_days)
    seasonality = 300 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)
    noise = np.random.normal(0, 100, n_days)
    sales = trend + seasonality + noise
    
    # Additional metrics
    profit = sales * np.random.uniform(0.1, 0.3, n_days)
    marketing_spend = np.random.exponential(200, n_days)
    
    # Categorical data
    regions = np.random.choice(['North', 'South', 'East', 'West'], n_days)
    products = np.random.choice(['Product_A', 'Product_B', 'Product_C', 'Product_D'], n_days)
    
    df = pd.DataFrame({
        'date': dates,
        'sales': sales,
        'profit': profit,
        'marketing_spend': marketing_spend,
        'region': regions,
        'product': products
    })
    
    # Ensure positive values
    df['sales'] = np.abs(df['sales'])
    df['profit'] = np.abs(df['profit'])
    df['marketing_spend'] = np.abs(df['marketing_spend'])
    
    return df

def demonstrate_basic_plots():
    """Demonstrate basic plotting capabilities."""
    
    print("=== BASIC PANDAS PLOTS ===")
    
    df = create_plotting_dataset()
    
    # Prepare monthly data for cleaner plots
    monthly_data = df.set_index('date').resample('M').agg({
        'sales': 'sum',
        'profit': 'sum',
        'marketing_spend': 'sum'
    })
    
    # 1. Line plots
    print(f"\n1. LINE PLOTS:")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Simple line plot
    monthly_data['sales'].plot(kind='line', ax=axes[0,0], title='Monthly Sales', color='blue')
    axes[0,0].set_ylabel('Sales ($)')
    axes[0,0].grid(True, alpha=0.3)
    
    # Multiple series line plot
    monthly_data[['sales', 'profit']].plot(kind='line', ax=axes[0,1], title='Sales vs Profit')
    axes[0,1].set_ylabel('Amount ($)')
    axes[0,1].grid(True, alpha=0.3)
    
    # 2. Bar plots
    region_sales = df.groupby('region')['sales'].sum()
    region_sales.plot(kind='bar', ax=axes[1,0], title='Sales by Region', color='skyblue')
    axes[1,0].set_ylabel('Total Sales ($)')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # 3. Scatter plot
    df.plot(kind='scatter', x='marketing_spend', y='sales', ax=axes[1,1], 
            title='Sales vs Marketing Spend', alpha=0.6, color='green')
    axes[1,1].set_xlabel('Marketing Spend ($)')
    axes[1,1].set_ylabel('Sales ($)')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return df
```

#### Explanation
1. **Built-in Integration**: Pandas `.plot()` method provides seamless integration with Matplotlib
2. **Multiple Chart Types**: Support for line, bar, histogram, scatter, box plots and more
3. **Quick Visualization**: Rapid prototyping of visualizations for exploratory data analysis

#### Use Cases
- **Exploratory Data Analysis**: Quick visualization during data investigation
- **Business Reporting**: Simple charts for presentations and reports
- **Data Quality Assessment**: Visual inspection of distributions and outliers

#### Best Practices
- **Choose Appropriate Plot Types**: Match visualization to data type and analysis goal
- **Add Meaningful Titles and Labels**: Always include descriptive titles and axis labels
- **Use Consistent Styling**: Apply consistent colors and formatting across plots

---

## Question 15

**How can you usechunkingto process largeCSV fileswithPandas?**

**Answer:**

#### Theory
Pandas provides integrated plotting capabilities through the `.plot()` method, which serves as a convenient wrapper around Matplotlib. This integration allows for quick data visualization directly from DataFrames and Series without needing to explicitly import matplotlib in simple cases. The plotting functionality supports various chart types including line plots, bar charts, histograms, scatter plots, and more, making it ideal for exploratory data analysis and rapid prototyping of visualizations.

#### Code Example

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)

# ======================== DATA PREPARATION ========================

def create_plotting_dataset():
    """Create sample dataset for plotting demonstrations."""
    
    np.random.seed(42)
    
    # Generate date range
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    # Create sample data
    n_days = len(dates)
    
    # Sales data with trend and seasonality
    trend = np.linspace(1000, 2000, n_days)
    seasonality = 300 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)
    noise = np.random.normal(0, 100, n_days)
    sales = trend + seasonality + noise
    
    # Additional metrics
    profit = sales * np.random.uniform(0.1, 0.3, n_days)
    marketing_spend = np.random.exponential(200, n_days)
    
    # Categorical data
    regions = np.random.choice(['North', 'South', 'East', 'West'], n_days)
    products = np.random.choice(['Product_A', 'Product_B', 'Product_C', 'Product_D'], n_days)
    
    df = pd.DataFrame({
        'date': dates,
        'sales': sales,
        'profit': profit,
        'marketing_spend': marketing_spend,
        'region': regions,
        'product': products
    })
    
    # Ensure positive values
    df['sales'] = np.abs(df['sales'])
    df['profit'] = np.abs(df['profit'])
    df['marketing_spend'] = np.abs(df['marketing_spend'])
    
    return df

def demonstrate_basic_plots():
    """Demonstrate basic plotting capabilities."""
    
    print("=== BASIC PANDAS PLOTS ===")
    
    df = create_plotting_dataset()
    
    # Prepare monthly data for cleaner plots
    monthly_data = df.set_index('date').resample('M').agg({
        'sales': 'sum',
        'profit': 'sum',
        'marketing_spend': 'sum'
    })
    
    # 1. Line plots
    print(f"\n1. LINE PLOTS:")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Simple line plot
    monthly_data['sales'].plot(kind='line', ax=axes[0,0], title='Monthly Sales', color='blue')
    axes[0,0].set_ylabel('Sales ($)')
    axes[0,0].grid(True, alpha=0.3)
    
    # Multiple series line plot
    monthly_data[['sales', 'profit']].plot(kind='line', ax=axes[0,1], title='Sales vs Profit')
    axes[0,1].set_ylabel('Amount ($)')
    axes[0,1].grid(True, alpha=0.3)
    
    # 2. Bar plots
    region_sales = df.groupby('region')['sales'].sum()
    region_sales.plot(kind='bar', ax=axes[1,0], title='Sales by Region', color='skyblue')
    axes[1,0].set_ylabel('Total Sales ($)')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # 3. Scatter plot
    df.plot(kind='scatter', x='marketing_spend', y='sales', ax=axes[1,1], 
            title='Sales vs Marketing Spend', alpha=0.6, color='green')
    axes[1,1].set_xlabel('Marketing Spend ($)')
    axes[1,1].set_ylabel('Sales ($)')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return df
```

#### Explanation
1. **Built-in Integration**: Pandas `.plot()` method provides seamless integration with Matplotlib
2. **Multiple Chart Types**: Support for line, bar, histogram, scatter, box plots and more
3. **Quick Visualization**: Rapid prototyping of visualizations for exploratory data analysis

#### Use Cases
- **Exploratory Data Analysis**: Quick visualization during data investigation
- **Business Reporting**: Simple charts for presentations and reports
- **Data Quality Assessment**: Visual inspection of distributions and outliers

#### Best Practices
- **Choose Appropriate Plot Types**: Match visualization to data type and analysis goal
- **Add Meaningful Titles and Labels**: Always include descriptive titles and axis labels
- **Use Consistent Styling**: Apply consistent colors and formatting across plots

---


