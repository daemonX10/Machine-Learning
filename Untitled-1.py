import geopandas as gpd
import numpy as np
import pandas as pd
import json
import os
import pickle
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, TypeVar, Callable, Optional, Tuple, Union
import functools
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt
import traceback
import hashlib

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("agriculture_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Type variable for generic function return
T = TypeVar('T')

class ExperimentTracker:
    """Track experiment results and save them for later analysis."""
    
    def __init__(self, base_dir="experiments"):
        """Initialize the experiment tracker.
        
        Args:
            base_dir: Base directory to save experiment results
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self.current_experiment = None
        self.results = []
        
    def start_experiment(self, name=None):
        """Start a new experiment."""
        if name is None:
            name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_experiment = {
            "name": name,
            "start_time": datetime.now(),
            "iterations": [],
            "best_score": 0,
            "best_features": None,
            "best_submission": None,
            "best_code": []
        }
        
        # Create experiment directory
        exp_dir = os.path.join(self.base_dir, name)
        os.makedirs(exp_dir, exist_ok=True)
        
        logger.info(f"Started experiment: {name}")
        return exp_dir
        
    def log_iteration(self, iteration_data):
        """Log an iteration of the experiment."""
        if self.current_experiment is None:
            raise ValueError("No active experiment. Call start_experiment first.")
            
        self.current_experiment["iterations"].append(iteration_data)
        
        # Update best score if improved
        if iteration_data.get("score", 0) > self.current_experiment["best_score"]:
            self.current_experiment["best_score"] = iteration_data.get("score", 0)
            self.current_experiment["best_features"] = iteration_data.get("features", None)
            self.current_experiment["best_submission"] = iteration_data.get("submission", None)
            if "code" in iteration_data:
                self.current_experiment["best_code"].append(iteration_data["code"])
        
    def end_experiment(self):
        """End the current experiment and save results."""
        if self.current_experiment is None:
            raise ValueError("No active experiment. Call start_experiment first.")
            
        self.current_experiment["end_time"] = datetime.now()
        self.current_experiment["duration"] = (
            self.current_experiment["end_time"] - self.current_experiment["start_time"]
        ).total_seconds()
        
        # Save experiment results
        exp_dir = os.path.join(self.base_dir, self.current_experiment["name"])
        
        # Save metadata
        with open(os.path.join(exp_dir, "metadata.json"), "w") as f:
            # Create a serializable copy without large objects
            metadata = {k: v for k, v in self.current_experiment.items() 
                       if k not in ["best_features", "best_submission"]}
            json.dump(metadata, f, default=str, indent=2)
        
        # Save best submission if available
        if self.current_experiment["best_submission"] is not None:
            self.current_experiment["best_submission"].to_csv(
                os.path.join(exp_dir, "best_submission.csv"), index=False
            )
        
        # Save best features if available (using pickle)
        if self.current_experiment["best_features"] is not None:
            with open(os.path.join(exp_dir, "best_features.pkl"), "wb") as f:
                pickle.dump(self.current_experiment["best_features"], f)
        
        # Save all generated code
        with open(os.path.join(exp_dir, "generated_code.txt"), "w") as f:
            for i, code in enumerate(self.current_experiment["best_code"]):
                f.write(f"# Function {i+1}\n")
                f.write(code)
                f.write("\n\n" + "="*80 + "\n\n")
        
        # Add to results list
        self.results.append(self.current_experiment)
        
        logger.info(f"Experiment {self.current_experiment['name']} completed with best score: {self.current_experiment['best_score']:.4f}")
        
        result = self.current_experiment
        self.current_experiment = None
        return result
    
    def get_best_experiment(self):
        """Get the experiment with the best score."""
        if not self.results:
            return None
        
        return max(self.results, key=lambda x: x["best_score"])


class GeminiRateLimiter:
    """
    A rate limiter for Google Gemini API that ensures requests don't exceed
    a specified number of requests per minute.
    """
    
    def __init__(self, requests_per_minute: int = 15, verbose: bool = True):
        """
        Initialize the rate limiter.
        
        Args:
            requests_per_minute: Maximum number of requests allowed per minute
            verbose: Whether to print status messages during rate limiting
        """
        self.requests_per_minute = requests_per_minute
        self.verbose = verbose
        self.request_times: List[float] = []  # Timestamps of recent requests
    
    def wait_if_needed(self):
        """
        Check if we need to wait before making another request and wait if necessary.
        This ensures we don't exceed the rate limit.
        """
        current_time = time.time()
        
        # Remove request times older than 60 seconds (sliding window)
        while self.request_times and current_time - self.request_times[0] > 60:
            self.request_times.pop(0)
        
        # If we're at the rate limit, wait until we can make another request
        if len(self.request_times) >= self.requests_per_minute:
            # Calculate when the oldest request will be 60+ seconds old
            wait_until = self.request_times[0] + 60  
            wait_time = wait_until - current_time
            
            if wait_time > 0:
                if self.verbose:
                    logger.warning(f"Rate limiting: Waiting {wait_time:.2f} seconds to stay within "
                                 f"{self.requests_per_minute} requests/minute")
                time.sleep(wait_time)
                
                # After waiting, remove old request times again
                current_time = time.time()
                while self.request_times and current_time - self.request_times[0] > 60:
                    self.request_times.pop(0)
    
    def record_request(self):
        """
        Record that a request has been made.
        """
        self.request_times.append(time.time())
        if self.verbose:
            logger.info(f"API request made ({len(self.request_times)}/{self.requests_per_minute} in current window)")
    
    def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute a function with rate limiting.
        
        Args:
            func: The function to execute (typically a Gemini API call)
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            The result of the function call
        """
        self.wait_if_needed()
        
        try:
            result = func(*args, **kwargs)
            self.record_request()
            return result
        except Exception as e:
            # Still record the request even if it failed
            # (API rate limits usually count failed requests too)
            self.record_request()
            raise e
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """
        Decorator for rate-limiting a function.
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.execute(func, *args, **kwargs)
        return wrapper


class DataLoader:
    """Load and preprocess geospatial data."""
    
    @staticmethod
    def load_data(train_path, test_path, verbose=True):
        """
        Load training and test data from geojson files.
        
        Args:
            train_path: Path to the training data
            test_path: Path to the test data
            verbose: Whether to print information about the loaded data
        
        Returns:
            Dictionary containing train and test data
        """
        try:
            # Load training data
            train = gpd.read_file(train_path)
            if verbose:
                logger.info(f"Training samples: {len(train)}")
                logger.info("Crop types distribution:")
                logger.info(train['crop'].value_counts())
            
            # Load test data
            test = gpd.read_file(test_path)
            if verbose:
                logger.info(f"Test samples: {len(test)}")
            
            return {
                "train_gdf": train,
                "test_gdf": test,
                "train_ids": train['ID'] if 'ID' in train.columns else None,
                "test_ids": test['ID'] if 'ID' in test.columns else None,
                "target": train['crop'] if 'crop' in train.columns else None
            }
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise


class FeatureEngineering:
    """Feature engineering for geospatial data."""
    
    @staticmethod
    def extract_geometry_features(gdf):
        """
        Extract features from polygon geometries.
        
        Args:
            gdf: GeoDataFrame with geometry column
            
        Returns:
            DataFrame with extracted features
        """
        features = []

        for geom in gdf.geometry:
            # Handle both Polygon and MultiPolygon
            if geom.geom_type == 'MultiPolygon':
                areas = [p.area for p in geom.geoms]
                perims = [p.length for p in geom.geoms]
                poly = geom.geoms[0]  # Take largest polygon for other features
            else:
                areas = [geom.area]
                perims = [geom.length]
                poly = geom

            # Basic shape features
            area = sum(areas)
            perimeter = sum(perims)
            compactness = (4 * np.pi * area) / (perimeter ** 2)

            # Convex hull features
            convex_hull = poly.convex_hull
            hull_area = convex_hull.area
            solidity = area / hull_area if hull_area > 0 else 0

            # Bounding box features
            minx, miny, maxx, maxy = poly.bounds
            width = maxx - minx
            height = maxy - miny
            aspect_ratio = width / height if height > 0 else 0

            # Moment invariants
            coords = np.array(poly.exterior.coords)
            dx = coords[:,0] - minx
            dy = coords[:,1] - miny
            m00 = area
            m10 = np.sum(dx)
            m01 = np.sum(dy)
            centroid_x = m10 / m00 if m00 > 0 else 0
            centroid_y = m01 / m00 if m00 > 0 else 0
            mu20 = np.sum((dx - centroid_x)**2) / m00 if m00 > 0 else 0
            mu02 = np.sum((dy - centroid_y)**2) / m00 if m00 > 0 else 0
            mu11 = np.sum((dx - centroid_x)*(dy - centroid_y)) / m00 if m00 > 0 else 0

            features.append({
                'area': area,
                'perimeter': perimeter,
                'compactness': compactness,
                'solidity': solidity,
                'aspect_ratio': aspect_ratio,
                'num_polygons': len(areas),
                'largest_area': max(areas),
                'smallest_area': min(areas),
                'mean_area': np.mean(areas),
                'area_std': np.std(areas),
                'mu20': mu20,
                'mu02': mu02,
                'mu11': mu11,
                'width': width,
                'height': height
            })

        return pd.DataFrame(features)
    
    @staticmethod
    def prepare_data(data_dict):
        """
        Prepare data for training by extracting features.
        
        Args:
            data_dict: Dictionary containing train and test GeoDataFrames
            
        Returns:
            Dictionary with prepared training and test features
        """
        try:
            # Extract features
            train_features = FeatureEngineering.extract_geometry_features(data_dict["train_gdf"])
            test_features = FeatureEngineering.extract_geometry_features(data_dict["test_gdf"])
            
            # Add target if available
            if data_dict.get("target") is not None:
                train_features['crop'] = data_dict["target"].values
                
            return {
                "X": train_features.drop('crop', axis=1) if 'crop' in train_features.columns else train_features,
                "y": train_features['crop'] if 'crop' in train_features.columns else None,
                "test_features": test_features,
                "train_ids": data_dict.get("train_ids"),
                "test_ids": data_dict.get("test_ids")
            }
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise
    
    @staticmethod
    def validate_features(features_df):
        """
        Validate that features are valid and numerical.
        
        Args:
            features_df: DataFrame with features to validate
            
        Returns:
            Cleaned feature DataFrame
        """
        # Check for non-numeric features
        non_numeric_cols = features_df.select_dtypes(exclude=['number']).columns
        if len(non_numeric_cols) > 0:
            logger.warning(f"Dropping non-numeric columns: {list(non_numeric_cols)}")
            features_df = features_df.drop(columns=non_numeric_cols)
        
        # Check for NaN values
        if features_df.isna().any().any():
            logger.warning(f"Filling NaN values with 0")
            features_df = features_df.fillna(0)
        
        # Check for infinite values
        if np.isinf(features_df.values).any():
            logger.warning(f"Replacing infinite values with 0")
            features_df = features_df.replace([np.inf, -np.inf], 0)
        
        return features_df


class ModelTrainer:
    """Train and evaluate machine learning models."""
    
    @staticmethod
    def train_and_predict(X, y, test_features, test_ids, model, cv=5, random_state=42):
        """
        Trains a model on the provided data and makes predictions.

        Args:
            X: Training features
            y: Target variable
            test_features: Test features
            test_ids: Test IDs
            model: The machine learning model to use
            cv: Number of cross-validation folds
            random_state: Random state for reproducibility

        Returns:
            A tuple containing the submission DataFrame and the mean F1 score
        """
        try:
            # Initialize StratifiedKFold
            skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

            f1_scores = []
            fold_probabilities = []
            
            # Initialize LabelEncoder
            label_encoder = LabelEncoder()

            # Fit and transform the target variable
            y_encoded = label_encoder.fit_transform(y)

            # Cross-validation
            for fold, (train_index, val_index) in enumerate(skf.split(X, y_encoded)):
                logger.info(f"Training fold {fold+1}/{cv}")
                X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
                y_train_fold, y_val_fold = y_encoded[train_index], y_encoded[val_index]

                # Clone model for each fold
                fold_model = clone_model(model)
                
                # Train model
                fold_model.fit(X_train_fold, y_train_fold)
                
                # Validate
                y_pred = fold_model.predict(X_val_fold).astype(int)
                f1 = f1_score(y_val_fold, y_pred, average='weighted')
                f1_scores.append(f1)
                logger.info(f"Fold {fold+1} F1 Score: {f1:.4f}")

                # Generate test predictions
                fold_probs = fold_model.predict_proba(test_features)
                fold_probabilities.append(fold_probs)

            # Calculate mean F1 score
            mean_f1 = np.mean(f1_scores)
            logger.info(f"Mean F1 Score across {cv} folds: {mean_f1:.4f}")

            # Average predictions across folds
            average_probabilities = np.mean(fold_probabilities, axis=0)
            predicted_labels = np.argmax(average_probabilities, axis=1).astype(int)
            predicted_crop_types = label_encoder.inverse_transform(predicted_labels)
            
            # Create submission DataFrame
            submission = pd.DataFrame({'ID': test_ids, 'Target': predicted_crop_types})

            return submission, mean_f1
        
        except Exception as e:
            logger.error(f"Error in train_and_predict: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    @staticmethod
    def clone_model(model):
        """
        Create a deep clone of a model.
        
        Args:
            model: The model to clone
            
        Returns:
            A new instance of the model with the same parameters
        """
        if hasattr(model, 'get_params'):
            params = model.get_params()
            return model.__class__(**params)
        else:
            # Fallback for models without get_params
            return model.__class__()
    
    @staticmethod
    def hyperparameter_tuning(X, y, model, param_grid, cv=3):
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            X: Training features
            y: Target variable
            model: Model to tune
            param_grid: Dictionary of parameters to search
            cv: Number of cross-validation folds
            
        Returns:
            Best model with optimized parameters
        """
        logger.info("Starting hyperparameter tuning")
        
        # Initialize GridSearchCV
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1, verbose=1
        )
        
        # Fit GridSearchCV
        grid_search.fit(X, y)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_


class FeatureGenerator:
    """Generate new features using AI API."""
    
    def __init__(self, api_key, rate_limiter=None, model="gemini-2.0-flash"):
        """
        Initialize the feature generator.
        
        Args:
            api_key: API key for Gemini API
            rate_limiter: Rate limiter for API calls
            model: Gemini model to use
        """
        try:
            import google.generativeai as genai
            from google.generativeai import types
            
            self.genai = genai
            self.types = types
            self.api_key = api_key
            self.model = model
            self.rate_limiter = rate_limiter or GeminiRateLimiter(requests_per_minute=10)
            
            # Initialize API client
            self.client = genai.Client(api_key=api_key)
            
            # Base code example for feature generation
            self.base_code = """
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon
def extract_geometry_features(gdf):

    features = []
    
    for geom in gdf.geometry:
        # Handle both Polygon and MultiPolygon
        if geom.geom_type == 'MultiPolygon':
            areas = [p.area for p in geom.geoms]
            perims = [p.length for p in geom.geoms]
            poly = geom.geoms[0]  # Take largest polygon for other features
        else:
            areas = [geom.area]
            perims = [geom.length]
            poly = geom
    
        # Basic shape features
        area = sum(areas)
        perimeter = sum(perims)
        compactness = (4 * np.pi * area) / (perimeter ** 2)
    
        # Convex hull features
        convex_hull = poly.convex_hull
        hull_area = convex_hull.area
        solidity = area / hull_area if hull_area > 0 else 0
    
        # Bounding box features
        minx, miny, maxx, maxy = poly.bounds
        width = maxx - minx
        height = maxy - miny
        aspect_ratio = width / height if height > 0 else 0
    
        # Moment invariants
        coords = np.array(poly.exterior.coords)
        dx = coords[:,0] - minx
        dy = coords[:,1] - miny
        m00 = area
        m10 = np.sum(dx)
        m01 = np.sum(dy)
        centroid_x = m10 / m00
        centroid_y = m01 / m00
        mu20 = np.sum((dx - centroid_x)**2) / m00
        mu02 = np.sum((dy - centroid_y)**2) / m00
        mu11 = np.sum((dx - centroid_x)*(dy - centroid_y)) / m00
    
        features.append({
            'area': area,
            'perimeter': perimeter,
            'compactness': compactness,
            'solidity': solidity,
            'aspect_ratio': aspect_ratio,
            'num_polygons': len(areas),
            'largest_area': max(areas),
            'smallest_area': min(areas),
            'mean_area': np.mean(areas),
            'area_std': np.std(areas),
            'mu20': mu20,
            'mu02': mu02,
            'mu11': mu11,
            'width': width,
            'height': height
        })
    
    return pd.DataFrame(features)
"""
            
            # Cache to prevent regenerating the same code
            self.function_cache = {}
            
        except ImportError:
            logger.error("Google Generative AI package not found. Please install it with 'pip install google-genai'")
            raise
    
    def function_hash(self, examples):
        """
        Create a hash of the examples to use as a cache key.
        
        Args:
            examples: List of example code strings
            
        Returns:
            Hash string
        """
        return hashlib.md5(('\n\n'.join(examples)).encode()).hexdigest()
    
    def generate_feature_function(self, examples):
        """
        Generate a new feature engineering function.
        
        Args:
            examples: List of example code strings
            
        Returns:
            Generated function string
        """
        # Check cache first
        cache_key = self.function_hash(examples)
        if cache_key in self.function_cache:
            logger.info("Using cached function")
            return self.function_cache[cache_key]
        
        # Call API with rate limiting
        @self.rate_limiter
        def api_call(examples_text):
            format_string = {'code': 'your generated code'}
            contents = [
                self.types.Content(
                    role="user",
                    parts=[
                        self.types.Part.from_text(text=f"""Generate a Python function for feature engineering on a geopandas GeoDataFrame. 
- Use geopandas, pandas, and shapely. 
- Please refer to example function below. 
- you must generate new features not existing in the example code.
- Handle potential errors gracefully. 
- Your generated python function must be named as add_new_features.
- The function must return a pandas dataframe with new features only. they must be numerical types.
- Please output the code directly in json format {format_string}

-- Start Example --

{examples_text}

-- End Example --"""),
                    ],
                ),
            ]
            generate_content_config = self.types.GenerateContentConfig(
                response_mime_type="application/json",
            )

            res = []
            for chunk in self.client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=generate_content_config,
            ):
                res.append(chunk.text)
            return ''.join(res)
        
        try:
            # Call API
            response = api_call('\n\n'.join(examples))
            
            # Parse response
            function_string = json.loads(response.strip())['code']
            
            # Cache result
            self.function_cache[cache_key] = function_string
            
            return function_string
        
        except Exception as e:
            logger.error(f"Error generating feature function: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def execute_feature_function(self, function_string, train_gdf, test_gdf):
        """
        Execute a generated feature function safely.
        
        Args:
            function_string: Python function as a string
            train_gdf: Training GeoDataFrame
            test_gdf: Test GeoDataFrame
            
        Returns:
            Tuple of (new_train_features, new_test_features) or (None, None) if execution fails
        """
        try:
            # Create namespace for execution
            namespace = {}
            
            # Execute function in namespace
            exec(function_string, namespace)
            
            # Get function from namespace
            add_new_features = namespace['add_new_features']
            
            # Execute function
            new_train_features = add_new_features(train_gdf)
            new_test_features = add_new_features(test_gdf)
            
            # Validate features
            new_train_features = FeatureEngineering.validate_features(new_train_features)
            new_test_features = FeatureEngineering.validate_features(new_test_features)
            
            return new_train_features, new_test_features
            
        except Exception as e:
            logger.error(f"Error executing feature function: {str(e)}")
            logger.error(traceback.format_exc())
            return None, None


class Pipeline:
    """Main pipeline for agricultural feature engineering and prediction."""
    
    def __init__(self, config=None):
        """
        Initialize the pipeline with a configuration.
        
        Args:
            config: Dictionary with pipeline configuration
        """
        # Default configuration
        default_config = {
            "data_paths": {
                "train_path": "/kaggle/input/cote-divoire-byte-sized-agriculture-challenge/train.geojson",
                "test_path": "/kaggle/input/cote-divoire-byte-sized-agriculture-challenge/test.geojson"
            },
            "api": {
                "api_key_name": "GEMINI_API_KEY",
                "model": "gemini-2.0-flash",
                "rate_limit": 10
            },
            "feature_generation": {
                "max_iterations": 15,
                "improvement_margin": 0.01,
                "max_features": 100
            },
            "model": {
                "type": "RandomForestClassifier",
                "params": {
                    "n_estimators": 100,
                    "random_state": 42
                },
                "cv_folds": 5
            },
            "output": {
                "save_submission": True,
                "submission_path": "submission.csv",
                "save_features": True,
                "features_path": "best_features.pkl"
            },
            "misc": {
                "verbose": True,
                "random_state": 42
            }
        }
        
        # Update default config with provided config
        self.config = default_config.copy()
        if config is not None:
            self._update_dict(self.config, config)
        
        # Initialize components
        self.experiment_tracker = ExperimentTracker()
        
        # Get API key
        try:
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            self.api_key = user_secrets.get_secret(self.config["api"]["api_key_name"])
        except ImportError:
            logger.warning("Kaggle secrets not available. Using empty API key.")
            self.api_key = os.environ.get(self.config["api"]["api_key_name"], "")
        
        # Initialize rate limiter
        self.rate_limiter = GeminiRateLimiter(
            requests_per_minute=self.config["api"]["rate_limit"],
            verbose=self.config["misc"]["verbose"]
        )
        
        # Initialize feature generator
        self.feature_generator = FeatureGenerator(
            api_key=self.api_key,
            rate_limiter=self.rate_limiter,
            model=self.config["api"]["model"]
        )
        
        # Initialize model
        self.model = self._create_model()
    
    def _update_dict(self, d, u):
        """Recursively update a dictionary."""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_dict(d[k], v)
            else:
                d[k] = v
    
    def _create_model(self):
        """Create a model based on configuration."""
        model_type = self.config["model"]["type"]
        params = self.config["model"]["params"]
        
        if model_type == "RandomForestClassifier":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(**params)
        elif model_type == "GradientBoostingClassifier":
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(**params)
        elif model_type == "XGBClassifier":
            try:
                from xgboost import XGBClassifier
                return XGBClassifier(**params)
            except ImportError:
                logger.warning("XGBoost not available. Falling back to RandomForestClassifier.")
                return RandomForestClassifier(**params)
        else:
            logger.warning(f"Unknown model type: {model_type}. Using RandomForestClassifier.")
            return RandomForestClassifier(**params)
    
    def run(self, experiment_name=None, load_data=True):
        """
        Run the complete pipeline.
        
        Args:
            experiment_name: Name for this experiment run
            load_data: Whether to load data (set to False if data is already loaded)
            
        Returns:
            Dictionary with experiment results
        """
        # Start experiment
        exp_dir = self.experiment_tracker.start_experiment(experiment_name)
        
        try:
            # Step 1: Load data
            if load_data:
                logger.info("Loading data...")
                data_dict = DataLoader.load_data(
                    self.config["data_paths"]["train_path"],
                    self.config["data_paths"]["test_path"],
                    verbose=self.config["misc"]["verbose"]
                )
                
                # Prepare initial features
                logger.info("Preparing initial features...")
                prepared_data = FeatureEngineering.prepare_data(data_dict)
                
                self.data_dict = data_dict
                self.prepared_data = prepared_data
            
            # Step 2: Run feature generation loop
            logger.info("Starting feature generation...")
            best_features, best_score, best_submission, best_code = self.feature_selection_loop(
                self.prepared_data["X"],
                self.prepared_data["y"],
                self.prepared_data["test_features"],
                self.prepared_data["test_ids"],
                self.model,
                max_iter=self.config["feature_generation"]["max_iterations"],
                margin=self.config["feature_generation"]["improvement_margin"]
            )
            
            # Step 3: Save results
            if self.config["output"]["save_submission"]:
                submission_path = os.path.join(exp_dir, self.config["output"]["submission_path"])
                best_submission.to_csv(submission_path, index=False)
                logger.info(f"Submission saved to {submission_path}")
            
            if self.config["output"]["save_features"]:
                features_path = os.path.join(exp_dir, self.config["output"]["features_path"])
                with open(features_path, "wb") as f:
                    pickle.dump(best_features, f)
                logger.info(f"Best features saved to {features_path}")
            
            # Step 4: End experiment
            result = self.experiment_tracker.end_experiment()
            
            return result
            
        except Exception as e:
            logger.error(f"Error in pipeline run: {str(e)}")
            logger.error(traceback.format_exc())
            self.experiment_tracker.end_experiment()
            raise
    
    def feature_selection_loop(self, X, y, test_features, test_ids, model, max_iter=15, margin=0.01):
        """
        Feature selection loop that generates and tests new features.
        
        Args:
            X: Initial training features
            y: Target variable
            test_features: Initial test features
            test_ids: Test IDs
            model: Model to use for evaluation
            max_iter: Maximum number of iterations
            margin: Minimum improvement margin to keep new features
            
        Returns:
            Tuple of (best_features, best_score, best_submission, best_code)
        """
        best_features = X.copy()
        test_features_copy = test_features.copy()
        examples = [self.feature_generator.base_code]
        
        logger.info("Running baseline model...")
        best_submission, best_score = ModelTrainer.train_and_predict(
            X, y, test_features_copy, test_ids, 
            ModelTrainer.clone_model(model),
            cv=self.config["model"]["cv_folds"],
            random_state=self.config["misc"]["random_state"]
        )
        logger.info(f"Baseline score: {best_score:.4f}")
        
        # Log baseline iteration
        self.experiment_tracker.log_iteration({
            "iteration": 0,
            "score": best_score,
            "features": best_features,
            "submission": best_submission,
            "num_features": best_features.shape[1]
        })
        
        for i in range(max_iter):
            logger.info(f"Feature generation iteration {i+1}/{max_iter}")
            
            try:
                # Generate new feature function
                function_string = self.feature_generator.generate_feature_function(examples)
                logger.info(f"Generated new function (length: {len(function_string)} characters)")
                
                # Execute function to get new features
                new_train_features, new_test_features = self.feature_generator.execute_feature_function(
                    function_string, self.data_dict["train_gdf"], self.data_dict["test_gdf"]
                )
                
                if new_train_features is None or new_test_features is None:
                    logger.warning("Feature generation failed, skipping iteration")
                    continue
                
                # Check feature counts
                logger.info(f"Generated {new_train_features.shape[1]} new features")
                
                # Limit feature count to prevent memory issues
                if new_train_features.shape[1] > self.config["feature_generation"]["max_features"]:
                    logger.warning(f"Too many features ({new_train_features.shape[1]}), truncating to {self.config['feature_generation']['max_features']}")
                    new_train_features = new_train_features.iloc[:, :self.config["feature_generation"]["max_features"]]
                    new_test_features = new_test_features.iloc[:, :self.config["feature_generation"]["max_features"]]
                
                # Combine with existing features
                combined_train_features = pd.concat([best_features, new_train_features], axis=1)
                combined_test_features = pd.concat([test_features_copy, new_test_features], axis=1)
                
                # Train and predict
                submission, score = ModelTrainer.train_and_predict(
                    combined_train_features, y, combined_test_features, test_ids, 
                    ModelTrainer.clone_model(model),
                    cv=self.config["model"]["cv_folds"],
                    random_state=self.config["misc"]["random_state"]
                )
                
                # Log iteration
                self.experiment_tracker.log_iteration({
                    "iteration": i+1,
                    "score": score,
                    "features": combined_train_features,
                    "submission": submission,
                    "code": function_string,
                    "num_features": combined_train_features.shape[1]
                })
                
                # Check if the score improved meaningfully
                if score > best_score + margin:
                    logger.info(f"Score improved from {best_score:.4f} to {score:.4f}. Adding new features.")
                    best_score = score
                    best_features = combined_train_features
                    test_features_copy = combined_test_features
                    examples.append(function_string)
                    best_submission = submission
                else:
                    logger.info(f"Score {score:.4f} did not improve significantly over {best_score:.4f}. Keeping previous features.")
            
            except Exception as e:
                logger.error(f"Error in iteration {i+1}: {str(e)}")
                logger.error(traceback.format_exc())
                continue
        
        return best_features, best_score, best_submission, examples


def clone_model(model):
    """
    Create a deep clone of a model.
    """
    if hasattr(model, 'get_params'):
        params = model.get_params()
        return model.__class__(**params)
    else:
        # Fallback for models without get_params
        return model.__class__()


def main():
    """Main entry point for command line execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Agricultural Feature Engineering Pipeline")
    parser.add_argument('--config', type=str, help='Path to JSON configuration file')
    parser.add_argument('--iterations', type=int, default=15, help='Number of feature generation iterations')
    parser.add_argument('--margin', type=float, default=0.01, help='Improvement margin for feature selection')
    parser.add_argument('--output', type=str, default='submission.csv', help='Path for output submission file')
    args = parser.parse_args()
    
    # Load configuration if provided
    config = None
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Error loading configuration file: {str(e)}")
            return
    
    # Update config with command line arguments
    if config is None:
        config = {}
    
    if args.iterations:
        if "feature_generation" not in config:
            config["feature_generation"] = {}
        config["feature_generation"]["max_iterations"] = args.iterations
    
    if args.margin:
        if "feature_generation" not in config:
            config["feature_generation"] = {}
        config["feature_generation"]["improvement_margin"] = args.margin
    
    if args.output:
        if "output" not in config:
            config["output"] = {}
        config["output"]["submission_path"] = args.output
    
    # Initialize and run pipeline
    pipeline = Pipeline(config)
    result = pipeline.run()
    
    logger.info(f"Pipeline completed with best score: {result['best_score']:.4f}")


if __name__ == "__main__":
    main()
    
    
from agriculture_pipeline import Pipeline

# Configure the pipeline
config = {
    "data_paths": {
        "train_path": "/kaggle/input/cote-divoire-byte-sized-agriculture-challenge/train.geojson",
        "test_path": "/kaggle/input/cote-divoire-byte-sized-agriculture-challenge/test.geojson"
    },
    "feature_generation": {
        "max_iterations": 10,  # Run 100 iterations
        "improvement_margin": 0.5  # More sensitive improvement detection
    },
    "model": {
        "type": "RandomForestClassifier",
        "params": {
            "n_estimators": 200,
            "random_state": 42
        }
    }
}

# Run the pipeline
pipeline = Pipeline(config)
result = pipeline.run("experiment_100_iterations")
print(f"Best F1 score: {result['best_score']}")