# Time Series Interview Questions - Coding Questions

## Question 1

**Implement aPython functionto performsimple exponential smoothingon atime series.**

**Answer:**

**Theoretical Foundation:**

Simple Exponential Smoothing (SES) is a fundamental time series forecasting technique that applies **exponentially decreasing weights** to historical observations. The mathematical formulation involves:

**Exponential Smoothing Equation:**
```
S_t = α × X_t + (1-α) × S_{t-1}

Where:
- S_t: Smoothed value at time t
- X_t: Observed value at time t  
- α: Smoothing parameter (0 < α ≤ 1)
- S_{t-1}: Previous smoothed value
```

**Forecast Equation:**
```
F_{t+h} = S_t  (for all h > 0)

Where:
- F_{t+h}: Forecast for h periods ahead
- h: Forecast horizon
```

**Comprehensive Implementation:**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class SimpleExponentialSmoothing:
    """
    Comprehensive Simple Exponential Smoothing implementation with 
    theoretical foundations and advanced features
    """
    
    def __init__(self, alpha=None, optimize=True):
        """
        Initialize Simple Exponential Smoothing
        
        Parameters:
        -----------
        alpha : float, optional
            Smoothing parameter (0 < alpha <= 1)
            If None, will be optimized
        optimize : bool, default=True
            Whether to optimize alpha parameter
        """
        self.alpha = alpha
        self.optimize = optimize
        self.fitted_values = None
        self.residuals = None
        self.forecasts = None
        self.smoothed_values = None
        self.initial_value = None
        self.optimized_alpha = None
        
    def _theoretical_foundations(self):
        """Mathematical foundations of exponential smoothing"""
        
        foundations = {
            'exponential_weights': {
                'formula': 'w_k = α(1-α)^k for k = 0,1,2,...',
                'interpretation': 'Weights decrease exponentially with age',
                'normalization': 'Σw_k = 1 (infinite sum)',
                'alpha_effect': 'Higher α → more weight on recent observations'
            },
            'memory_properties': {
                'infinite_memory': 'All past observations contribute',
                'exponential_decay': 'Older observations have exponentially less influence',
                'effective_memory': '95% weight within ≈ 3/α observations',
                'stability': 'Method is inherently stable for 0 < α ≤ 1'
            },
            'mathematical_properties': {
                'linearity': 'Linear combination of observations',
                'unbiasedness': 'Unbiased for constant series',
                'variance': 'Var(S_t) = α²σ²/(2-α) in steady state',
                'lag_operator': 'S_t = α/(1-(1-α)L) × X_t'
            },
            'optimization_theory': {
                'objective': 'Minimize sum of squared errors',
                'mse_formula': 'MSE(α) = (1/n)Σ(X_t - S_{t-1})²',
                'bias_variance_tradeoff': 'α controls bias-variance balance',
                'optimal_alpha': 'Depends on signal-to-noise ratio'
            }
        }
        
        return foundations
    
    def _initialize_smoothing(self, data, method='first'):
        """
        Initialize smoothing with different methods
        
        Parameters:
        -----------
        data : array-like
            Time series data
        method : str, default='first'
            Initialization method: 'first', 'mean', 'ols'
        """
        
        if method == 'first':
            # Use first observation
            initial = data[0]
            
        elif method == 'mean':
            # Use mean of first few observations
            n_init = min(10, len(data) // 4)
            initial = np.mean(data[:n_init])
            
        elif method == 'ols':
            # Ordinary least squares initialization
            # Minimize SSE over α and initial value simultaneously
            def objective(params):
                alpha_init, s0 = params
                if alpha_init <= 0 or alpha_init > 1:
                    return np.inf
                
                smoothed = self._smooth_series(data, alpha_init, s0)
                errors = data[1:] - smoothed[:-1]  # One-step-ahead errors
                return np.sum(errors**2)
            
            # Optimize both alpha and initial value
            from scipy.optimize import minimize
            result = minimize(objective, [0.3, data[0]], 
                            bounds=[(0.01, 1.0), (None, None)])
            
            if result.success:
                initial = result.x[1]
                if self.alpha is None:
                    self.optimized_alpha = result.x[0]
            else:
                initial = data[0]
                
        else:
            raise ValueError("Invalid initialization method")
            
        return initial
    
    def _smooth_series(self, data, alpha, initial_value):
        """Core smoothing algorithm"""
        
        n = len(data)
        smoothed = np.zeros(n)
        smoothed[0] = initial_value
        
        # Apply exponential smoothing recursively
        for t in range(1, n):
            smoothed[t] = alpha * data[t-1] + (1 - alpha) * smoothed[t-1]
            
        return smoothed
    
    def _optimize_alpha(self, data, initial_value):
        """Optimize smoothing parameter using MSE"""
        
        def mse_objective(alpha):
            """Objective function for alpha optimization"""
            if alpha <= 0 or alpha > 1:
                return np.inf
                
            smoothed = self._smooth_series(data, alpha, initial_value)
            
            # One-step-ahead forecast errors
            errors = data[1:] - smoothed[:-1]
            mse = np.mean(errors**2)
            
            return mse
        
        # Optimize alpha
        result = minimize_scalar(mse_objective, bounds=(0.01, 1.0), method='bounded')
        
        if result.success:
            optimal_alpha = result.x
            optimal_mse = result.fun
        else:
            # Fallback to default value
            optimal_alpha = 0.3
            optimal_mse = mse_objective(optimal_alpha)
            
        return optimal_alpha, optimal_mse
    
    def fit(self, data, initialization='first'):
        """
        Fit Simple Exponential Smoothing model
        
        Parameters:
        -----------
        data : array-like
            Time series data
        initialization : str, default='first'
            Initialization method for smoothing
        """
        
        # Convert to numpy array
        data = np.asarray(data)
        
        if len(data) < 2:
            raise ValueError("Data must contain at least 2 observations")
        
        # Initialize smoothing
        self.initial_value = self._initialize_smoothing(data, initialization)
        
        # Optimize alpha if needed
        if self.alpha is None or self.optimize:
            self.optimized_alpha, optimal_mse = self._optimize_alpha(data, self.initial_value)
            working_alpha = self.optimized_alpha
        else:
            working_alpha = self.alpha
            
        # Compute smoothed values
        self.smoothed_values = self._smooth_series(data, working_alpha, self.initial_value)
        
        # Compute fitted values (one-step-ahead forecasts)
        self.fitted_values = np.zeros_like(data)
        self.fitted_values[0] = self.initial_value
        self.fitted_values[1:] = self.smoothed_values[:-1]
        
        # Calculate residuals
        self.residuals = data - self.fitted_values
        
        # Store final alpha
        self.final_alpha = working_alpha
        
        return self
    
    def forecast(self, steps=1):
        """
        Generate forecasts
        
        Parameters:
        -----------
        steps : int, default=1
            Number of periods to forecast
        """
        
        if self.smoothed_values is None:
            raise ValueError("Model must be fitted before forecasting")
            
        # For SES, all future forecasts equal the last smoothed value
        last_smoothed = self.smoothed_values[-1]
        forecasts = np.full(steps, last_smoothed)
        
        return forecasts
    
    def get_model_statistics(self, data):
        """Calculate comprehensive model statistics"""
        
        if self.fitted_values is None:
            raise ValueError("Model must be fitted first")
            
        data = np.asarray(data)
        n = len(data)
        
        # Forecast errors
        errors = self.residuals[1:]  # Skip first observation
        
        # Error metrics
        mse = np.mean(errors**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(errors))
        mape = np.mean(np.abs(errors / data[1:])) * 100
        
        # Information criteria (approximations for SES)
        log_likelihood = -0.5 * n * np.log(2 * np.pi * mse) - 0.5 * np.sum(errors**2) / mse
        aic = -2 * log_likelihood + 2 * 2  # 2 parameters: alpha, initial
        bic = -2 * log_likelihood + np.log(n) * 2
        
        # Residual diagnostics
        residual_mean = np.mean(errors)
        residual_std = np.std(errors)
        
        # Ljung-Box test statistic (simplified)
        def ljung_box_statistic(residuals, lags=10):
            """Calculate Ljung-Box test statistic"""
            n = len(residuals)
            lags = min(lags, n//4)
            
            # Calculate autocorrelations
            autocorrs = []
            for lag in range(1, lags + 1):
                if n - lag > 0:
                    autocorr = np.corrcoef(residuals[:-lag], residuals[lag:])[0, 1]
                    if not np.isnan(autocorr):
                        autocorrs.append(autocorr)
                    else:
                        autocorrs.append(0)
                else:
                    autocorrs.append(0)
            
            # Ljung-Box statistic
            lb_stat = n * (n + 2) * sum((autocorr**2) / (n - lag) 
                                       for lag, autocorr in enumerate(autocorrs, 1))
            
            return lb_stat, autocorrs
        
        lb_stat, autocorrs = ljung_box_statistic(errors)
        
        statistics = {
            'model_parameters': {
                'alpha': self.final_alpha,
                'initial_value': self.initial_value,
                'optimized': self.alpha is None
            },
            'error_metrics': {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'mape': mape
            },
            'information_criteria': {
                'aic': aic,
                'bic': bic,
                'log_likelihood': log_likelihood
            },
            'residual_diagnostics': {
                'mean': residual_mean,
                'std': residual_std,
                'ljung_box_statistic': lb_stat,
                'autocorrelations': autocorrs[:5]  # First 5 lags
            },
            'theoretical_properties': {
                'effective_memory_span': 3 / self.final_alpha,
                'steady_state_variance_ratio': self.final_alpha**2 / (2 - self.final_alpha),
                'bias_for_trend': 'Non-zero for trending data'
            }
        }
        
        return statistics
    
    def plot_results(self, data, title="Simple Exponential Smoothing Results"):
        """Comprehensive visualization of results"""
        
        if self.fitted_values is None:
            raise ValueError("Model must be fitted first")
            
        data = np.asarray(data)
        time_index = np.arange(len(data))
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Plot 1: Original vs Fitted
        axes[0, 0].plot(time_index, data, 'b-', label='Original Data', linewidth=1.5)
        axes[0, 0].plot(time_index, self.fitted_values, 'r--', label='Fitted Values', linewidth=1.5)
        axes[0, 0].plot(time_index, self.smoothed_values, 'g:', label='Smoothed Values', linewidth=2)
        axes[0, 0].set_title(f'Original vs Fitted (α={self.final_alpha:.3f})')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Residuals
        axes[0, 1].plot(time_index[1:], self.residuals[1:], 'ko-', markersize=3)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.7)
        axes[0, 1].set_title('Residuals')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Residual')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Residual histogram
        axes[1, 0].hist(self.residuals[1:], bins=20, density=True, alpha=0.7, color='skyblue')
        axes[1, 0].set_title('Residual Distribution')
        axes[1, 0].set_xlabel('Residual Value')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Forecasts
        forecast_steps = min(len(data)//4, 10)
        forecasts = self.forecast(forecast_steps)
        forecast_index = np.arange(len(data), len(data) + forecast_steps)
        
        # Plot last portion of data + forecasts
        plot_start = max(0, len(data) - 20)
        axes[1, 1].plot(time_index[plot_start:], data[plot_start:], 'b-', 
                       label='Historical Data', linewidth=1.5)
        axes[1, 1].plot(forecast_index, forecasts, 'r--o', 
                       label=f'Forecasts ({forecast_steps} steps)', linewidth=1.5, markersize=4)
        axes[1, 1].set_title('Forecasts')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig

def demonstrate_exponential_smoothing():
    """Comprehensive demonstration of Simple Exponential Smoothing"""
    
    print("=== Simple Exponential Smoothing Demonstration ===")
    
    # Generate sample data with trend and noise
    np.random.seed(42)
    n = 100
    t = np.arange(n)
    
    # Create synthetic time series
    trend = 0.5 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 12)
    noise = np.random.normal(0, 5, n)
    data = 50 + trend + seasonal + noise
    
    print("\n1. Data Generation:")
    print(f"   • Series length: {n}")
    print(f"   • Components: trend + seasonal + noise")
    print(f"   • Mean: {np.mean(data):.2f}")
    print(f"   • Std: {np.std(data):.2f}")
    
    # Initialize and fit model
    print("\n2. Model Fitting:")
    
    # Fit with optimization
    ses_model = SimpleExponentialSmoothing(optimize=True)
    ses_model.fit(data)
    
    print(f"   • Optimized α: {ses_model.final_alpha:.4f}")
    print(f"   • Initial value: {ses_model.initial_value:.3f}")
    print(f"   • Effective memory span: {3/ses_model.final_alpha:.1f} periods")
    
    # Get model statistics
    stats = ses_model.get_model_statistics(data)
    
    print("\n3. Model Performance:")
    print(f"   • RMSE: {stats['error_metrics']['rmse']:.3f}")
    print(f"   • MAE: {stats['error_metrics']['mae']:.3f}")
    print(f"   • MAPE: {stats['error_metrics']['mape']:.2f}%")
    print(f"   • AIC: {stats['information_criteria']['aic']:.2f}")
    
    print("\n4. Residual Diagnostics:")
    print(f"   • Residual mean: {stats['residual_diagnostics']['mean']:.4f}")
    print(f"   • Residual std: {stats['residual_diagnostics']['std']:.3f}")
    print(f"   • Ljung-Box statistic: {stats['residual_diagnostics']['ljung_box_statistic']:.2f}")
    
    # Generate forecasts
    forecasts = ses_model.forecast(10)
    print(f"\n5. Forecasting:")
    print(f"   • 10-step ahead forecasts: {forecasts[0]:.2f} (constant)")
    print(f"   • Forecast interpretation: Last smoothed value")
    
    # Compare different alpha values
    print("\n6. Alpha Sensitivity Analysis:")
    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    for alpha in alphas:
        ses_temp = SimpleExponentialSmoothing(alpha=alpha, optimize=False)
        ses_temp.fit(data)
        temp_stats = ses_temp.get_model_statistics(data)
        rmse = temp_stats['error_metrics']['rmse']
        print(f"   • α={alpha}: RMSE={rmse:.3f}")
    
    # Theoretical insights
    print("\n7. Theoretical Properties:")
    foundations = ses_model._theoretical_foundations()
    
    print("   • Exponential weights decrease as α(1-α)^k")
    print(f"   • 95% weight concentrated in ≈ {3/ses_model.final_alpha:.0f} recent observations")
    print("   • Method assumes level-constant model (no trend/seasonality)")
    print("   • Optimal for short-term forecasting of smooth series")
    
    # Plot results
    ses_model.plot_results(data, "Simple Exponential Smoothing Analysis")
    
    return ses_model, stats

# Execute demonstration
if __name__ == "__main__":
    model, statistics = demonstrate_exponential_smoothing()
```

**Key Theoretical Insights:**

1. **Exponential Weight Decay**: Weights follow w_k = α(1-α)^k, ensuring recent observations have more influence

2. **Memory Properties**: Infinite memory with exponential decay; effective memory span ≈ 3/α periods

3. **Bias-Variance Tradeoff**: Higher α reduces bias but increases variance in forecasts

4. **Steady-State Properties**: For stationary series, smoothed values converge to true level

5. **Optimization Theory**: α minimizes MSE of one-step-ahead forecasts

6. **Limitations**: Assumes constant level; poor performance with trend or strong seasonality

7. **Forecast Properties**: All future forecasts equal last smoothed value (naive approach)

8. **Computational Efficiency**: O(n) time complexity; suitable for real-time applications

**Practical Applications:**
- **Inventory Management**: Short-term demand forecasting
- **Financial Markets**: Smoothing noisy price data
- **Quality Control**: Process monitoring and control charts
- **Signal Processing**: Noise reduction and trend extraction

---

## Question 2

**Usingpandas, write ascriptto detectseasonalityin atime series dataset.**

**Answer:**

**Theoretical Foundation:**

Seasonality detection involves identifying **recurring patterns** at fixed intervals in time series data. This requires understanding **multiple statistical approaches** and **frequency domain analysis** to distinguish between genuine seasonal patterns and random fluctuations.

**Mathematical Framework:**

**Seasonal Decomposition Model:**
```
Y_t = T_t + S_t + E_t  (Additive)
Y_t = T_t × S_t × E_t  (Multiplicative)

Where:
- Y_t: Observed value at time t
- T_t: Trend component
- S_t: Seasonal component  
- E_t: Error/irregular component
```

**Statistical Tests for Seasonality:**
```
H₀: No seasonality present
H₁: Seasonal pattern exists

Tests:
1. Kruskal-Wallis test
2. Friedman test  
3. Spectral analysis
4. Autocorrelation analysis
```

**Comprehensive Implementation:**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import kruskal, friedmanchisquare
from scipy.fft import fft, fftfreq
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class SeasonalityDetector:
    """
    Comprehensive seasonality detection and analysis framework
    using multiple statistical and signal processing approaches
    """
    
    def __init__(self):
        self.results = {}
        self.seasonal_patterns = {}
        self.statistical_tests = {}
        
    def theoretical_foundations(self):
        """Theoretical framework for seasonality detection"""
        
        foundations = {
            'seasonal_definitions': {
                'deterministic_seasonality': {
                    'definition': 'Fixed seasonal patterns that repeat exactly',
                    'characteristics': 'Constant amplitude and phase',
                    'examples': 'Calendar effects, business cycles'
                },
                'stochastic_seasonality': {
                    'definition': 'Seasonal patterns with random variations',
                    'characteristics': 'Time-varying amplitude/phase',
                    'examples': 'Weather patterns, economic seasonality'
                },
                'multiple_seasonality': {
                    'definition': 'Multiple seasonal patterns at different frequencies',
                    'characteristics': 'Hierarchical seasonal structure',
                    'examples': 'Daily and weekly patterns in electricity demand'
                }
            },
            'detection_approaches': {
                'visual_methods': {
                    'seasonal_plots': 'Plot data by seasonal periods',
                    'subseries_plots': 'Compare values within seasons',
                    'lag_plots': 'Plot Y_t vs Y_{t-s}'
                },
                'statistical_tests': {
                    'kruskal_wallis': 'Non-parametric test for group differences',
                    'friedman_test': 'Test for differences across blocks',
                    'ljung_box': 'Test for autocorrelation at seasonal lags'
                },
                'frequency_domain': {
                    'spectral_analysis': 'Identify dominant frequencies',
                    'periodogram': 'Power spectral density estimation',
                    'autocorrelation': 'Detect periodic correlations'
                }
            },
            'mathematical_properties': {
                'fourier_analysis': {
                    'frequency_resolution': 'Δf = 1/(N×Δt)',
                    'nyquist_frequency': 'f_max = 1/(2×Δt)',
                    'seasonal_frequency': 'f_s = 1/s (s = seasonal period)'
                },
                'autocorrelation_theory': {
                    'seasonal_acf': 'ρ(s), ρ(2s), ρ(3s) > threshold',
                    'significance_level': '±1.96/√n for 95% confidence',
                    'partial_autocorr': 'Removes intermediate correlations'
                }
            }
        }
        
        return foundations
    
    def detect_seasonality_comprehensive(self, data, date_column=None, value_column=None, 
                                       potential_periods=None, significance_level=0.05):
        """
        Comprehensive seasonality detection using multiple approaches
        
        Parameters:
        -----------
        data : pd.DataFrame or pd.Series
            Time series data
        date_column : str, optional
            Name of date column if DataFrame
        value_column : str, optional  
            Name of value column if DataFrame
        potential_periods : list, optional
            List of potential seasonal periods to test
        significance_level : float, default=0.05
            Statistical significance level
        """
        
        # Data preprocessing
        ts_data, time_index = self._preprocess_data(data, date_column, value_column)
        
        # Set default potential periods based on data frequency
        if potential_periods is None:
            potential_periods = self._infer_potential_periods(time_index)
        
        print("=== Comprehensive Seasonality Detection ===")
        print(f"Data points: {len(ts_data)}")
        print(f"Testing seasonal periods: {potential_periods}")
        
        # 1. Visual inspection
        visual_results = self._visual_seasonality_analysis(ts_data, time_index, potential_periods)
        
        # 2. Statistical tests
        statistical_results = self._statistical_seasonality_tests(ts_data, potential_periods, significance_level)
        
        # 3. Frequency domain analysis
        frequency_results = self._frequency_domain_analysis(ts_data, time_index, potential_periods)
        
        # 4. Autocorrelation analysis
        autocorr_results = self._autocorrelation_analysis(ts_data, potential_periods, significance_level)
        
        # 5. Decomposition analysis
        decomposition_results = self._decomposition_analysis(ts_data, time_index, potential_periods)
        
        # 6. Machine learning approach
        ml_results = self._ml_seasonality_detection(ts_data, time_index, potential_periods)
        
        # Combine results
        combined_results = self._combine_seasonality_evidence(
            statistical_results, frequency_results, autocorr_results, 
            decomposition_results, ml_results, potential_periods
        )
        
        # Store comprehensive results
        self.results = {
            'visual_analysis': visual_results,
            'statistical_tests': statistical_results,
            'frequency_analysis': frequency_results,
            'autocorrelation_analysis': autocorr_results,
            'decomposition_analysis': decomposition_results,
            'ml_analysis': ml_results,
            'combined_assessment': combined_results,
            'data_info': {
                'length': len(ts_data),
                'potential_periods': potential_periods,
                'significance_level': significance_level
            }
        }
        
        return self.results
    
    def _preprocess_data(self, data, date_column, value_column):
        """Preprocess and validate input data"""
        
        if isinstance(data, pd.Series):
            ts_data = data.values
            time_index = data.index
        elif isinstance(data, pd.DataFrame):
            if value_column is None:
                raise ValueError("value_column must be specified for DataFrame input")
            ts_data = data[value_column].values
            
            if date_column is not None:
                time_index = pd.to_datetime(data[date_column])
            else:
                time_index = data.index
        else:
            # Assume numpy array
            ts_data = np.asarray(data)
            time_index = pd.date_range(start='2020-01-01', periods=len(ts_data), freq='D')
        
        # Remove NaN values
        valid_mask = ~np.isnan(ts_data)
        ts_data = ts_data[valid_mask]
        time_index = time_index[valid_mask] if hasattr(time_index, '__getitem__') else time_index
        
        return ts_data, time_index
    
    def _infer_potential_periods(self, time_index):
        """Infer potential seasonal periods based on data frequency"""
        
        # Try to infer frequency
        if hasattr(time_index, 'freq') and time_index.freq is not None:
            freq = time_index.freq
        elif len(time_index) > 1:
            # Estimate frequency from time differences
            time_diff = pd.Series(time_index).diff().median()
            
            if time_diff <= pd.Timedelta(hours=1):
                freq = 'H'  # Hourly
            elif time_diff <= pd.Timedelta(days=1):
                freq = 'D'  # Daily
            elif time_diff <= pd.Timedelta(weeks=1):
                freq = 'W'  # Weekly
            else:
                freq = 'M'  # Monthly
        else:
            freq = 'D'  # Default to daily
        
        # Set potential periods based on frequency
        if freq in ['H', 'T', 'S']:  # High frequency
            potential_periods = [24, 168, 720]  # Daily, weekly, monthly
        elif freq == 'D':  # Daily
            potential_periods = [7, 30, 90, 365]  # Weekly, monthly, quarterly, yearly
        elif freq == 'W':  # Weekly
            potential_periods = [4, 13, 52]  # Monthly, quarterly, yearly
        elif freq == 'M':  # Monthly
            potential_periods = [3, 4, 6, 12]  # Quarterly, seasonal, yearly
        else:
            potential_periods = [4, 7, 12, 24]  # Generic periods
        
        # Filter periods that are reasonable for data length
        max_period = len(time_index) // 3
        potential_periods = [p for p in potential_periods if p <= max_period and p >= 2]
        
        return potential_periods
    
    def _visual_seasonality_analysis(self, data, time_index, periods):
        """Visual analysis of seasonal patterns"""
        
        visual_results = {}
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Visual Seasonality Analysis', fontsize=16, fontweight='bold')
        
        # Time series plot
        axes[0, 0].plot(time_index, data, linewidth=1)
        axes[0, 0].set_title('Original Time Series')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Seasonal subseries plots for dominant period
        if periods:
            dominant_period = periods[0]
            n_seasons = len(data) // dominant_period
            
            if n_seasons >= 2:
                seasonal_data = data[:n_seasons * dominant_period].reshape(n_seasons, dominant_period)
                
                # Box plot by season
                axes[0, 1].boxplot([seasonal_data[:, i] for i in range(min(12, dominant_period))])
                axes[0, 1].set_title(f'Seasonal Patterns (Period={dominant_period})')
                axes[0, 1].set_xlabel('Season')
                axes[0, 1].set_ylabel('Value')
                axes[0, 1].grid(True, alpha=0.3)
                
                # Seasonal means
                seasonal_means = np.mean(seasonal_data, axis=0)
                visual_results['seasonal_means'] = seasonal_means
                visual_results['seasonal_variance'] = np.var(seasonal_means)
        
        # Lag plots for seasonal periods
        if len(periods) > 0:
            lag = periods[0]
            if lag < len(data):
                axes[1, 0].scatter(data[:-lag], data[lag:], alpha=0.6)
                axes[1, 0].plot([data.min(), data.max()], [data.min(), data.max()], 'r--', alpha=0.8)
                axes[1, 0].set_title(f'Lag Plot (lag={lag})')
                axes[1, 0].set_xlabel(f'Y(t)')
                axes[1, 0].set_ylabel(f'Y(t+{lag})')
                axes[1, 0].grid(True, alpha=0.3)
        
        # Moving average plot
        if len(data) > 30:
            window = min(30, len(data) // 10)
            ma = pd.Series(data).rolling(window=window).mean()
            axes[1, 1].plot(time_index, data, alpha=0.5, label='Original')
            axes[1, 1].plot(time_index, ma, linewidth=2, label=f'MA({window})')
            axes[1, 1].set_title('Trend Analysis')
            axes[1, 1].set_xlabel('Time')
            axes[1, 1].set_ylabel('Value')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return visual_results
    
    def _statistical_seasonality_tests(self, data, periods, significance_level):
        """Statistical tests for seasonality"""
        
        test_results = {}
        
        for period in periods:
            if period >= len(data):
                continue
                
            period_results = {}
            
            # Kruskal-Wallis test
            try:
                # Group data by seasonal period
                n_complete_cycles = len(data) // period
                if n_complete_cycles >= 2:
                    seasonal_groups = []
                    for season in range(period):
                        group_data = []
                        for cycle in range(n_complete_cycles):
                            idx = cycle * period + season
                            if idx < len(data):
                                group_data.append(data[idx])
                        if len(group_data) >= 2:
                            seasonal_groups.append(group_data)
                    
                    if len(seasonal_groups) >= 2:
                        kw_stat, kw_pvalue = kruskal(*seasonal_groups)
                        period_results['kruskal_wallis'] = {
                            'statistic': kw_stat,
                            'p_value': kw_pvalue,
                            'significant': kw_pvalue < significance_level
                        }
            except Exception as e:
                period_results['kruskal_wallis'] = {'error': str(e)}
            
            # Friedman test (if enough data)
            try:
                if n_complete_cycles >= 3 and period <= 20:
                    # Reshape data for Friedman test
                    friedman_data = data[:n_complete_cycles * period].reshape(n_complete_cycles, period)
                    
                    friedman_stat, friedman_pvalue = friedmanchisquare(*friedman_data.T)
                    period_results['friedman'] = {
                        'statistic': friedman_stat,
                        'p_value': friedman_pvalue,
                        'significant': friedman_pvalue < significance_level
                    }
            except Exception as e:
                period_results['friedman'] = {'error': str(e)}
            
            # Ljung-Box test at seasonal lag
            try:
                if period < len(data) // 2:
                    from statsmodels.stats.diagnostic import acorr_ljungbox
                    ljung_result = acorr_ljungbox(data, lags=[period], return_df=True)
                    
                    period_results['ljung_box'] = {
                        'statistic': ljung_result['lb_stat'].iloc[0],
                        'p_value': ljung_result['lb_pvalue'].iloc[0],
                        'significant': ljung_result['lb_pvalue'].iloc[0] < significance_level
                    }
            except Exception as e:
                period_results['ljung_box'] = {'error': str(e)}
            
            test_results[f'period_{period}'] = period_results
        
        return test_results
    
    def _frequency_domain_analysis(self, data, time_index, periods):
        """Frequency domain analysis for seasonality"""
        
        # FFT analysis
        n = len(data)
        fft_vals = fft(data - np.mean(data))  # Remove DC component
        freqs = fftfreq(n, d=1.0)  # Normalized frequencies
        power_spectrum = np.abs(fft_vals)**2
        
        # Focus on positive frequencies
        pos_freqs = freqs[:n//2]
        pos_power = power_spectrum[:n//2]
        
        # Find dominant frequencies
        peak_indices = []
        threshold = np.mean(pos_power) + 2 * np.std(pos_power)
        
        for i in range(1, len(pos_power)-1):
            if (pos_power[i] > threshold and 
                pos_power[i] > pos_power[i-1] and 
                pos_power[i] > pos_power[i+1]):
                peak_indices.append(i)
        
        # Convert frequency peaks to periods
        detected_periods = []
        for idx in peak_indices:
            if pos_freqs[idx] > 0:
                period = 1.0 / pos_freqs[idx]
                if 2 <= period <= n//3:  # Reasonable period range
                    detected_periods.append(period)
        
        # Spectral analysis for each candidate period
        period_analysis = {}
        for period in periods:
            if period < n//2:
                # Expected frequency for this period
                expected_freq = 1.0 / period
                
                # Find closest frequency in spectrum
                freq_idx = np.argmin(np.abs(pos_freqs - expected_freq))
                actual_freq = pos_freqs[freq_idx]
                power_at_freq = pos_power[freq_idx]
                
                # Statistical significance of peak
                noise_level = np.median(pos_power)
                signal_to_noise = power_at_freq / (noise_level + 1e-8)
                
                period_analysis[f'period_{period}'] = {
                    'expected_frequency': expected_freq,
                    'actual_frequency': actual_freq,
                    'power': power_at_freq,
                    'signal_to_noise_ratio': signal_to_noise,
                    'significant': signal_to_noise > 3.0  # Threshold for significance
                }
        
        frequency_results = {
            'dominant_frequencies': pos_freqs[peak_indices],
            'detected_periods': detected_periods,
            'period_analysis': period_analysis,
            'power_spectrum': pos_power,
            'frequencies': pos_freqs
        }
        
        return frequency_results
    
    def _autocorrelation_analysis(self, data, periods, significance_level):
        """Autocorrelation analysis for seasonality"""
        
        # Calculate ACF
        max_lag = min(len(data)//2, max(periods) * 2) if periods else len(data)//4
        acf_vals = acf(data, nlags=max_lag, alpha=significance_level)
        
        # Significance bounds
        n = len(data)
        confidence_interval = 1.96 / np.sqrt(n)
        
        autocorr_results = {}
        
        for period in periods:
            if period <= max_lag:
                # Check autocorrelation at seasonal lags
                seasonal_acf = []
                significant_lags = []
                
                for mult in range(1, 4):  # Check first 3 multiples
                    lag = period * mult
                    if lag <= max_lag:
                        acf_val = acf_vals[0][lag]  # acf returns tuple (values, confint)
                        seasonal_acf.append(acf_val)
                        
                        if abs(acf_val) > confidence_interval:
                            significant_lags.append(lag)
                
                autocorr_results[f'period_{period}'] = {
                    'seasonal_autocorrelations': seasonal_acf,
                    'significant_lags': significant_lags,
                    'max_seasonal_acf': max(seasonal_acf) if seasonal_acf else 0,
                    'seasonality_strength': np.mean([abs(x) for x in seasonal_acf]) if seasonal_acf else 0
                }
        
        autocorr_results['full_acf'] = acf_vals[0]
        autocorr_results['confidence_interval'] = confidence_interval
        
        return autocorr_results
    
    def _decomposition_analysis(self, data, time_index, periods):
        """Seasonal decomposition analysis"""
        
        decomposition_results = {}
        
        for period in periods:
            if period >= 4 and period < len(data)//2:
                try:
                    # Create pandas series for decomposition
                    ts_series = pd.Series(data, index=time_index)
                    
                    # Classical decomposition
                    if len(data) >= 2 * period:
                        decomp = seasonal_decompose(ts_series, model='additive', period=period)
                        
                        # Analyze seasonal component
                        seasonal_component = decomp.seasonal.dropna()
                        
                        # Seasonal strength (variance of seasonal / variance of deseasonalized)
                        deseasonalized = data - seasonal_component.values[:len(data)]
                        seasonal_strength = np.var(seasonal_component) / (np.var(deseasonalized) + 1e-8)
                        
                        # Consistency of seasonal pattern
                        n_seasons = len(seasonal_component) // period
                        if n_seasons >= 2:
                            seasonal_matrix = seasonal_component.values[:n_seasons*period].reshape(n_seasons, period)
                            seasonal_consistency = 1 - np.mean(np.std(seasonal_matrix, axis=0)) / (np.std(seasonal_component) + 1e-8)
                        else:
                            seasonal_consistency = 0
                        
                        decomposition_results[f'period_{period}'] = {
                            'seasonal_strength': seasonal_strength,
                            'seasonal_consistency': seasonal_consistency,
                            'seasonal_range': np.ptp(seasonal_component),
                            'trend_strength': np.var(decomp.trend.dropna()) / (np.var(data) + 1e-8),
                            'significant_seasonality': seasonal_strength > 0.1 and seasonal_consistency > 0.5
                        }
                        
                except Exception as e:
                    decomposition_results[f'period_{period}'] = {'error': str(e)}
        
        return decomposition_results
    
    def _ml_seasonality_detection(self, data, time_index, periods):
        """Machine learning approach to seasonality detection"""
        
        ml_results = {}
        
        # Feature engineering for seasonal patterns
        for period in periods:
            if period < len(data)//3:
                try:
                    # Create seasonal features
                    features = []
                    for i in range(len(data)):
                        seasonal_pos = i % period
                        features.append([
                            seasonal_pos / period,  # Normalized position
                            np.sin(2 * np.pi * seasonal_pos / period),  # Sine component
                            np.cos(2 * np.pi * seasonal_pos / period)   # Cosine component
                        ])
                    
                    features = np.array(features)
                    
                    # Use k-means clustering to detect seasonal groups
                    if len(data) >= period * 2:
                        scaler = StandardScaler()
                        data_scaled = scaler.fit_transform(data.reshape(-1, 1))
                        features_scaled = scaler.fit_transform(features)
                        
                        # Combine time series values with seasonal features
                        combined_features = np.hstack([data_scaled, features_scaled])
                        
                        # K-means with number of clusters = period
                        kmeans = KMeans(n_clusters=min(period, 10), random_state=42, n_init=10)
                        cluster_labels = kmeans.fit_predict(combined_features)
                        
                        # Analyze cluster consistency with seasonal positions
                        seasonal_positions = np.array([i % period for i in range(len(data))])
                        
                        # Calculate how well clusters align with seasonal positions
                        alignment_score = 0
                        for cluster_id in range(kmeans.n_clusters):
                            cluster_mask = cluster_labels == cluster_id
                            if np.sum(cluster_mask) > 0:
                                cluster_positions = seasonal_positions[cluster_mask]
                                # Measure consistency of seasonal positions within cluster
                                position_std = np.std(cluster_positions)
                                alignment_score += 1 / (1 + position_std)
                        
                        alignment_score /= kmeans.n_clusters
                        
                        ml_results[f'period_{period}'] = {
                            'clustering_alignment': alignment_score,
                            'inertia': kmeans.inertia_,
                            'n_clusters': kmeans.n_clusters,
                            'significant_seasonality': alignment_score > 0.5
                        }
                        
                except Exception as e:
                    ml_results[f'period_{period}'] = {'error': str(e)}
        
        return ml_results
    
    def _combine_seasonality_evidence(self, statistical_results, frequency_results, 
                                    autocorr_results, decomposition_results, 
                                    ml_results, periods):
        """Combine evidence from all methods to assess seasonality"""
        
        combined_assessment = {}
        
        for period in periods:
            period_key = f'period_{period}'
            evidence_count = 0
            evidence_strength = 0
            evidence_details = {}
            
            # Statistical evidence
            if period_key in statistical_results:
                stat_results = statistical_results[period_key]
                stat_evidence = 0
                
                for test_name, test_result in stat_results.items():
                    if isinstance(test_result, dict) and 'significant' in test_result:
                        if test_result['significant']:
                            stat_evidence += 1
                            evidence_details[f'statistical_{test_name}'] = 'significant'
                
                if stat_evidence > 0:
                    evidence_count += 1
                    evidence_strength += stat_evidence / len(stat_results)
            
            # Frequency domain evidence
            if period_key in frequency_results['period_analysis']:
                freq_result = frequency_results['period_analysis'][period_key]
                if freq_result.get('significant', False):
                    evidence_count += 1
                    evidence_strength += freq_result.get('signal_to_noise_ratio', 0) / 10
                    evidence_details['frequency_domain'] = f"SNR: {freq_result.get('signal_to_noise_ratio', 0):.2f}"
            
            # Autocorrelation evidence
            if period_key in autocorr_results:
                autocorr_result = autocorr_results[period_key]
                seasonality_strength = autocorr_result.get('seasonality_strength', 0)
                if seasonality_strength > 0.1:  # Threshold for significance
                    evidence_count += 1
                    evidence_strength += seasonality_strength
                    evidence_details['autocorrelation'] = f"strength: {seasonality_strength:.3f}"
            
            # Decomposition evidence
            if period_key in decomposition_results:
                decomp_result = decomposition_results[period_key]
                if decomp_result.get('significant_seasonality', False):
                    evidence_count += 1
                    evidence_strength += decomp_result.get('seasonal_strength', 0)
                    evidence_details['decomposition'] = f"strength: {decomp_result.get('seasonal_strength', 0):.3f}"
            
            # ML evidence
            if period_key in ml_results:
                ml_result = ml_results[period_key]
                if ml_result.get('significant_seasonality', False):
                    evidence_count += 1
                    evidence_strength += ml_result.get('clustering_alignment', 0)
                    evidence_details['machine_learning'] = f"alignment: {ml_result.get('clustering_alignment', 0):.3f}"
            
            # Overall assessment
            total_methods = 5  # Number of method categories
            evidence_ratio = evidence_count / total_methods
            avg_strength = evidence_strength / max(evidence_count, 1)
            
            # Classification
            if evidence_ratio >= 0.6 and avg_strength > 0.3:
                conclusion = "Strong seasonality"
            elif evidence_ratio >= 0.4 and avg_strength > 0.2:
                conclusion = "Moderate seasonality"
            elif evidence_ratio >= 0.2:
                conclusion = "Weak seasonality"
            else:
                conclusion = "No significant seasonality"
            
            combined_assessment[period_key] = {
                'period': period,
                'evidence_count': evidence_count,
                'evidence_ratio': evidence_ratio,
                'average_strength': avg_strength,
                'conclusion': conclusion,
                'evidence_details': evidence_details
            }
        
        # Find most likely seasonal period
        best_period = None
        best_score = 0
        
        for period_key, assessment in combined_assessment.items():
            score = assessment['evidence_ratio'] * assessment['average_strength']
            if score > best_score:
                best_score = score
                best_period = assessment['period']
        
        combined_assessment['overall_conclusion'] = {
            'most_likely_period': best_period,
            'confidence_score': best_score,
            'has_seasonality': best_score > 0.1
        }
        
        return combined_assessment

def demonstrate_seasonality_detection():
    """Comprehensive demonstration of seasonality detection"""
    
    print("=== Seasonality Detection Demonstration ===")
    
    # Generate synthetic data with known seasonality
    np.random.seed(42)
    n = 365  # One year of daily data
    t = np.arange(n)
    
    # Create time series with multiple seasonal components
    trend = 0.02 * t
    annual_seasonal = 20 * np.sin(2 * np.pi * t / 365)  # Annual
    weekly_seasonal = 5 * np.sin(2 * np.pi * t / 7)     # Weekly
    noise = np.random.normal(0, 3, n)
    
    data = 100 + trend + annual_seasonal + weekly_seasonal + noise
    
    # Create DataFrame
    dates = pd.date_range(start='2023-01-01', periods=n, freq='D')
    df = pd.DataFrame({
        'date': dates,
        'value': data
    })
    
    print(f"\nGenerated synthetic data:")
    print(f"  • Length: {n} days")
    print(f"  • Known seasonality: 7-day (weekly) and 365-day (annual)")
    print(f"  • Mean: {np.mean(data):.2f}")
    print(f"  • Std: {np.std(data):.2f}")
    
    # Initialize detector
    detector = SeasonalityDetector()
    
    # Detect seasonality
    results = detector.detect_seasonality_comprehensive(
        df, 
        date_column='date', 
        value_column='value',
        potential_periods=[7, 30, 90, 365]
    )
    
    # Display results
    print("\n=== Detection Results ===")
    
    combined_results = results['combined_assessment']
    overall_conclusion = combined_results['overall_conclusion']
    
    print(f"\nOverall Conclusion:")
    print(f"  • Has seasonality: {overall_conclusion['has_seasonality']}")
    print(f"  • Most likely period: {overall_conclusion['most_likely_period']}")
    print(f"  • Confidence score: {overall_conclusion['confidence_score']:.3f}")
    
    print(f"\nDetailed Analysis by Period:")
    for period_key, assessment in combined_results.items():
        if period_key != 'overall_conclusion':
            period = assessment['period']
            conclusion = assessment['conclusion']
            evidence_ratio = assessment['evidence_ratio']
            avg_strength = assessment['average_strength']
            
            print(f"  • Period {period}: {conclusion}")
            print(f"    - Evidence ratio: {evidence_ratio:.2f}")
            print(f"    - Average strength: {avg_strength:.3f}")
            
            if assessment['evidence_details']:
                print(f"    - Supporting evidence: {list(assessment['evidence_details'].keys())}")
    
    # Test with non-seasonal data
    print(f"\n=== Testing with Non-Seasonal Data ===")
    
    # Generate random walk (no seasonality)
    random_walk = np.cumsum(np.random.normal(0, 1, 200))
    df_noseasonal = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=200, freq='D'),
        'value': random_walk
    })
    
    results_noseasonal = detector.detect_seasonality_comprehensive(
        df_noseasonal,
        date_column='date',
        value_column='value',
        potential_periods=[7, 30]
    )
    
    overall_noseasonal = results_noseasonal['combined_assessment']['overall_conclusion']
    print(f"Non-seasonal data results:")
    print(f"  • Has seasonality: {overall_noseasonal['has_seasonality']}")
    print(f"  • Confidence score: {overall_noseasonal['confidence_score']:.3f}")
    
    return detector, results

# Execute demonstration
if __name__ == "__main__":
    detector, results = demonstrate_seasonality_detection()
```

**Key Theoretical Insights:**

1. **Multi-Method Approach**: Combines statistical tests, frequency analysis, autocorrelation, decomposition, and machine learning for robust detection

2. **Statistical Foundation**: Uses Kruskal-Wallis and Friedman tests for non-parametric seasonal group comparisons

3. **Frequency Domain Analysis**: Applies FFT to identify dominant frequencies corresponding to seasonal periods

4. **Autocorrelation Structure**: Examines ACF at seasonal lags to detect periodic correlations

5. **Decomposition Theory**: Separates trend, seasonal, and residual components to quantify seasonal strength

6. **Evidence Integration**: Combines multiple sources of evidence for confident seasonality assessment

7. **Significance Testing**: Applies appropriate statistical thresholds for each detection method

8. **Robustness**: Handles various data characteristics and potential periods automatically

**Practical Applications:**
- **Business Analytics**: Sales forecasting, inventory management
- **Financial Markets**: Trading strategy development, risk management  
- **Operations Research**: Resource planning, demand forecasting
- **Quality Control**: Process monitoring, pattern detection

---

## Question 3

**Code anARIMA modelinPythonon a given dataset and visualize the forecasts.**

**Answer:**

**Theoretical Foundation:**

ARIMA (AutoRegressive Integrated Moving Average) models represent a comprehensive framework for **univariate time series forecasting** that combines autoregression, differencing, and moving averages. The mathematical foundation integrates **Box-Jenkins methodology** with **maximum likelihood estimation**.

**Mathematical Framework:**

**ARIMA(p,d,q) Model Structure:**
```
(1 - φ₁L - φ₂L² - ... - φₚLᵖ)(1-L)ᵈXₜ = (1 + θ₁L + θ₂L² + ... + θₑLᵠ)εₜ

Where:
- L: Lag operator (LXₜ = Xₜ₋₁)
- φᵢ: Autoregressive parameters
- θⱼ: Moving average parameters  
- εₜ: White noise error term
- d: Degree of differencing
```

**Component Analysis:**
```
AR(p): Xₜ = φ₁Xₜ₋₁ + φ₂Xₜ₋₂ + ... + φₚXₜ₋ₚ + εₜ
MA(q): Xₜ = εₜ + θ₁εₜ₋₁ + θ₂εₜ₋₂ + ... + θₑεₜ₋ₑ
I(d): ∇ᵈXₜ = (1-L)ᵈXₜ (differencing to achieve stationarity)
```

**Comprehensive Implementation:**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
import itertools
import warnings
warnings.filterwarnings('ignore')

class ARIMAAnalysis:
    """
    Comprehensive ARIMA modeling framework with theoretical foundations,
    automated model selection, and advanced diagnostics
    """
    
    def __init__(self):
        self.model = None
        self.fitted_model = None
        self.results = {}
        self.diagnostics = {}
        
    def theoretical_foundations(self):
        """Mathematical and theoretical foundations of ARIMA modeling"""
        
        foundations = {
            'stationarity_theory': {
                'definition': 'Constant mean, variance, and autocovariance structure',
                'weak_stationarity': 'E[Xₜ] = μ, Var(Xₜ) = σ², Cov(Xₜ,Xₜ₊ₕ) = γ(h)',
                'strict_stationarity': 'Joint distribution invariant under time shifts',
                'importance': 'Required for consistent parameter estimation'
            },
            'box_jenkins_methodology': {
                'identification': 'Use ACF/PACF to determine (p,d,q) orders',
                'estimation': 'Maximum likelihood estimation of parameters',
                'diagnostic_checking': 'Residual analysis and model validation',
                'forecasting': 'Generate predictions with confidence intervals'
            },
            'model_identification_rules': {
                'ar_pattern': 'PACF cuts off at lag p, ACF decays exponentially',
                'ma_pattern': 'ACF cuts off at lag q, PACF decays exponentially',
                'arma_pattern': 'Both ACF and PACF decay exponentially',
                'differencing': 'Apply until unit root tests indicate stationarity'
            },
            'estimation_theory': {
                'likelihood_function': 'L(θ) = ∏ f(xₜ|xₜ₋₁,...,x₁;θ)',
                'information_matrix': 'I(θ) = E[-∂²log L/∂θ∂θᵀ]',
                'asymptotic_properties': 'Consistency, efficiency, normality',
                'numerical_optimization': 'Newton-Raphson, BFGS algorithms'
            },
            'forecasting_theory': {
                'minimum_mse_predictor': 'E[Xₜ₊ₕ|Iₜ] minimizes forecast error variance',
                'prediction_intervals': 'Based on forecast error variance',
                'forecast_updating': 'Kalman filter for real-time updating',
                'forecast_evaluation': 'Multiple accuracy measures and tests'
            }
        }
        
        return foundations
    
    def comprehensive_stationarity_testing(self, data, alpha=0.05):
        """
        Comprehensive stationarity testing using multiple approaches
        
        Parameters:
        -----------
        data : array-like
            Time series data
        alpha : float, default=0.05
            Significance level for tests
        """
        
        print("=== Comprehensive Stationarity Analysis ===")
        
        stationarity_results = {}
        
        # 1. Augmented Dickey-Fuller Test
        print("\n1. Augmented Dickey-Fuller Test:")
        adf_result = adfuller(data, autolag='AIC')
        
        adf_stats = {
            'test_statistic': adf_result[0],
            'p_value': adf_result[1],
            'critical_values': adf_result[4],
            'used_lag': adf_result[2],
            'n_observations': adf_result[3],
            'is_stationary': adf_result[1] < alpha
        }
        
        print(f"   • Test Statistic: {adf_stats['test_statistic']:.4f}")
        print(f"   • p-value: {adf_stats['p_value']:.4f}")
        print(f"   • Critical Values: {adf_stats['critical_values']}")
        print(f"   • Conclusion: {'Stationary' if adf_stats['is_stationary'] else 'Non-stationary'}")
        
        stationarity_results['adf'] = adf_stats
        
        # 2. KPSS Test (Kwiatkowski-Phillips-Schmidt-Shin)
        print("\n2. KPSS Test:")
        kpss_result = kpss(data, regression='c', nlags='auto')
        
        kpss_stats = {
            'test_statistic': kpss_result[0],
            'p_value': kpss_result[1],
            'critical_values': kpss_result[3],
            'used_lags': kpss_result[2],
            'is_stationary': kpss_result[1] > alpha  # KPSS: H0 is stationarity
        }
        
        print(f"   • Test Statistic: {kpss_stats['test_statistic']:.4f}")
        print(f"   • p-value: {kpss_stats['p_value']:.4f}")
        print(f"   • Critical Values: {kpss_stats['critical_values']}")
        print(f"   • Conclusion: {'Stationary' if kpss_stats['is_stationary'] else 'Non-stationary'}")
        
        stationarity_results['kpss'] = kpss_stats
        
        # 3. Combined interpretation
        adf_stationary = adf_stats['is_stationary']
        kpss_stationary = kpss_stats['is_stationary']
        
        if adf_stationary and kpss_stationary:
            combined_conclusion = "Stationary (both tests agree)"
        elif not adf_stationary and not kpss_stationary:
            combined_conclusion = "Non-stationary (both tests agree)"
        elif adf_stationary and not kpss_stationary:
            combined_conclusion = "Difference-stationary (trending)"
        else:
            combined_conclusion = "Trend-stationary or inconclusive"
        
        print(f"\n3. Combined Interpretation: {combined_conclusion}")
        
        stationarity_results['combined_conclusion'] = combined_conclusion
        
        return stationarity_results
    
    def determine_differencing_order(self, data, max_d=3):
        """
        Determine optimal differencing order for achieving stationarity
        
        Parameters:
        -----------
        data : array-like
            Time series data
        max_d : int, default=3
            Maximum differencing order to test
        """
        
        print("\n=== Determining Differencing Order ===")
        
        current_data = np.array(data)
        differencing_results = {}
        
        for d in range(max_d + 1):
            print(f"\nDifferencing order d={d}:")
            
            # Apply differencing
            if d == 0:
                diff_data = current_data
            else:
                diff_data = current_data.copy()
                for _ in range(d):
                    diff_data = np.diff(diff_data)
            
            # Stationarity tests
            adf_result = adfuller(diff_data, autolag='AIC')
            
            try:
                kpss_result = kpss(diff_data, regression='c', nlags='auto')
                kpss_pvalue = kpss_result[1]
                kpss_stationary = kpss_pvalue > 0.05
            except:
                kpss_pvalue = np.nan
                kpss_stationary = False
            
            adf_stationary = adf_result[1] < 0.05
            
            # Variance of differenced series (prefer lower variance)
            diff_variance = np.var(diff_data)
            
            differencing_results[d] = {
                'adf_pvalue': adf_result[1],
                'adf_stationary': adf_stationary,
                'kpss_pvalue': kpss_pvalue,
                'kpss_stationary': kpss_stationary,
                'both_stationary': adf_stationary and kpss_stationary,
                'variance': diff_variance,
                'length': len(diff_data)
            }
            
            print(f"   • ADF p-value: {adf_result[1]:.4f} ({'Stationary' if adf_stationary else 'Non-stationary'})")
            if not np.isnan(kpss_pvalue):
                print(f"   • KPSS p-value: {kpss_pvalue:.4f} ({'Stationary' if kpss_stationary else 'Non-stationary'})")
            print(f"   • Variance: {diff_variance:.4f}")
        
        # Select optimal d
        optimal_d = 0
        for d in range(max_d + 1):
            if differencing_results[d]['both_stationary']:
                optimal_d = d
                break
        
        print(f"\nOptimal differencing order: d = {optimal_d}")
        
        return optimal_d, differencing_results
    
    def identify_arima_orders(self, data, d=0, max_p=5, max_q=5):
        """
        Identify ARIMA orders using ACF/PACF analysis and information criteria
        
        Parameters:
        -----------
        data : array-like
            Time series data (should be stationary if d=0)
        d : int, default=0
            Differencing order
        max_p : int, default=5
            Maximum AR order to consider
        max_q : int, default=5
            Maximum MA order to consider
        """
        
        print(f"\n=== ARIMA Order Identification (d={d}) ===")
        
        # Apply differencing if needed
        if d > 0:
            working_data = np.array(data)
            for _ in range(d):
                working_data = np.diff(working_data)
        else:
            working_data = np.array(data)
        
        # Calculate ACF and PACF
        max_lags = min(len(working_data)//4, 20)
        
        acf_values = acf(working_data, nlags=max_lags, alpha=0.05)
        pacf_values = pacf(working_data, nlags=max_lags, alpha=0.05)
        
        # Confidence bounds
        n = len(working_data)
        confidence_bound = 1.96 / np.sqrt(n)
        
        print(f"\n1. ACF/PACF Analysis:")
        print(f"   • Series length: {len(working_data)}")
        print(f"   • Confidence bound: ±{confidence_bound:.3f}")
        
        # Identify cutoff points
        def find_cutoff(values, confidence_bound, max_search=10):
            """Find where values first become insignificant"""
            cutoff = max_search
            
            for i in range(1, min(len(values), max_search + 1)):
                if abs(values[i]) < confidence_bound:
                    cutoff = i - 1
                    break
            
            return max(0, cutoff)
        
        acf_cutoff = find_cutoff(acf_values[0][1:], confidence_bound)  # Skip lag 0
        pacf_cutoff = find_cutoff(pacf_values[0][1:], confidence_bound)  # Skip lag 0
        
        print(f"   • ACF apparent cutoff: lag {acf_cutoff}")
        print(f"   • PACF apparent cutoff: lag {pacf_cutoff}")
        
        # Initial model suggestions based on patterns
        if acf_cutoff <= 2 and pacf_cutoff > acf_cutoff:
            suggested_pattern = f"MA({acf_cutoff}) pattern"
            initial_p, initial_q = 0, acf_cutoff
        elif pacf_cutoff <= 2 and acf_cutoff > pacf_cutoff:
            suggested_pattern = f"AR({pacf_cutoff}) pattern"
            initial_p, initial_q = pacf_cutoff, 0
        else:
            suggested_pattern = f"ARMA({min(pacf_cutoff, 2)},{min(acf_cutoff, 2)}) pattern"
            initial_p, initial_q = min(pacf_cutoff, 2), min(acf_cutoff, 2)
        
        print(f"   • Suggested pattern: {suggested_pattern}")
        
        # Model selection using information criteria
        print(f"\n2. Information Criteria-Based Selection:")
        
        model_results = []
        
        # Grid search over (p,q) combinations
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                try:
                    # Fit ARIMA model
                    model = ARIMA(data, order=(p, d, q))
                    fitted_model = model.fit()
                    
                    # Extract information criteria
                    aic = fitted_model.aic
                    bic = fitted_model.bic
                    hqic = fitted_model.hqic
                    
                    # Log likelihood
                    llf = fitted_model.llf
                    
                    # Parameter significance (t-statistics)
                    params = fitted_model.params
                    pvalues = fitted_model.pvalues
                    significant_params = np.sum(pvalues < 0.05)
                    
                    model_results.append({
                        'p': p, 'q': q,
                        'aic': aic, 'bic': bic, 'hqic': hqic,
                        'llf': llf,
                        'n_significant_params': significant_params,
                        'n_params': len(params),
                        'convergence': fitted_model.mle_retvals['converged'] if hasattr(fitted_model, 'mle_retvals') else True
                    })
                    
                except Exception as e:
                    # Model failed to converge or other issues
                    continue
        
        if not model_results:
            raise ValueError("No models could be fitted successfully")
        
        # Convert to DataFrame for easier analysis
        results_df = pd.DataFrame(model_results)
        
        # Find best models by each criterion
        best_aic = results_df.loc[results_df['aic'].idxmin()]
        best_bic = results_df.loc[results_df['bic'].idxmin()]
        best_hqic = results_df.loc[results_df['hqic'].idxmin()]
        
        print(f"   • Best AIC: ARIMA({int(best_aic['p'])},{d},{int(best_aic['q'])}) - AIC={best_aic['aic']:.2f}")
        print(f"   • Best BIC: ARIMA({int(best_bic['p'])},{d},{int(best_bic['q'])}) - BIC={best_bic['bic']:.2f}")
        print(f"   • Best HQIC: ARIMA({int(best_hqic['p'])},{d},{int(best_hqic['q'])}) - HQIC={best_hqic['hqic']:.2f}")
        
        # Recommend final model (prefer BIC for its parsimony penalty)
        recommended_model = (int(best_bic['p']), d, int(best_bic['q']))
        
        print(f"\n3. Recommended Model: ARIMA{recommended_model}")
        
        identification_results = {
            'acf_values': acf_values,
            'pacf_values': pacf_values,
            'acf_cutoff': acf_cutoff,
            'pacf_cutoff': pacf_cutoff,
            'suggested_pattern': suggested_pattern,
            'model_comparison': results_df,
            'best_models': {
                'aic': (int(best_aic['p']), d, int(best_aic['q'])),
                'bic': (int(best_bic['p']), d, int(best_bic['q'])),
                'hqic': (int(best_hqic['p']), d, int(best_hqic['q']))
            },
            'recommended_order': recommended_model
        }
        
        return identification_results
    
    def fit_arima_model(self, data, order, seasonal_order=None):
        """
        Fit ARIMA model with comprehensive analysis
        
        Parameters:
        -----------
        data : array-like
            Time series data
        order : tuple
            ARIMA order (p, d, q)
        seasonal_order : tuple, optional
            Seasonal ARIMA order (P, D, Q, s)
        """
        
        print(f"\n=== Fitting ARIMA{order} Model ===")
        
        # Fit the model
        if seasonal_order is not None:
            print(f"Including seasonal component: {seasonal_order}")
            self.model = ARIMA(data, order=order, seasonal_order=seasonal_order)
        else:
            self.model = ARIMA(data, order=order)
        
        self.fitted_model = self.model.fit()
        
        # Model summary
        print(f"\n1. Model Summary:")
        print(f"   • Order: ARIMA{order}")
        print(f"   • AIC: {self.fitted_model.aic:.3f}")
        print(f"   • BIC: {self.fitted_model.bic:.3f}")
        print(f"   • Log Likelihood: {self.fitted_model.llf:.3f}")
        print(f"   • Observations: {self.fitted_model.nobs}")
        
        # Parameter estimates
        print(f"\n2. Parameter Estimates:")
        params = self.fitted_model.params
        pvalues = self.fitted_model.pvalues
        conf_int = self.fitted_model.conf_int()
        
        for i, (param_name, param_value) in enumerate(params.items()):
            p_val = pvalues.iloc[i]
            lower_ci = conf_int.iloc[i, 0]
            upper_ci = conf_int.iloc[i, 1]
            significant = p_val < 0.05
            
            print(f"   • {param_name}: {param_value:.4f} [{lower_ci:.4f}, {upper_ci:.4f}] " +
                  f"(p={p_val:.4f}) {'*' if significant else ''}")
        
        # Model diagnostics
        self.run_model_diagnostics(data)
        
        return self.fitted_model
    
    def run_model_diagnostics(self, original_data):
        """
        Comprehensive model diagnostics and residual analysis
        
        Parameters:
        -----------
        original_data : array-like
            Original time series data
        """
        
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        
        print(f"\n=== Model Diagnostics ===")
        
        # Extract residuals
        residuals = self.fitted_model.resid
        standardized_residuals = residuals / np.std(residuals)
        
        # 1. Residual statistics
        print(f"\n1. Residual Statistics:")
        print(f"   • Mean: {np.mean(residuals):.6f}")
        print(f"   • Std: {np.std(residuals):.4f}")
        print(f"   • Skewness: {stats.skew(residuals):.4f}")
        print(f"   • Kurtosis: {stats.kurtosis(residuals):.4f}")
        
        # 2. Ljung-Box test for residual autocorrelation
        print(f"\n2. Ljung-Box Test for Residual Autocorrelation:")
        
        max_lag = min(10, len(residuals)//5)
        lb_test = acorr_ljungbox(residuals, lags=max_lag, return_df=True)
        
        for lag in [1, 5, 10]:
            if lag <= max_lag and lag-1 < len(lb_test):
                lb_stat = lb_test.iloc[lag-1]['lb_stat']
                lb_pvalue = lb_test.iloc[lag-1]['lb_pvalue']
                print(f"   • Lag {lag}: LB={lb_stat:.3f}, p-value={lb_pvalue:.4f} " +
                      f"({'No autocorr' if lb_pvalue > 0.05 else 'Autocorr detected'})")
        
        # 3. Normality tests
        print(f"\n3. Normality Tests:")
        
        # Jarque-Bera test
        jb_stat, jb_pvalue = stats.jarque_bera(residuals)
        print(f"   • Jarque-Bera: {jb_stat:.3f}, p-value={jb_pvalue:.4f} " +
              f"({'Normal' if jb_pvalue > 0.05 else 'Non-normal'})")
        
        # Shapiro-Wilk test (for smaller samples)
        if len(residuals) <= 5000:
            sw_stat, sw_pvalue = stats.shapiro(residuals)
            print(f"   • Shapiro-Wilk: {sw_stat:.3f}, p-value={sw_pvalue:.4f} " +
                  f"({'Normal' if sw_pvalue > 0.05 else 'Non-normal'})")
        
        # 4. Heteroscedasticity tests
        print(f"\n4. Heteroscedasticity Analysis:")
        
        # Simple test: correlation between squared residuals and fitted values
        fitted_values = self.fitted_model.fittedvalues
        squared_residuals = residuals**2
        
        if len(fitted_values) == len(squared_residuals):
            het_corr = np.corrcoef(fitted_values, squared_residuals)[0, 1]
            print(f"   • Correlation(fitted, residuals²): {het_corr:.4f}")
            
            # If correlation is significant, suggests heteroscedasticity
            het_test_stat = het_corr * np.sqrt(len(residuals) - 2) / np.sqrt(1 - het_corr**2)
            het_critical = stats.t.ppf(0.975, len(residuals) - 2)
            
            print(f"   • Heteroscedasticity test: {'Detected' if abs(het_test_stat) > het_critical else 'Not detected'}")
        
        # Store diagnostic results
        self.diagnostics = {
            'residuals': residuals,
            'standardized_residuals': standardized_residuals,
            'ljung_box_test': lb_test,
            'jarque_bera': {'statistic': jb_stat, 'p_value': jb_pvalue},
            'residual_stats': {
                'mean': np.mean(residuals),
                'std': np.std(residuals),
                'skewness': stats.skew(residuals),
                'kurtosis': stats.kurtosis(residuals)
            }
        }
        
        return self.diagnostics
    
    def generate_forecasts(self, steps=10, alpha=0.05):
        """
        Generate forecasts with confidence intervals
        
        Parameters:
        -----------
        steps : int, default=10
            Number of steps ahead to forecast
        alpha : float, default=0.05
            Significance level for confidence intervals
        """
        
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        
        print(f"\n=== Generating {steps}-Step Ahead Forecasts ===")
        
        # Generate forecasts
        forecast_result = self.fitted_model.get_forecast(steps=steps, alpha=alpha)
        
        # Extract forecasts and confidence intervals
        forecasts = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int()
        forecast_se = forecast_result.se_mean
        
        print(f"\nForecast Results:")
        print(f"{'Step':<6} {'Forecast':<12} {'Std Error':<12} {'Lower CI':<12} {'Upper CI':<12}")
        print("-" * 60)
        
        for i in range(steps):
            step = i + 1
            fc = forecasts.iloc[i]
            se = forecast_se.iloc[i]
            lower = conf_int.iloc[i, 0]
            upper = conf_int.iloc[i, 1]
            
            print(f"{step:<6} {fc:<12.3f} {se:<12.3f} {lower:<12.3f} {upper:<12.3f}")
        
        # Calculate forecast accuracy metrics (in-sample)
        fitted_values = self.fitted_model.fittedvalues
        original_data = self.model.endog
        
        if len(fitted_values) == len(original_data):
            mse = mean_squared_error(original_data, fitted_values)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(original_data, fitted_values)
            mape = np.mean(np.abs((original_data - fitted_values) / original_data)) * 100
            
            print(f"\nIn-Sample Accuracy Metrics:")
            print(f"   • MSE: {mse:.4f}")
            print(f"   • RMSE: {rmse:.4f}")
            print(f"   • MAE: {mae:.4f}")
            print(f"   • MAPE: {mape:.2f}%")
        
        forecast_results = {
            'forecasts': forecasts,
            'confidence_intervals': conf_int,
            'standard_errors': forecast_se,
            'alpha': alpha
        }
        
        return forecast_results
    
    def plot_comprehensive_analysis(self, data, forecast_results=None):
        """
        Create comprehensive visualization of ARIMA analysis
        
        Parameters:
        -----------
        data : array-like
            Original time series data
        forecast_results : dict, optional
            Results from generate_forecasts method
        """
        
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        
        fig = plt.figure(figsize=(16, 12))
        
        # Create time index
        time_index = np.arange(len(data))
        
        # Plot 1: Original vs Fitted
        plt.subplot(3, 2, 1)
        plt.plot(time_index, data, 'b-', label='Original', linewidth=1.5)
        
        fitted_values = self.fitted_model.fittedvalues
        if len(fitted_values) == len(data):
            plt.plot(time_index, fitted_values, 'r--', label='Fitted', linewidth=1.5)
        
        # Add forecasts if provided
        if forecast_results is not None:
            forecast_index = np.arange(len(data), len(data) + len(forecast_results['forecasts']))
            forecasts = forecast_results['forecasts']
            conf_int = forecast_results['confidence_intervals']
            
            plt.plot(forecast_index, forecasts, 'g-', label='Forecast', linewidth=2)
            plt.fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], 
                           alpha=0.3, color='green', label='Confidence Interval')
        
        plt.title(f'ARIMA{self.model.order} - Original vs Fitted vs Forecast')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Residuals
        plt.subplot(3, 2, 2)
        residuals = self.fitted_model.resid
        plt.plot(residuals, 'ko-', markersize=3)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        plt.title('Residuals')
        plt.xlabel('Time')
        plt.ylabel('Residual')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Residual ACF
        plt.subplot(3, 2, 3)
        max_lags = min(20, len(residuals)//4)
        acf_res = acf(residuals, nlags=max_lags, alpha=0.05)
        
        lags = np.arange(max_lags + 1)
        plt.stem(lags, acf_res[0], linefmt='b-', markerfmt='bo')
        
        # Add confidence bounds
        conf_bound = 1.96 / np.sqrt(len(residuals))
        plt.axhline(y=conf_bound, color='r', linestyle='--', alpha=0.7)
        plt.axhline(y=-conf_bound, color='r', linestyle='--', alpha=0.7)
        
        plt.title('Residual ACF')
        plt.xlabel('Lag')
        plt.ylabel('ACF')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Q-Q plot for normality
        plt.subplot(3, 2, 4)
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q Plot (Normality Check)')
        plt.grid(True, alpha=0.3)
        
        # Plot 5: Residual histogram
        plt.subplot(3, 2, 5)
        plt.hist(residuals, bins=20, density=True, alpha=0.7, color='skyblue')
        
        # Overlay normal distribution
        mu, sigma = np.mean(residuals), np.std(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal PDF')
        
        plt.title('Residual Distribution')
        plt.xlabel('Residual Value')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 6: Forecast error evolution (if forecasts available)
        plt.subplot(3, 2, 6)
        if forecast_results is not None:
            se = forecast_results['standard_errors']
            steps = np.arange(1, len(se) + 1)
            plt.plot(steps, se, 'ro-', linewidth=2, markersize=5)
            plt.title('Forecast Standard Error Evolution')
            plt.xlabel('Forecast Horizon')
            plt.ylabel('Standard Error')
            plt.grid(True, alpha=0.3)
        else:
            # Plot fitted vs residuals
            if len(fitted_values) == len(residuals):
                plt.scatter(fitted_values, residuals, alpha=0.6)
                plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
                plt.title('Fitted vs Residuals')
                plt.xlabel('Fitted Values')
                plt.ylabel('Residuals')
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig

def demonstrate_arima_modeling():
    """Comprehensive demonstration of ARIMA modeling process"""
    
    print("=== ARIMA Modeling Demonstration ===")
    
    # Generate synthetic non-stationary time series
    np.random.seed(42)
    n = 200
    
    # Generate ARIMA(1,1,1) process
    # Start with AR(1) process, then integrate, then add MA(1) component
    phi = 0.7  # AR coefficient
    theta = 0.3  # MA coefficient
    
    # Generate white noise
    epsilon = np.random.normal(0, 1, n + 100)  # Extra for burn-in
    
    # Generate ARMA(1,1) process
    arma_process = np.zeros(n + 100)
    for t in range(1, n + 100):
        arma_process[t] = phi * arma_process[t-1] + epsilon[t] + theta * epsilon[t-1]
    
    # Take cumulative sum to create I(1) process
    integrated_process = np.cumsum(arma_process[100:])  # Remove burn-in
    
    # Add trend and level
    trend = 0.05 * np.arange(n)
    data = 50 + trend + integrated_process
    
    print(f"\nGenerated ARIMA(1,1,1) data:")
    print(f"  • Length: {n}")
    print(f"  • True parameters: φ={phi}, θ={theta}")
    print(f"  • Mean: {np.mean(data):.2f}")
    print(f"  • Std: {np.std(data):.2f}")
    
    # Initialize ARIMA analyzer
    analyzer = ARIMAAnalysis()
    
    # Step 1: Test for stationarity
    stationarity_results = analyzer.comprehensive_stationarity_testing(data)
    
    # Step 2: Determine differencing order
    optimal_d, differencing_results = analyzer.determine_differencing_order(data)
    
    # Step 3: Identify ARIMA orders
    identification_results = analyzer.identify_arima_orders(data, d=optimal_d)
    
    recommended_order = identification_results['recommended_order']
    
    # Step 4: Fit ARIMA model
    fitted_model = analyzer.fit_arima_model(data, recommended_order)
    
    # Step 5: Generate forecasts
    forecast_results = analyzer.generate_forecasts(steps=20)
    
    # Step 6: Create comprehensive plots
    analyzer.plot_comprehensive_analysis(data, forecast_results)
    
    # Step 7: Model comparison
    print(f"\n=== Model Validation ===")
    print(f"True model: ARIMA(1,1,1)")
    print(f"Identified model: ARIMA{recommended_order}")
    
    # Compare with true parameters
    true_params = {'ar.L1': phi, 'ma.L1': theta}
    estimated_params = fitted_model.params
    
    print(f"\nParameter Comparison:")
    for param_name in ['ar.L1', 'ma.L1']:
        if param_name in estimated_params.index:
            true_val = true_params.get(param_name, 0)
            est_val = estimated_params[param_name]
            error = abs(est_val - true_val)
            print(f"  • {param_name}: True={true_val:.3f}, Estimated={est_val:.3f}, Error={error:.3f}")
    
    return analyzer, fitted_model, forecast_results

# Execute demonstration
if __name__ == "__main__":
    analyzer, model, forecasts = demonstrate_arima_modeling()
```

**Key Theoretical Insights:**

1. **Box-Jenkins Methodology**: Systematic approach through identification, estimation, diagnostic checking, and forecasting phases

2. **Stationarity Requirements**: Combined ADF and KPSS testing provides robust stationarity assessment with different null hypotheses

3. **Order Identification**: ACF/PACF patterns combined with information criteria provide multiple perspectives on model selection

4. **Maximum Likelihood Estimation**: Theoretical foundation for parameter estimation with asymptotic properties

5. **Diagnostic Checking**: Comprehensive residual analysis ensures model adequacy and identifies violations of assumptions

6. **Forecasting Theory**: Minimum MSE predictors with theoretically derived confidence intervals

7. **Model Selection**: Information criteria balance goodness-of-fit with model parsimony

8. **Practical Implementation**: Real-world considerations including convergence, significance testing, and robustness

**Applications:**
- **Economic Forecasting**: GDP, inflation, unemployment predictions
- **Financial Markets**: Asset price modeling, volatility forecasting
- **Operations**: Demand forecasting, inventory management
- **Quality Control**: Process monitoring, defect prediction

---

## Question 4

**Fit aGARCH modelto afinancial time series datasetand interpret the results.**

**Answer:**

**Theoretical Foundation:**

GARCH (Generalized Autoregressive Conditional Heteroscedasticity) models represent a sophisticated framework for modeling **time-varying volatility** in financial time series. Developed by **Tim Bollerslev (1986)** as an extension of **Engle's ARCH model (1982)**, GARCH theory addresses the **heteroscedasticity** and **volatility clustering** commonly observed in financial markets.

**Mathematical Framework:**

**GARCH(p,q) Model Structure:**
```
Return Equation: r_t = μ + ε_t
Variance Equation: σ_t² = ω + Σᵢ₌₁ᵖ αᵢε²_{t-i} + Σⱼ₌₁ᵠ βⱼσ²_{t-j}

Where:
- r_t: Return at time t
- ε_t: Error term, ε_t = σ_t × z_t
- z_t: Standardized residuals (i.i.d., E[z_t]=0, Var(z_t)=1)
- σ_t²: Conditional variance at time t
- ω > 0: Constant term
- αᵢ ≥ 0: ARCH coefficients (i=1,...,p)
- βⱼ ≥ 0: GARCH coefficients (j=1,...,q)
```

**Stationarity Conditions:**
```
Σᵢ₌₁ᵖ αᵢ + Σⱼ₌₁ᵠ βⱼ < 1  (Covariance stationarity)

For GARCH(1,1): α + β < 1
```

**Comprehensive Implementation:**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
from arch.unitroot import ADF, KPSS
from scipy import stats
from sklearn.metrics import mean_squared_error
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class GARCHAnalysis:
    """
    Comprehensive GARCH modeling framework for financial time series
    with theoretical foundations and advanced diagnostics
    """
    
    def __init__(self):
        self.data = None
        self.returns = None
        self.garch_model = None
        self.garch_results = None
        self.diagnostics = {}
        
    def theoretical_foundations(self):
        """Theoretical foundations of GARCH modeling"""
        
        foundations = {
            'volatility_stylized_facts': {
                'volatility_clustering': 'High volatility periods followed by high volatility',
                'fat_tails': 'Return distributions have heavier tails than normal',
                'leverage_effect': 'Negative returns associated with higher volatility',
                'mean_reversion': 'Volatility tends to revert to long-run average',
                'non_linear_dependence': 'Returns uncorrelated but not independent'
            },
            'garch_properties': {
                'conditional_heteroscedasticity': 'Var(ε_t|I_{t-1}) = σ_t²',
                'unconditional_homoscedasticity': 'Var(ε_t) = constant',
                'fat_tailed_distribution': 'Unconditional distribution has excess kurtosis',
                'volatility_persistence': 'Measured by α + β',
                'long_run_variance': 'σ² = ω/(1 - α - β)'
            },
            'estimation_theory': {
                'maximum_likelihood': 'L(θ) = ∏ f(r_t|I_{t-1};θ)',
                'log_likelihood': 'ℓ(θ) = Σ log f(r_t|I_{t-1};θ)',
                'conditional_density': 'f(r_t|I_{t-1}) = (1/σ_t)f(ε_t/σ_t)',
                'numerical_optimization': 'BFGS, Newton-Raphson methods',
                'asymptotic_properties': 'Consistency, efficiency, normality'
            },
            'extensions_variants': {
                'egarch': 'Exponential GARCH for leverage effects',
                'tgarch': 'Threshold GARCH for asymmetric volatility',
                'gjr_garch': 'Glosten-Jagannathan-Runkle GARCH',
                'igarch': 'Integrated GARCH (α + β = 1)',
                'figarch': 'Fractionally integrated GARCH'
            },
            'applications': {
                'risk_management': 'VaR calculation, portfolio optimization',
                'option_pricing': 'Volatility inputs for Black-Scholes variants',
                'volatility_forecasting': 'Multi-step ahead volatility predictions',
                'market_microstructure': 'High-frequency volatility modeling',
                'regulatory_capital': 'Basel III market risk calculations'
            }
        }
        
        return foundations
    
    def load_financial_data(self, symbol='SPY', start_date='2020-01-01', end_date=None):
        """
        Load financial data and compute returns
        
        Parameters:
        -----------
        symbol : str, default='SPY'
            Financial instrument symbol
        start_date : str, default='2020-01-01'
            Start date for data
        end_date : str, optional
            End date for data (defaults to today)
        """
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"=== Loading Financial Data: {symbol} ===")
        print(f"Period: {start_date} to {end_date}")
        
        try:
            # Download data using yfinance
            ticker = yf.Ticker(symbol)
            self.data = ticker.history(start=start_date, end=end_date)
            
            if self.data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Calculate returns
            self.returns = 100 * self.data['Close'].pct_change().dropna()  # Percentage returns
            
            print(f"Data loaded successfully:")
            print(f"  • Observations: {len(self.data)}")
            print(f"  • Return observations: {len(self.returns)}")
            print(f"  • Date range: {self.data.index[0].strftime('%Y-%m-%d')} to {self.data.index[-1].strftime('%Y-%m-%d')}")
            
            # Basic return statistics
            self._basic_return_analysis()
            
        except Exception as e:
            print(f"Error loading data: {e}")
            # Generate synthetic data as fallback
            self._generate_synthetic_data()
    
    def _generate_synthetic_data(self, n=1000):
        """Generate synthetic financial returns with GARCH properties"""
        
        print(f"\nGenerating synthetic financial data ({n} observations)")
        
        np.random.seed(42)
        
        # GARCH(1,1) parameters
        omega = 0.1
        alpha = 0.1
        beta = 0.85
        
        # Initialize
        returns = np.zeros(n)
        sigma_sq = np.zeros(n)
        sigma_sq[0] = omega / (1 - alpha - beta)  # Unconditional variance
        
        # Generate GARCH process
        for t in range(1, n):
            # Update conditional variance
            sigma_sq[t] = omega + alpha * returns[t-1]**2 + beta * sigma_sq[t-1]
            
            # Generate return
            returns[t] = np.sqrt(sigma_sq[t]) * np.random.normal()
        
        # Create DataFrame
        dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
        self.data = pd.DataFrame({
            'Close': 100 * np.exp(np.cumsum(returns/100)),
            'Volume': np.random.randint(1000000, 10000000, n)
        }, index=dates)
        
        self.returns = pd.Series(returns, index=dates)
        
        print(f"Synthetic data generated:")
        print(f"  • True GARCH(1,1) parameters: ω={omega}, α={alpha}, β={beta}")
        print(f"  • Persistence: α+β = {alpha+beta:.3f}")
        
    def _basic_return_analysis(self):
        """Basic statistical analysis of returns"""
        
        print(f"\n=== Basic Return Analysis ===")
        
        # Descriptive statistics
        returns_stats = {
            'mean': self.returns.mean(),
            'std': self.returns.std(),
            'skewness': self.returns.skew(),
            'kurtosis': self.returns.kurtosis(),
            'min': self.returns.min(),
            'max': self.returns.max(),
            'jarque_bera': stats.jarque_bera(self.returns)
        }
        
        print(f"Descriptive Statistics:")
        print(f"  • Mean: {returns_stats['mean']:.4f}%")
        print(f"  • Std Dev: {returns_stats['std']:.4f}%")
        print(f"  • Skewness: {returns_stats['skewness']:.4f}")
        print(f"  • Excess Kurtosis: {returns_stats['kurtosis']:.4f}")
        print(f"  • Min: {returns_stats['min']:.4f}%")
        print(f"  • Max: {returns_stats['max']:.4f}%")
        
        # Jarque-Bera test for normality
        jb_stat, jb_pvalue = returns_stats['jarque_bera']
        print(f"  • Jarque-Bera: {jb_stat:.3f} (p-value: {jb_pvalue:.4f})")
        print(f"  • Distribution: {'Normal' if jb_pvalue > 0.05 else 'Non-normal'}")
        
        # Test for ARCH effects
        self._test_arch_effects()
        
        return returns_stats
    
    def _test_arch_effects(self):
        """Test for ARCH effects in return series"""
        
        print(f"\n=== ARCH Effects Testing ===")
        
        # Ljung-Box test on squared returns
        from statsmodels.stats.diagnostic import acorr_ljungbox
        
        squared_returns = self.returns**2
        
        # Test at different lags
        for lag in [5, 10, 20]:
            if lag < len(squared_returns):
                lb_result = acorr_ljungbox(squared_returns, lags=lag, return_df=True)
                lb_stat = lb_result['lb_stat'].iloc[-1]
                lb_pvalue = lb_result['lb_pvalue'].iloc[-1]
                
                print(f"  • Ljung-Box (lag {lag}): {lb_stat:.3f} (p-value: {lb_pvalue:.4f})")
                print(f"    {'ARCH effects detected' if lb_pvalue < 0.05 else 'No ARCH effects'}")
        
        # ARCH-LM test using arch package
        try:
            from arch.unitroot import DFGLS
            from arch.utility.array import ensure1d
            
            # Simple ARCH-LM test
            returns_array = ensure1d(self.returns.values, 'returns')
            
            # Manual ARCH-LM test
            # Regress squared returns on lagged squared returns
            from sklearn.linear_model import LinearRegression
            
            lags = 5
            if len(returns_array) > lags:
                y = squared_returns.iloc[lags:].values
                X = np.column_stack([squared_returns.shift(i+1).iloc[lags:].values 
                                   for i in range(lags)])
                
                # Remove any NaN values
                valid_mask = ~(np.isnan(y) | np.any(np.isnan(X), axis=1))
                y_clean = y[valid_mask]
                X_clean = X[valid_mask]
                
                if len(y_clean) > 0:
                    reg = LinearRegression().fit(X_clean, y_clean)
                    r_squared = reg.score(X_clean, y_clean)
                    
                    # LM statistic
                    n = len(y_clean)
                    lm_stat = n * r_squared
                    
                    # Chi-square critical value
                    critical_value = stats.chi2.ppf(0.95, lags)
                    
                    print(f"  • ARCH-LM test: LM={lm_stat:.3f}, Critical={critical_value:.3f}")
                    print(f"    {'ARCH effects detected' if lm_stat > critical_value else 'No ARCH effects'}")
                    
        except Exception as e:
            print(f"  • ARCH-LM test failed: {e}")
    
    def identify_garch_order(self, max_p=3, max_q=3):
        """
        Identify optimal GARCH order using information criteria
        
        Parameters:
        -----------
        max_p : int, default=3
            Maximum ARCH order
        max_q : int, default=3
            Maximum GARCH order
        """
        
        print(f"\n=== GARCH Order Identification ===")
        print(f"Testing orders up to GARCH({max_p},{max_q})")
        
        results = []
        
        for p in range(1, max_p + 1):
            for q in range(1, max_q + 1):
                try:
                    # Fit GARCH model
                    model = arch_model(self.returns, vol='Garch', p=p, q=q, rescale=False)
                    fitted_model = model.fit(disp='off', show_warning=False)
                    
                    # Extract information criteria
                    results.append({
                        'p': p, 'q': q,
                        'aic': fitted_model.aic,
                        'bic': fitted_model.bic,
                        'hqic': fitted_model.hqic,
                        'llf': fitted_model.loglikelihood,
                        'converged': fitted_model.convergence_flag == 0
                    })
                    
                except Exception as e:
                    print(f"    GARCH({p},{q}) failed: {e}")
                    continue
        
        if not results:
            raise ValueError("No GARCH models could be fitted")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Filter converged models
        converged_models = results_df[results_df['converged']]
        
        if converged_models.empty:
            print("Warning: No models converged properly")
            converged_models = results_df
        
        # Find best models
        best_aic = converged_models.loc[converged_models['aic'].idxmin()]
        best_bic = converged_models.loc[converged_models['bic'].idxmin()]
        best_hqic = converged_models.loc[converged_models['hqic'].idxmin()]
        
        print(f"\nModel Comparison Results:")
        print(f"  • Best AIC: GARCH({int(best_aic['p'])},{int(best_aic['q'])}) - AIC={best_aic['aic']:.3f}")
        print(f"  • Best BIC: GARCH({int(best_bic['p'])},{int(best_bic['q'])}) - BIC={best_bic['bic']:.3f}")
        print(f"  • Best HQIC: GARCH({int(best_hqic['p'])},{int(best_hqic['q'])}) - HQIC={best_hqic['hqic']:.3f}")
        
        # Recommend model (prefer BIC for parsimony)
        recommended_order = (int(best_bic['p']), int(best_bic['q']))
        print(f"\nRecommended model: GARCH{recommended_order}")
        
        return recommended_order, results_df
    
    def fit_garch_model(self, p=1, q=1, distribution='normal', mean='constant'):
        """
        Fit GARCH model with comprehensive analysis
        
        Parameters:
        -----------
        p : int, default=1
            ARCH order
        q : int, default=1
            GARCH order
        distribution : str, default='normal'
            Error distribution ('normal', 't', 'skewt', 'ged')
        mean : str, default='constant'
            Mean model ('constant', 'zero', 'ar')
        """
        
        print(f"\n=== Fitting GARCH({p},{q}) Model ===")
        print(f"Distribution: {distribution}, Mean model: {mean}")
        
        # Create and fit GARCH model
        self.garch_model = arch_model(
            self.returns, 
            vol='Garch', 
            p=p, 
            q=q, 
            mean=mean,
            dist=distribution,
            rescale=False
        )
        
        # Fit the model
        self.garch_results = self.garch_model.fit(disp='off', show_warning=False)
        
        # Display results
        print(f"\n1. Model Summary:")
        print(f"   • Converged: {'Yes' if self.garch_results.convergence_flag == 0 else 'No'}")
        print(f"   • Log Likelihood: {self.garch_results.loglikelihood:.3f}")
        print(f"   • AIC: {self.garch_results.aic:.3f}")
        print(f"   • BIC: {self.garch_results.bic:.3f}")
        
        # Parameter estimates
        print(f"\n2. Parameter Estimates:")
        params = self.garch_results.params
        pvalues = self.garch_results.pvalues
        conf_int = self.garch_results.conf_int()
        
        for param_name in params.index:
            param_value = params[param_name]
            p_value = pvalues[param_name]
            lower_ci = conf_int.loc[param_name, 'lower']
            upper_ci = conf_int.loc[param_name, 'upper']
            significant = p_value < 0.05
            
            print(f"   • {param_name}: {param_value:.6f} [{lower_ci:.6f}, {upper_ci:.6f}] " +
                  f"(p={p_value:.4f}) {'*' if significant else ''}")
        
        # Calculate persistence and long-run variance
        self._analyze_garch_properties()
        
        # Run diagnostics
        self._garch_model_diagnostics()
        
        return self.garch_results
    
    def _analyze_garch_properties(self):
        """Analyze GARCH model properties"""
        
        print(f"\n3. GARCH Properties Analysis:")
        
        params = self.garch_results.params
        
        # Extract GARCH parameters
        omega = params.get('omega', 0)
        
        # Calculate persistence
        alpha_sum = sum(params[param] for param in params.index if param.startswith('alpha'))
        beta_sum = sum(params[param] for param in params.index if param.startswith('beta'))
        persistence = alpha_sum + beta_sum
        
        print(f"   • Persistence (α + β): {persistence:.6f}")
        
        if persistence < 1:
            # Long-run variance
            long_run_variance = omega / (1 - persistence)
            long_run_volatility = np.sqrt(long_run_variance)
            
            print(f"   • Long-run variance: {long_run_variance:.6f}")
            print(f"   • Long-run volatility: {long_run_volatility:.4f}%")
            print(f"   • Model: {'Stationary' if persistence < 1 else 'Non-stationary'}")
            
            # Half-life of volatility shocks
            if persistence > 0:
                half_life = np.log(0.5) / np.log(persistence)
                print(f"   • Half-life of shocks: {half_life:.2f} periods")
        else:
            print(f"   • Model: Integrated GARCH (IGARCH)")
            print(f"   • Long-run variance: Undefined (unit root in volatility)")
        
        # Unconditional moments
        if hasattr(self.garch_results, 'conditional_volatility'):
            cond_vol = self.garch_results.conditional_volatility
            print(f"   • Average conditional volatility: {cond_vol.mean():.4f}%")
            print(f"   • Max conditional volatility: {cond_vol.max():.4f}%")
            print(f"   • Min conditional volatility: {cond_vol.min():.4f}%")
    
    def _garch_model_diagnostics(self):
        """Comprehensive GARCH model diagnostics"""
        
        print(f"\n=== GARCH Model Diagnostics ===")
        
        # Extract standardized residuals
        std_residuals = self.garch_results.std_resid
        
        # 1. Standardized residual tests
        print(f"\n1. Standardized Residual Analysis:")
        print(f"   • Mean: {std_residuals.mean():.6f}")
        print(f"   • Std: {std_residuals.std():.6f}")
        print(f"   • Skewness: {std_residuals.skew():.4f}")
        print(f"   • Excess Kurtosis: {std_residuals.kurtosis():.4f}")
        
        # Jarque-Bera test
        jb_stat, jb_pvalue = stats.jarque_bera(std_residuals)
        print(f"   • Jarque-Bera: {jb_stat:.3f} (p-value: {jb_pvalue:.4f})")
        print(f"   • Distribution: {'Normal' if jb_pvalue > 0.05 else 'Non-normal'}")
        
        # 2. Test for remaining ARCH effects
        print(f"\n2. Remaining ARCH Effects:")
        
        from statsmodels.stats.diagnostic import acorr_ljungbox
        
        squared_std_residuals = std_residuals**2
        
        for lag in [5, 10]:
            if lag < len(squared_std_residuals):
                lb_result = acorr_ljungbox(squared_std_residuals, lags=lag, return_df=True)
                lb_stat = lb_result['lb_stat'].iloc[-1]
                lb_pvalue = lb_result['lb_pvalue'].iloc[-1]
                
                print(f"   • Ljung-Box (lag {lag}): {lb_stat:.3f} (p-value: {lb_pvalue:.4f})")
                print(f"     {'ARCH effects remain' if lb_pvalue < 0.05 else 'No remaining ARCH effects'}")
        
        # 3. Sign bias test
        print(f"\n3. Sign Bias Test:")
        self._sign_bias_test(std_residuals)
        
        # Store diagnostic results
        self.diagnostics = {
            'std_residuals': std_residuals,
            'jarque_bera': {'statistic': jb_stat, 'p_value': jb_pvalue},
            'ljung_box_squared': lb_result if 'lb_result' in locals() else None
        }
    
    def _sign_bias_test(self, std_residuals):
        """Test for sign bias (leverage effects not captured)"""
        
        try:
            # Create sign variables
            negative_returns = (self.returns.iloc[1:] < 0).astype(int)
            
            # Align with standardized residuals
            if len(negative_returns) != len(std_residuals):
                min_len = min(len(negative_returns), len(std_residuals))
                negative_returns = negative_returns.iloc[:min_len]
                std_residuals_aligned = std_residuals.iloc[:min_len]
            else:
                std_residuals_aligned = std_residuals
            
            # Regression: std_resid² = α + β × I(negative return) + error
            from sklearn.linear_model import LinearRegression
            
            X = negative_returns.values.reshape(-1, 1)
            y = std_residuals_aligned**2
            
            reg = LinearRegression().fit(X, y)
            
            # Calculate test statistic
            y_pred = reg.predict(X)
            residuals = y - y_pred
            mse = np.mean(residuals**2)
            
            # Simple t-test for significance of coefficient
            coef = reg.coef_[0]
            
            print(f"   • Sign bias coefficient: {coef:.6f}")
            print(f"   • Interpretation: {'Leverage effects detected' if abs(coef) > 0.01 else 'No significant bias'}")
            
        except Exception as e:
            print(f"   • Sign bias test failed: {e}")
    
    def forecast_volatility(self, horizon=10):
        """
        Generate volatility forecasts
        
        Parameters:
        -----------
        horizon : int, default=10
            Forecast horizon
        """
        
        if self.garch_results is None:
            raise ValueError("GARCH model must be fitted first")
        
        print(f"\n=== Volatility Forecasting ({horizon} periods) ===")
        
        # Generate forecasts
        forecasts = self.garch_results.forecast(horizon=horizon, method='simulation')
        
        # Extract variance forecasts
        variance_forecasts = forecasts.variance.iloc[-1, :]
        volatility_forecasts = np.sqrt(variance_forecasts)
        
        print(f"\nVolatility Forecasts:")
        print(f"{'Period':<8} {'Volatility (%)':<15} {'Variance':<15}")
        print("-" * 40)
        
        for i in range(horizon):
            period = i + 1
            vol = volatility_forecasts.iloc[i]
            var = variance_forecasts.iloc[i]
            
            print(f"{period:<8} {vol:<15.4f} {var:<15.6f}")
        
        # Calculate forecast statistics
        mean_forecast_vol = volatility_forecasts.mean()
        current_vol = self.garch_results.conditional_volatility.iloc[-1]
        
        print(f"\nForecast Summary:")
        print(f"   • Current volatility: {current_vol:.4f}%")
        print(f"   • Mean forecast volatility: {mean_forecast_vol:.4f}%")
        print(f"   • Long-run volatility: {self._get_long_run_volatility():.4f}%")
        
        return volatility_forecasts, variance_forecasts
    
    def _get_long_run_volatility(self):
        """Calculate long-run volatility from model parameters"""
        
        params = self.garch_results.params
        omega = params.get('omega', 0)
        
        alpha_sum = sum(params[param] for param in params.index if param.startswith('alpha'))
        beta_sum = sum(params[param] for param in params.index if param.startswith('beta'))
        persistence = alpha_sum + beta_sum
        
        if persistence < 1:
            long_run_variance = omega / (1 - persistence)
            return np.sqrt(long_run_variance)
        else:
            return np.nan
    
    def plot_comprehensive_analysis(self, volatility_forecasts=None):
        """Create comprehensive visualization of GARCH analysis"""
        
        if self.garch_results is None:
            raise ValueError("GARCH model must be fitted first")
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle('GARCH Model Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Returns time series
        axes[0, 0].plot(self.returns.index, self.returns, linewidth=0.8, alpha=0.8)
        axes[0, 0].set_title('Returns Time Series')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Returns (%)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Conditional volatility
        cond_vol = self.garch_results.conditional_volatility
        axes[0, 1].plot(cond_vol.index, cond_vol, 'r-', linewidth=1.5)
        
        # Add volatility forecasts if provided
        if volatility_forecasts is not None:
            forecast_dates = pd.date_range(start=cond_vol.index[-1] + pd.Timedelta(days=1), 
                                         periods=len(volatility_forecasts), freq='D')
            axes[0, 1].plot(forecast_dates, volatility_forecasts, 'g--', linewidth=2, 
                          label='Forecast')
            axes[0, 1].legend()
        
        axes[0, 1].set_title('Conditional Volatility')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Volatility (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Standardized residuals
        std_residuals = self.garch_results.std_resid
        axes[1, 0].plot(std_residuals.index, std_residuals, 'b-', linewidth=0.8, alpha=0.8)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.7)
        axes[1, 0].set_title('Standardized Residuals')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Std. Residuals')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Q-Q plot of standardized residuals
        stats.probplot(std_residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot (Std. Residuals)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 5: ACF of squared standardized residuals
        from statsmodels.tsa.stattools import acf
        
        squared_std_resid = std_residuals**2
        max_lags = min(20, len(squared_std_resid)//4)
        acf_values = acf(squared_std_resid, nlags=max_lags, alpha=0.05)
        
        lags = np.arange(max_lags + 1)
        axes[2, 0].stem(lags, acf_values[0], linefmt='b-', markerfmt='bo')
        
        # Confidence bounds
        conf_bound = 1.96 / np.sqrt(len(squared_std_resid))
        axes[2, 0].axhline(y=conf_bound, color='r', linestyle='--', alpha=0.7)
        axes[2, 0].axhline(y=-conf_bound, color='r', linestyle='--', alpha=0.7)
        
        axes[2, 0].set_title('ACF of Squared Std. Residuals')
        axes[2, 0].set_xlabel('Lag')
        axes[2, 0].set_ylabel('ACF')
        axes[2, 0].grid(True, alpha=0.3)
        
        # Plot 6: Histogram of standardized residuals
        axes[2, 1].hist(std_residuals, bins=30, density=True, alpha=0.7, color='skyblue')
        
        # Overlay normal distribution
        x = np.linspace(std_residuals.min(), std_residuals.max(), 100)
        axes[2, 1].plot(x, stats.norm.pdf(x, 0, 1), 'r-', linewidth=2, label='Standard Normal')
        
        axes[2, 1].set_title('Standardized Residuals Distribution')
        axes[2, 1].set_xlabel('Std. Residuals')
        axes[2, 1].set_ylabel('Density')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def calculate_var_es(self, confidence_levels=[0.95, 0.99], horizon=1):
        """
        Calculate Value at Risk (VaR) and Expected Shortfall (ES)
        
        Parameters:
        -----------
        confidence_levels : list, default=[0.95, 0.99]
            Confidence levels for VaR calculation
        horizon : int, default=1
            Time horizon for risk measures
        """
        
        if self.garch_results is None:
            raise ValueError("GARCH model must be fitted first")
        
        print(f"\n=== Risk Measures (VaR & Expected Shortfall) ===")
        print(f"Horizon: {horizon} period(s)")
        
        # Get current conditional volatility
        current_vol = self.garch_results.conditional_volatility.iloc[-1]
        
        # Adjust for horizon
        horizon_vol = current_vol * np.sqrt(horizon)
        
        # Get standardized residuals for distribution fitting
        std_residuals = self.garch_results.std_resid
        
        results = {}
        
        for confidence in confidence_levels:
            alpha = 1 - confidence
            
            # Parametric VaR (assuming normal distribution)
            var_normal = -stats.norm.ppf(alpha) * horizon_vol
            
            # Non-parametric VaR (historical simulation)
            var_historical = -np.percentile(std_residuals, alpha * 100) * horizon_vol
            
            # Expected Shortfall (Conditional VaR)
            # For normal distribution
            es_normal = horizon_vol * stats.norm.pdf(stats.norm.ppf(alpha)) / alpha
            
            # For historical distribution
            threshold = np.percentile(std_residuals, alpha * 100)
            tail_losses = std_residuals[std_residuals <= threshold]
            es_historical = -np.mean(tail_losses) * horizon_vol if len(tail_losses) > 0 else np.nan
            
            results[confidence] = {
                'var_normal': var_normal,
                'var_historical': var_historical,
                'es_normal': es_normal,
                'es_historical': es_historical
            }
            
            print(f"\nConfidence Level: {confidence*100}%")
            print(f"   • VaR (Normal): {var_normal:.4f}%")
            print(f"   • VaR (Historical): {var_historical:.4f}%")
            print(f"   • ES (Normal): {es_normal:.4f}%")
            print(f"   • ES (Historical): {es_historical:.4f}%")
        
        return results

def demonstrate_garch_modeling():
    """Comprehensive demonstration of GARCH modeling"""
    
    print("=== GARCH Modeling Demonstration ===")
    
    # Initialize GARCH analyzer
    garch_analyzer = GARCHAnalysis()
    
    # Load financial data
    garch_analyzer.load_financial_data('SPY', start_date='2020-01-01')
    
    # Identify optimal GARCH order
    recommended_order, model_comparison = garch_analyzer.identify_garch_order()
    
    # Fit GARCH model
    garch_results = garch_analyzer.fit_garch_model(
        p=recommended_order[0], 
        q=recommended_order[1],
        distribution='normal'
    )
    
    # Generate volatility forecasts
    vol_forecasts, var_forecasts = garch_analyzer.forecast_volatility(horizon=15)
    
    # Calculate risk measures
    risk_measures = garch_analyzer.calculate_var_es(confidence_levels=[0.95, 0.99])
    
    # Create comprehensive plots
    garch_analyzer.plot_comprehensive_analysis(vol_forecasts)
    
    # Summary interpretation
    print(f"\n=== Model Interpretation Summary ===")
    params = garch_results.params
    
    omega = params.get('omega', 0)
    alpha = params.get('alpha[1]', 0)
    beta = params.get('beta[1]', 0)
    persistence = alpha + beta
    
    print(f"\nGARCH({recommended_order[0]},{recommended_order[1]}) Results:")
    print(f"   • Persistence (α + β): {persistence:.4f}")
    print(f"   • Volatility clustering: {'Strong' if persistence > 0.9 else 'Moderate' if persistence > 0.7 else 'Weak'}")
    print(f"   • Mean reversion: {'Yes' if persistence < 1 else 'No (IGARCH)'}")
    
    if persistence < 1:
        long_run_vol = np.sqrt(omega / (1 - persistence))
        print(f"   • Long-run volatility: {long_run_vol:.4f}%")
        half_life = np.log(0.5) / np.log(persistence)
        print(f"   • Half-life of shocks: {half_life:.2f} periods")
    
    # Current market assessment
    current_vol = garch_results.conditional_volatility.iloc[-1]
    mean_vol = garch_results.conditional_volatility.mean()
    
    print(f"\nCurrent Market Assessment:")
    print(f"   • Current volatility: {current_vol:.4f}%")
    print(f"   • Average volatility: {mean_vol:.4f}%")
    print(f"   • Relative level: {'High' if current_vol > 1.2 * mean_vol else 'Low' if current_vol < 0.8 * mean_vol else 'Normal'}")
    
    return garch_analyzer, garch_results

# Execute demonstration
if __name__ == "__main__":
    analyzer, results = demonstrate_garch_modeling()
```

**Key Theoretical Insights:**

1. **Volatility Clustering**: GARCH captures the empirical fact that high volatility periods are followed by high volatility periods

2. **Conditional vs Unconditional Moments**: Models time-varying conditional variance while maintaining constant unconditional variance

3. **Persistence Measurement**: α + β quantifies volatility persistence; values close to 1 indicate long memory in volatility

4. **Mean Reversion**: When α + β < 1, volatility exhibits mean reversion to long-run level ω/(1-α-β)

5. **Fat Tails**: GARCH generates unconditional distributions with excess kurtosis even with normal innovations

6. **Risk Management Applications**: Provides dynamic volatility estimates for VaR, ES, and portfolio optimization

7. **Leverage Effects**: Extensions like EGARCH and GJR-GARCH capture asymmetric volatility responses

8. **Forecasting Properties**: Multi-step volatility forecasts converge to long-run unconditional volatility

**Financial Applications:**
- **Risk Management**: Dynamic VaR and Expected Shortfall calculations
- **Option Pricing**: Volatility inputs for derivatives valuation
- **Portfolio Optimization**: Time-varying covariance matrix estimation
- **Regulatory Capital**: Basel III market risk calculations
- **Trading Strategies**: Volatility-based position sizing and timing

---

## Question 5

**Create aPython scriptthat decomposes atime seriesintotrend,seasonality, andresidualsusingstatsmodels library.**

**Answer:**

**Theoretical Foundation:**

Time series decomposition is a fundamental technique that **separates a time series into constituent components**, enabling better understanding of underlying patterns and improved forecasting. The mathematical framework encompasses **classical decomposition**, **X-11/X-13 methods**, and **STL (Seasonal and Trend decomposition using Loess)** approaches.

**Mathematical Framework:**

**Additive Decomposition:**
```
Y_t = T_t + S_t + R_t

Where:
- Y_t: Observed value at time t
- T_t: Trend component (long-term movement)
- S_t: Seasonal component (regular periodic pattern)
- R_t: Residual/irregular component (random fluctuations)
```

**Multiplicative Decomposition:**
```
Y_t = T_t × S_t × R_t

Or equivalently (log-transformed):
log(Y_t) = log(T_t) + log(S_t) + log(R_t)
```

**STL Decomposition Framework:**
```
Y_t = T_t + S_t + R_t

Where components are estimated iteratively using:
- Loess smoothing for trend estimation
- Seasonal subseries smoothing
- Robust fitting procedures
```

**Comprehensive Implementation:**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.x13 import x13_arima_analysis
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesDecomposition:
    """
    Comprehensive time series decomposition framework with multiple methods
    and advanced analysis capabilities
    """
    
    def __init__(self):
        self.data = None
        self.decomposition_results = {}
        self.diagnostics = {}
        
    def theoretical_foundations(self):
        """Theoretical foundations of time series decomposition"""
        
        foundations = {
            'decomposition_models': {
                'additive_model': {
                    'equation': 'Y_t = T_t + S_t + R_t',
                    'assumption': 'Seasonal fluctuations constant over time',
                    'applications': 'Economic data, linear trends',
                    'properties': 'Seasonal component independent of level'
                },
                'multiplicative_model': {
                    'equation': 'Y_t = T_t × S_t × R_t',
                    'assumption': 'Seasonal fluctuations proportional to level',
                    'applications': 'Financial data, exponential growth',
                    'properties': 'Seasonal component scales with trend'
                },
                'log_additive_model': {
                    'equation': 'log(Y_t) = log(T_t) + log(S_t) + log(R_t)',
                    'assumption': 'Multiplicative in original scale',
                    'applications': 'Positive data with growth',
                    'properties': 'Combines benefits of both models'
                }
            },
            'classical_decomposition': {
                'trend_estimation': 'Moving averages (centered)',
                'seasonal_estimation': 'Average seasonal deviations',
                'residual_calculation': 'Y_t - T_t - S_t (additive)',
                'limitations': 'Fixed seasonal pattern, edge effects',
                'assumptions': 'Stable seasonal pattern, linear trend'
            },
            'stl_decomposition': {
                'trend_component': 'Loess regression with large bandwidth',
                'seasonal_component': 'Seasonal subseries loess smoothing',
                'residual_component': 'Robust residual calculation',
                'advantages': 'Flexible, robust to outliers, time-varying seasonality',
                'parameters': 'seasonal, trend, robust settings'
            },
            'x11_x13_methods': {
                'development': 'US Census Bureau methods',
                'approach': 'Iterative seasonal adjustment',
                'features': 'Trading day effects, outlier detection',
                'applications': 'Official statistics, economic indicators',
                'advantages': 'Industry standard, comprehensive diagnostics'
            },
            'mathematical_properties': {
                'identifiability': 'Components not uniquely identified without constraints',
                'constraints': 'Σ S_t = 0 (seasonal constraint)',
                'optimality': 'Minimize sum of component variations',
                'statistical_properties': 'Bias, variance, consistency of estimators'
            }
        }
        
        return foundations
    
    def generate_synthetic_data(self, n=365, trend_type='linear', seasonal_periods=[12, 7], 
                              noise_level=0.1, random_seed=42):
        """
        Generate synthetic time series with known components
        
        Parameters:
        -----------
        n : int, default=365
            Number of observations
        trend_type : str, default='linear'
            Type of trend ('linear', 'quadratic', 'exponential', 'none')
        seasonal_periods : list, default=[12, 7]
            Seasonal periods to include
        noise_level : float, default=0.1
            Relative noise level
        random_seed : int, default=42
            Random seed for reproducibility
        """
        
        np.random.seed(random_seed)
        
        print(f"=== Generating Synthetic Time Series ===")
        print(f"Observations: {n}, Trend: {trend_type}, Seasonal periods: {seasonal_periods}")
        
        # Time index
        t = np.arange(n)
        dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
        
        # Generate trend component
        if trend_type == 'linear':
            trend = 100 + 0.1 * t
        elif trend_type == 'quadratic':
            trend = 100 + 0.1 * t + 0.0001 * t**2
        elif trend_type == 'exponential':
            trend = 100 * np.exp(0.001 * t)
        elif trend_type == 'step':
            trend = 100 + 5 * (t > n//2).astype(int)
        else:  # none
            trend = np.full(n, 100)
        
        # Generate seasonal components
        seasonal = np.zeros(n)
        seasonal_components = {}
        
        for period in seasonal_periods:
            if period == 12:  # Monthly pattern
                amplitude = 10
                seasonal_comp = amplitude * np.sin(2 * np.pi * t / period)
            elif period == 7:  # Weekly pattern
                amplitude = 5
                seasonal_comp = amplitude * np.cos(2 * np.pi * t / period)
            else:  # Generic pattern
                amplitude = 8
                seasonal_comp = amplitude * np.sin(2 * np.pi * t / period + np.pi/4)
            
            seasonal += seasonal_comp
            seasonal_components[f'period_{period}'] = seasonal_comp
        
        # Generate residual component (noise + irregular patterns)
        base_noise = np.random.normal(0, 1, n)
        
        # Add some irregular patterns
        irregular = np.zeros(n)
        
        # Add some outliers
        outlier_indices = np.random.choice(n, size=n//50, replace=False)
        outlier_magnitudes = np.random.normal(0, 5, len(outlier_indices))
        irregular[outlier_indices] = outlier_magnitudes
        
        # Add autoregressive noise
        ar_noise = np.zeros(n)
        phi = 0.3
        for i in range(1, n):
            ar_noise[i] = phi * ar_noise[i-1] + np.random.normal(0, 0.5)
        
        residual = base_noise + irregular + ar_noise
        
        # Scale noise relative to signal
        signal_level = np.std(trend + seasonal)
        residual = residual * noise_level * signal_level / np.std(residual)
        
        # Combine components
        observed = trend + seasonal + residual
        
        # Create DataFrame
        self.data = pd.DataFrame({
            'observed': observed,
            'trend_true': trend,
            'seasonal_true': seasonal,
            'residual_true': residual
        }, index=dates)
        
        # Store component details
        for period, comp in seasonal_components.items():
            self.data[f'seasonal_{period}_true'] = comp
        
        print(f"Generated components:")
        print(f"  • Trend range: {trend.min():.2f} to {trend.max():.2f}")
        print(f"  • Seasonal amplitude: {np.std(seasonal):.2f}")
        print(f"  • Residual std: {np.std(residual):.2f}")
        print(f"  • Signal-to-noise ratio: {np.std(trend + seasonal) / np.std(residual):.2f}")
        
        return self.data
    
    def classical_decomposition(self, data=None, model='additive', period=None, 
                              two_sided=True, extrapolate_trend=0):
        """
        Perform classical seasonal decomposition
        
        Parameters:
        -----------
        data : pd.Series, optional
            Time series data (uses self.data['observed'] if None)
        model : str, default='additive'
            Type of decomposition ('additive' or 'multiplicative')
        period : int, optional
            Seasonal period (auto-detected if None)
        two_sided : bool, default=True
            Use two-sided moving average for trend
        extrapolate_trend : int, default=0
            Number of periods to extrapolate trend at edges
        """
        
        if data is None:
            if self.data is None:
                raise ValueError("No data available. Generate or load data first.")
            data = self.data['observed']
        
        print(f"\n=== Classical Decomposition ({model.title()}) ===")
        
        # Auto-detect period if not specified
        if period is None:
            period = self._detect_seasonal_period(data)
            print(f"Auto-detected seasonal period: {period}")
        
        # Perform decomposition
        decomposition = seasonal_decompose(
            data, 
            model=model, 
            period=period,
            two_sided=two_sided,
            extrapolate_trend=extrapolate_trend
        )
        
        # Store results
        method_name = f'classical_{model}'
        self.decomposition_results[method_name] = {
            'method': 'Classical Decomposition',
            'model': model,
            'period': period,
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'resid': decomposition.resid,
            'observed': decomposition.observed,
            'decomposition_object': decomposition
        }
        
        # Calculate decomposition statistics
        self._calculate_decomposition_statistics(method_name, data)
        
        print(f"Classical decomposition completed:")
        print(f"  • Model: {model}")
        print(f"  • Period: {period}")
        print(f"  • Two-sided trend: {two_sided}")
        
        return decomposition
    
    def stl_decomposition(self, data=None, period=None, seasonal=7, trend=None, 
                         robust=False):
        """
        Perform STL (Seasonal and Trend decomposition using Loess) decomposition
        
        Parameters:
        -----------
        data : pd.Series, optional
            Time series data
        period : int, optional
            Seasonal period
        seasonal : int, default=7
            Length of seasonal smoother (odd number)
        trend : int, optional
            Length of trend smoother
        robust : bool, default=False
            Use robust fitting
        """
        
        if data is None:
            if self.data is None:
                raise ValueError("No data available. Generate or load data first.")
            data = self.data['observed']
        
        print(f"\n=== STL Decomposition ===")
        
        # Auto-detect period if not specified
        if period is None:
            period = self._detect_seasonal_period(data)
            print(f"Auto-detected seasonal period: {period}")
        
        # Set default trend parameter
        if trend is None:
            trend = int(np.ceil(1.5 * period / (1 - 1.5 / seasonal)))
            if trend % 2 == 0:  # Ensure odd number
                trend += 1
        
        # Ensure seasonal is odd
        if seasonal % 2 == 0:
            seasonal += 1
        
        print(f"STL parameters: seasonal={seasonal}, trend={trend}, robust={robust}")
        
        # Perform STL decomposition
        stl = STL(data, seasonal=seasonal, trend=trend, period=period, robust=robust)
        stl_result = stl.fit()
        
        # Store results
        method_name = 'stl'
        self.decomposition_results[method_name] = {
            'method': 'STL Decomposition',
            'model': 'additive',
            'period': period,
            'trend': stl_result.trend,
            'seasonal': stl_result.seasonal,
            'resid': stl_result.resid,
            'observed': data,
            'stl_object': stl_result,
            'parameters': {
                'seasonal': seasonal,
                'trend': trend,
                'robust': robust
            }
        }
        
        # Calculate decomposition statistics
        self._calculate_decomposition_statistics(method_name, data)
        
        print(f"STL decomposition completed successfully")
        
        return stl_result
    
    def custom_loess_decomposition(self, data=None, period=None, trend_frac=0.3, 
                                  seasonal_frac=0.1, iterations=2):
        """
        Custom implementation of LOESS-based decomposition
        
        Parameters:
        -----------
        data : pd.Series, optional
            Time series data
        period : int, optional
            Seasonal period
        trend_frac : float, default=0.3
            Fraction of data for trend smoothing
        seasonal_frac : float, default=0.1
            Fraction of data for seasonal smoothing
        iterations : int, default=2
            Number of iterations for robust fitting
        """
        
        if data is None:
            if self.data is None:
                raise ValueError("No data available. Generate or load data first.")
            data = self.data['observed']
        
        print(f"\n=== Custom LOESS Decomposition ===")
        
        if period is None:
            period = self._detect_seasonal_period(data)
        
        print(f"Parameters: period={period}, trend_frac={trend_frac}, seasonal_frac={seasonal_frac}")
        
        # Convert to numpy arrays
        x = np.arange(len(data))
        y = data.values
        
        # Initialize components
        trend = np.zeros_like(y)
        seasonal = np.zeros_like(y)
        residual = np.zeros_like(y)
        
        # Iterative decomposition
        current_y = y.copy()
        
        for iteration in range(iterations):
            print(f"  Iteration {iteration + 1}/{iterations}")
            
            # Step 1: Estimate trend using LOESS
            trend_smoothed = lowess(current_y, x, frac=trend_frac, return_sorted=False)
            
            # Step 2: Detrend
            detrended = current_y - trend_smoothed
            
            # Step 3: Estimate seasonal component
            seasonal_estimates = self._estimate_seasonal_loess(detrended, period, seasonal_frac)
            
            # Step 4: Calculate residuals
            residual_current = current_y - trend_smoothed - seasonal_estimates
            
            # Update for next iteration (robust fitting)
            if iteration < iterations - 1:
                # Weight by residual magnitude for robustness
                weights = self._calculate_robust_weights(residual_current)
                current_y = y * weights + trend_smoothed * (1 - weights)
            
            # Store current estimates
            trend = trend_smoothed
            seasonal = seasonal_estimates
            residual = residual_current
        
        # Create pandas series
        trend_series = pd.Series(trend, index=data.index)
        seasonal_series = pd.Series(seasonal, index=data.index)
        residual_series = pd.Series(residual, index=data.index)
        
        # Store results
        method_name = 'custom_loess'
        self.decomposition_results[method_name] = {
            'method': 'Custom LOESS Decomposition',
            'model': 'additive',
            'period': period,
            'trend': trend_series,
            'seasonal': seasonal_series,
            'resid': residual_series,
            'observed': data,
            'parameters': {
                'trend_frac': trend_frac,
                'seasonal_frac': seasonal_frac,
                'iterations': iterations
            }
        }
        
        # Calculate decomposition statistics
        self._calculate_decomposition_statistics(method_name, data)
        
        print(f"Custom LOESS decomposition completed")
        
        return {
            'trend': trend_series,
            'seasonal': seasonal_series,
            'resid': residual_series
        }
    
    def _detect_seasonal_period(self, data, max_period=None):
        """Auto-detect seasonal period using multiple methods"""
        
        if max_period is None:
            max_period = min(len(data) // 3, 365)
        
        # Method 1: Autocorrelation-based detection
        from statsmodels.tsa.stattools import acf
        
        max_lag = min(max_period, len(data) // 4)
        acf_values = acf(data, nlags=max_lag, fft=True)
        
        # Find peaks in ACF
        peaks = []
        for i in range(2, len(acf_values) - 1):
            if (acf_values[i] > acf_values[i-1] and 
                acf_values[i] > acf_values[i+1] and 
                acf_values[i] > 0.1):  # Threshold for significance
                peaks.append((i, acf_values[i]))
        
        if peaks:
            # Return period with highest ACF value
            best_period = max(peaks, key=lambda x: x[1])[0]
        else:
            # Default periods based on data frequency
            if len(data) > 365:
                best_period = 365  # Daily data -> yearly seasonality
            elif len(data) > 52:
                best_period = 52   # Weekly data -> yearly seasonality
            elif len(data) > 12:
                best_period = 12   # Monthly data -> yearly seasonality
            else:
                best_period = 4    # Quarterly data
        
        return best_period
    
    def _estimate_seasonal_loess(self, detrended_data, period, frac):
        """Estimate seasonal component using LOESS on seasonal subseries"""
        
        n = len(detrended_data)
        seasonal_estimates = np.zeros(n)
        
        # For each seasonal position
        for s in range(period):
            # Extract subseries for this seasonal position
            indices = np.arange(s, n, period)
            
            if len(indices) > 1:
                subseries = detrended_data[indices]
                subseries_x = indices
                
                # Apply LOESS smoothing to subseries
                if len(indices) > 3:  # Need minimum points for LOESS
                    smoothed_subseries = lowess(subseries, subseries_x, frac=frac, return_sorted=False)
                else:
                    smoothed_subseries = subseries  # Use original values
                
                # Assign back to seasonal estimates
                seasonal_estimates[indices] = smoothed_subseries
            else:
                # Single point - use mean of all available seasonal values
                seasonal_estimates[s] = 0
        
        # Ensure seasonal component sums to zero (seasonal constraint)
        for cycle_start in range(0, n - period + 1, period):
            cycle_end = min(cycle_start + period, n)
            cycle_mean = np.mean(seasonal_estimates[cycle_start:cycle_end])
            seasonal_estimates[cycle_start:cycle_end] -= cycle_mean
        
        return seasonal_estimates
    
    def _calculate_robust_weights(self, residuals):
        """Calculate robust weights for iterative fitting"""
        
        # Median absolute deviation
        mad = np.median(np.abs(residuals - np.median(residuals)))
        
        if mad == 0:
            return np.ones_like(residuals)
        
        # Standardized residuals
        standardized_resid = residuals / (6 * mad)
        
        # Tukey's bisquare weights
        weights = np.where(np.abs(standardized_resid) < 1,
                          (1 - standardized_resid**2)**2,
                          0)
        
        return weights
    
    def _calculate_decomposition_statistics(self, method_name, original_data):
        """Calculate comprehensive statistics for decomposition quality"""
        
        result = self.decomposition_results[method_name]
        
        trend = result['trend'].dropna()
        seasonal = result['seasonal'].dropna()
        residual = result['resid'].dropna()
        
        # Align all series for calculation
        common_index = trend.index.intersection(seasonal.index).intersection(residual.index)
        
        if len(common_index) == 0:
            print(f"Warning: No common index for {method_name} statistics")
            return
        
        trend_aligned = trend[common_index]
        seasonal_aligned = seasonal[common_index]
        residual_aligned = residual[common_index]
        reconstructed = trend_aligned + seasonal_aligned + residual_aligned
        original_aligned = original_data[common_index]
        
        # Reconstruction accuracy
        reconstruction_mse = mean_squared_error(original_aligned, reconstructed)
        reconstruction_r2 = r2_score(original_aligned, reconstructed)
        
        # Component strength measures
        total_variance = np.var(original_aligned)
        trend_strength = np.var(trend_aligned) / total_variance
        seasonal_strength = np.var(seasonal_aligned) / total_variance
        residual_strength = np.var(residual_aligned) / total_variance
        
        # Seasonal stability (consistency across cycles)
        period = result['period']
        seasonal_stability = self._calculate_seasonal_stability(seasonal_aligned, period)
        
        # Trend smoothness
        trend_smoothness = self._calculate_trend_smoothness(trend_aligned)
        
        # Residual diagnostics
        residual_autocorr = self._calculate_residual_autocorrelation(residual_aligned)
        
        # Store statistics
        self.decomposition_results[method_name]['statistics'] = {
            'reconstruction_mse': reconstruction_mse,
            'reconstruction_r2': reconstruction_r2,
            'trend_strength': trend_strength,
            'seasonal_strength': seasonal_strength,
            'residual_strength': residual_strength,
            'seasonal_stability': seasonal_stability,
            'trend_smoothness': trend_smoothness,
            'residual_autocorr': residual_autocorr,
            'component_variance_explained': {
                'trend': trend_strength,
                'seasonal': seasonal_strength,
                'residual': residual_strength
            }
        }
        
        print(f"  Statistics for {method_name}:")
        print(f"    • Reconstruction R²: {reconstruction_r2:.4f}")
        print(f"    • Trend strength: {trend_strength:.3f}")
        print(f"    • Seasonal strength: {seasonal_strength:.3f}")
        print(f"    • Residual autocorr: {residual_autocorr:.3f}")
    
    def _calculate_seasonal_stability(self, seasonal_component, period):
        """Calculate seasonal pattern stability across cycles"""
        
        n = len(seasonal_component)
        n_cycles = n // period
        
        if n_cycles < 2:
            return np.nan
        
        # Reshape into cycles
        cycles_data = seasonal_component.iloc[:n_cycles * period].values.reshape(n_cycles, period)
        
        # Calculate coefficient of variation for each seasonal position
        cv_values = []
        for pos in range(period):
            position_values = cycles_data[:, pos]
            if np.std(position_values) > 0:
                cv = np.std(position_values) / (np.abs(np.mean(position_values)) + 1e-8)
                cv_values.append(cv)
        
        # Return inverse of mean CV (higher = more stable)
        mean_cv = np.mean(cv_values) if cv_values else np.inf
        stability = 1 / (1 + mean_cv)
        
        return stability
    
    def _calculate_trend_smoothness(self, trend_component):
        """Calculate trend smoothness using second differences"""
        
        if len(trend_component) < 3:
            return np.nan
        
        # Calculate second differences
        first_diff = np.diff(trend_component)
        second_diff = np.diff(first_diff)
        
        # Smoothness = inverse of variance of second differences
        if np.var(second_diff) > 0:
            smoothness = 1 / (1 + np.var(second_diff))
        else:
            smoothness = 1.0
        
        return smoothness
    
    def _calculate_residual_autocorrelation(self, residuals, max_lag=10):
        """Calculate residual autocorrelation for diagnostic purposes"""
        
        from statsmodels.tsa.stattools import acf
        
        max_lag = min(max_lag, len(residuals) // 4)
        
        if max_lag < 1:
            return np.nan
        
        try:
            acf_values = acf(residuals, nlags=max_lag, fft=True)
            # Return mean absolute autocorrelation (excluding lag 0)
            return np.mean(np.abs(acf_values[1:]))
        except:
            return np.nan
    
    def compare_decomposition_methods(self):
        """Compare all decomposition methods and recommend best approach"""
        
        if not self.decomposition_results:
            raise ValueError("No decomposition results available")
        
        print(f"\n=== Decomposition Methods Comparison ===")
        
        comparison_data = []
        
        for method_name, result in self.decomposition_results.items():
            if 'statistics' in result:
                stats = result['statistics']
                comparison_data.append({
                    'Method': result['method'],
                    'Model': result['model'],
                    'R²': stats['reconstruction_r2'],
                    'Trend Strength': stats['trend_strength'],
                    'Seasonal Strength': stats['seasonal_strength'],
                    'Seasonal Stability': stats['seasonal_stability'],
                    'Trend Smoothness': stats['trend_smoothness'],
                    'Residual Autocorr': stats['residual_autocorr']
                })
        
        if not comparison_data:
            print("No statistics available for comparison")
            return
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print(comparison_df.round(4))
        
        # Recommend best method
        # Criteria: High R², low residual autocorr, high seasonal stability
        scores = []
        for _, row in comparison_df.iterrows():
            score = (
                row['R²'] * 0.4 +  # Reconstruction quality
                (1 - row['Residual Autocorr']) * 0.3 +  # Low residual autocorr
                row['Seasonal Stability'] * 0.2 +  # Seasonal stability
                row['Trend Smoothness'] * 0.1  # Trend smoothness
            )
            scores.append(score)
        
        best_idx = np.argmax(scores)
        best_method = comparison_df.iloc[best_idx]['Method']
        
        print(f"\nRecommended method: {best_method}")
        print(f"Overall score: {scores[best_idx]:.3f}")
        
        return comparison_df
    
    def plot_comprehensive_decomposition(self, method_name=None, figsize=(16, 12)):
        """Create comprehensive visualization of decomposition results"""
        
        if not self.decomposition_results:
            raise ValueError("No decomposition results available")
        
        if method_name is None:
            method_name = list(self.decomposition_results.keys())[0]
        
        if method_name not in self.decomposition_results:
            raise ValueError(f"Method {method_name} not found")
        
        result = self.decomposition_results[method_name]
        
        fig, axes = plt.subplots(4, 1, figsize=figsize)
        fig.suptitle(f'{result["method"]} - {result["model"].title()} Model', 
                     fontsize=16, fontweight='bold')
        
        # Original time series
        axes[0].plot(result['observed'].index, result['observed'], 'b-', linewidth=1.5)
        axes[0].set_title('Original Time Series')
        axes[0].set_ylabel('Value')
        axes[0].grid(True, alpha=0.3)
        
        # Trend component
        trend_data = result['trend'].dropna()
        axes[1].plot(trend_data.index, trend_data, 'g-', linewidth=2)
        axes[1].set_title('Trend Component')
        axes[1].set_ylabel('Trend')
        axes[1].grid(True, alpha=0.3)
        
        # Seasonal component
        seasonal_data = result['seasonal'].dropna()
        axes[2].plot(seasonal_data.index, seasonal_data, 'r-', linewidth=1.5)
        axes[2].set_title('Seasonal Component')
        axes[2].set_ylabel('Seasonal')
        axes[2].grid(True, alpha=0.3)
        
        # Residual component
        residual_data = result['resid'].dropna()
        axes[3].plot(residual_data.index, residual_data, 'k-', linewidth=1, alpha=0.7)
        axes[3].axhline(y=0, color='r', linestyle='--', alpha=0.7)
        axes[3].set_title('Residual Component')
        axes[3].set_xlabel('Time')
        axes[3].set_ylabel('Residual')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Additional diagnostic plots
        self._plot_decomposition_diagnostics(result, method_name)
        
        return fig
    
    def _plot_decomposition_diagnostics(self, result, method_name):
        """Create diagnostic plots for decomposition quality assessment"""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Decomposition Diagnostics - {method_name}', fontsize=14, fontweight='bold')
        
        # Reconstruction vs Original
        original = result['observed']
        trend = result['trend'].dropna()
        seasonal = result['seasonal'].dropna()
        
        # Align for reconstruction
        common_index = trend.index.intersection(seasonal.index)
        if len(common_index) > 0:
            reconstructed = trend[common_index] + seasonal[common_index]
            original_aligned = original[common_index]
            
            axes[0, 0].scatter(original_aligned, reconstructed, alpha=0.6)
            axes[0, 0].plot([original_aligned.min(), original_aligned.max()], 
                          [original_aligned.min(), original_aligned.max()], 'r--', alpha=0.8)
            axes[0, 0].set_xlabel('Original')
            axes[0, 0].set_ylabel('Reconstructed')
            axes[0, 0].set_title('Reconstruction Accuracy')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Residual histogram
        residuals = result['resid'].dropna()
        axes[0, 1].hist(residuals, bins=30, density=True, alpha=0.7, color='skyblue')
        
        # Overlay normal distribution
        mu, sigma = residuals.mean(), residuals.std()
        x = np.linspace(residuals.min(), residuals.max(), 100)
        axes[0, 1].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Residual Value')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Residual Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Residual ACF
        from statsmodels.tsa.stattools import acf
        max_lag = min(20, len(residuals) // 4)
        
        if max_lag > 0:
            acf_values = acf(residuals, nlags=max_lag, alpha=0.05)
            lags = np.arange(max_lag + 1)
            
            axes[0, 2].stem(lags, acf_values[0], linefmt='b-', markerfmt='bo')
            conf_bound = 1.96 / np.sqrt(len(residuals))
            axes[0, 2].axhline(y=conf_bound, color='r', linestyle='--', alpha=0.7)
            axes[0, 2].axhline(y=-conf_bound, color='r', linestyle='--', alpha=0.7)
            axes[0, 2].set_xlabel('Lag')
            axes[0, 2].set_ylabel('ACF')
            axes[0, 2].set_title('Residual Autocorrelation')
            axes[0, 2].grid(True, alpha=0.3)
        
        # Seasonal pattern consistency
        period = result['period']
        seasonal_data = result['seasonal'].dropna()
        
        if len(seasonal_data) >= period:
            n_cycles = len(seasonal_data) // period
            seasonal_matrix = seasonal_data.iloc[:n_cycles * period].values.reshape(n_cycles, period)
            
            # Plot each cycle
            for i in range(min(n_cycles, 5)):  # Limit to first 5 cycles
                axes[1, 0].plot(range(period), seasonal_matrix[i, :], alpha=0.6, linewidth=1)
            
            # Plot mean seasonal pattern
            mean_seasonal = np.mean(seasonal_matrix, axis=0)
            axes[1, 0].plot(range(period), mean_seasonal, 'k-', linewidth=3, label='Mean')
            axes[1, 0].set_xlabel('Seasonal Position')
            axes[1, 0].set_ylabel('Seasonal Value')
            axes[1, 0].set_title('Seasonal Pattern Consistency')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Trend analysis
        trend_data = result['trend'].dropna()
        
        # Calculate trend acceleration
        if len(trend_data) > 2:
            trend_diff = np.diff(trend_data)
            trend_accel = np.diff(trend_diff)
            
            axes[1, 1].plot(trend_data.index[2:], trend_accel, 'g-', linewidth=1.5)
            axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.7)
            axes[1, 1].set_xlabel('Time')
            axes[1, 1].set_ylabel('Trend Acceleration')
            axes[1, 1].set_title('Trend Acceleration')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Component variance pie chart
        if 'statistics' in result:
            stats_data = result['statistics']
            
            strengths = [
                stats_data['trend_strength'],
                stats_data['seasonal_strength'],
                stats_data['residual_strength']
            ]
            labels = ['Trend', 'Seasonal', 'Residual']
            colors = ['green', 'red', 'blue']
            
            axes[1, 2].pie(strengths, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            axes[1, 2].set_title('Component Variance Explained')
        
        plt.tight_layout()
        plt.show()
        
        return fig

def demonstrate_time_series_decomposition():
    """Comprehensive demonstration of time series decomposition"""
    
    print("=== Time Series Decomposition Demonstration ===")
    
    # Initialize decomposition analyzer
    decomposer = TimeSeriesDecomposition()
    
    # Generate synthetic data with known components
    data = decomposer.generate_synthetic_data(
        n=730,  # 2 years of daily data
        trend_type='linear',
        seasonal_periods=[365, 7],  # Annual and weekly seasonality
        noise_level=0.15
    )
    
    # Perform different decomposition methods
    print(f"\n=== Performing Multiple Decomposition Methods ===")
    
    # 1. Classical additive decomposition
    classical_add = decomposer.classical_decomposition(
        model='additive', 
        period=365
    )
    
    # 2. Classical multiplicative decomposition
    classical_mult = decomposer.classical_decomposition(
        model='multiplicative', 
        period=365
    )
    
    # 3. STL decomposition
    stl_result = decomposer.stl_decomposition(
        period=365,
        seasonal=7,
        robust=True
    )
    
    # 4. Custom LOESS decomposition
    custom_result = decomposer.custom_loess_decomposition(
        period=365,
        trend_frac=0.2,
        seasonal_frac=0.1
    )
    
    # Compare methods
    comparison_df = decomposer.compare_decomposition_methods()
    
    # Visualize best method
    decomposer.plot_comprehensive_decomposition('stl')
    
    # Validation against true components (since we have synthetic data)
    print(f"\n=== Validation Against True Components ===")
    
    true_trend = data['trend_true']
    true_seasonal = data['seasonal_true']
    
    for method_name, result in decomposer.decomposition_results.items():
        estimated_trend = result['trend'].dropna()
        estimated_seasonal = result['seasonal'].dropna()
        
        # Align series
        common_trend_idx = true_trend.index.intersection(estimated_trend.index)
        common_seasonal_idx = true_seasonal.index.intersection(estimated_seasonal.index)
        
        if len(common_trend_idx) > 0:
            trend_correlation = np.corrcoef(true_trend[common_trend_idx], 
                                          estimated_trend[common_trend_idx])[0, 1]
        else:
            trend_correlation = np.nan
            
        if len(common_seasonal_idx) > 0:
            seasonal_correlation = np.corrcoef(true_seasonal[common_seasonal_idx], 
                                             estimated_seasonal[common_seasonal_idx])[0, 1]
        else:
            seasonal_correlation = np.nan
        
        print(f"{method_name}:")
        print(f"  • Trend correlation: {trend_correlation:.3f}")
        print(f"  • Seasonal correlation: {seasonal_correlation:.3f}")
    
    # Practical insights
    print(f"\n=== Practical Insights ===")
    print("Method Selection Guidelines:")
    print("  • Classical: Simple, fast, assumes stable seasonality")
    print("  • STL: Flexible, robust, handles changing seasonality")
    print("  • Custom LOESS: Maximum control, computationally intensive")
    print("  • Multiplicative: Use when seasonal variation scales with level")
    
    return decomposer, comparison_df

# Execute demonstration
if __name__ == "__main__":
    decomposer, comparison = demonstrate_time_series_decomposition()
```

**Key Theoretical Insights:**

1. **Component Identifiability**: Time series components are not uniquely identified without constraints (e.g., seasonal constraint Σ S_t = 0)

2. **Model Selection**: Choose additive when seasonal fluctuations are constant, multiplicative when they scale with level

3. **Classical vs. STL**: Classical assumes stable seasonality; STL allows time-varying seasonal patterns

4. **LOESS Smoothing**: Local regression provides flexible trend and seasonal estimation with robustness to outliers

5. **Decomposition Quality**: Assess via reconstruction accuracy, component strength, seasonal stability, and residual diagnostics

6. **Edge Effects**: Classical decomposition suffers from edge effects; STL and custom methods can mitigate these

7. **Robust Estimation**: Iterative robust fitting reduces impact of outliers on component estimates

8. **Practical Applications**: Decomposition enables better forecasting, seasonal adjustment, and anomaly detection

**Applications:**
- **Economic Analysis**: Seasonal adjustment of economic indicators
- **Business Intelligence**: Trend analysis and seasonal pattern identification
- **Forecasting**: Component-wise modeling for improved predictions
- **Anomaly Detection**: Residual analysis for outlier identification

---

## Question 6

**Write aPython functionto calculate and plot theACFandPACFfor a giventime series.**

**Answer:**

**Theoretical Foundation:**

The **Autocorrelation Function (ACF)** and **Partial Autocorrelation Function (PACF)** are fundamental tools in time series analysis for **model identification**, **pattern recognition**, and **statistical inference**. These functions provide deep insights into the **temporal dependence structure** and are essential for **Box-Jenkins methodology** in ARIMA modeling.

**Mathematical Framework:**

**Autocorrelation Function (ACF):**
```
ρ(k) = Cov(X_t, X_{t-k}) / Var(X_t) = γ(k) / γ(0)

Where:
- ρ(k): Autocorrelation at lag k
- γ(k): Autocovariance at lag k
- γ(0): Variance of the series
```

**Sample Autocorrelation:**
```
r_k = Σ(X_t - X̄)(X_{t-k} - X̄) / Σ(X_t - X̄)²

Asymptotic Distribution: r_k ~ N(0, 1/n) under white noise null hypothesis
```

**Partial Autocorrelation Function (PACF):**
```
φ_{kk} = Corr(X_t, X_{t-k} | X_{t-1}, X_{t-2}, ..., X_{t-k+1})

The PACF at lag k is the correlation between X_t and X_{t-k} after removing
the linear dependence on the intermediate variables.
```

**Yule-Walker Equations for PACF:**
```
φ_{kk} = (ρ_k - Σ_{j=1}^{k-1} φ_{k-1,j} ρ_{k-j}) / (1 - Σ_{j=1}^{k-1} φ_{k-1,j} ρ_j)
```

**Comprehensive Implementation:**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import acf, pacf, ccf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
from scipy.special import factorial
import warnings
warnings.filterwarnings('ignore')

class ACFPACFAnalysis:
    """
    Comprehensive ACF and PACF analysis framework with theoretical foundations,
    statistical testing, and advanced diagnostic capabilities
    """
    
    def __init__(self):
        self.acf_results = {}
        self.pacf_results = {}
        self.diagnostic_results = {}
        
    def theoretical_foundations(self):
        """Theoretical foundations of ACF and PACF analysis"""
        
        foundations = {
            'acf_properties': {
                'definition': 'Correlation between X_t and X_{t-k}',
                'range': '[-1, 1] for all lags',
                'symmetry': 'ρ(k) = ρ(-k) for stationary processes',
                'normalization': 'ρ(0) = 1 by definition',
                'white_noise': 'ρ(k) = 0 for k ≠ 0 in white noise'
            },
            'pacf_properties': {
                'definition': 'Partial correlation removing intermediate dependencies',
                'interpretation': 'Direct correlation at lag k',
                'ar_identification': 'PACF cuts off at order p for AR(p)',
                'computational_methods': 'Yule-Walker, Burg, MLE methods',
                'statistical_properties': 'φ_{kk} ~ N(0, 1/n) for k > p in AR(p)'
            },
            'model_identification_patterns': {
                'ar_process': {
                    'acf': 'Exponential decay or damped sinusoid',
                    'pacf': 'Cuts off after lag p',
                    'example': 'AR(1): ρ(k) = φ^k, φ_{11} = φ, φ_{kk} = 0 for k > 1'
                },
                'ma_process': {
                    'acf': 'Cuts off after lag q',
                    'pacf': 'Exponential decay or damped sinusoid',
                    'example': 'MA(1): ρ(1) = θ/(1+θ²), ρ(k) = 0 for k > 1'
                },
                'arma_process': {
                    'acf': 'Exponential decay after lag q',
                    'pacf': 'Exponential decay after lag p',
                    'complexity': 'Both tails off without clear cutoff'
                }
            },
            'statistical_inference': {
                'confidence_bounds': '±1.96/√n for 95% confidence under H₀',
                'ljung_box_test': 'Joint test for multiple lags',
                'box_pierce_test': 'Classical portmanteau test',
                'modified_tests': 'Robust versions for heteroscedasticity'
            },
            'computational_methods': {
                'acf_estimation': 'Biased vs unbiased estimators',
                'pacf_estimation': 'Yule-Walker vs Burg algorithm',
                'fft_computation': 'Fast computation for large series',
                'missing_data': 'Handling irregular observations'
            }
        }
        
        return foundations
    
    def calculate_acf_comprehensive(self, data, nlags=40, alpha=0.05, fft=True, 
                                  missing='none', adjusted=False):
        """
        Comprehensive ACF calculation with multiple methods and diagnostics
        
        Parameters:
        -----------
        data : array-like
            Time series data
        nlags : int, default=40
            Number of lags to calculate
        alpha : float, default=0.05
            Significance level for confidence intervals
        fft : bool, default=True
            Use FFT for faster computation
        missing : str, default='none'
            How to handle missing data ('none', 'raise', 'conservative', 'drop')
        adjusted : bool, default=False
            Use bias-adjusted correlation estimates
        """
        
        print(f"=== Comprehensive ACF Analysis ===")
        
        # Convert to numpy array and handle missing data
        data_clean = self._preprocess_data(data, missing)
        n = len(data_clean)
        
        # Adjust nlags if necessary
        nlags = min(nlags, n // 4)
        
        print(f"Series length: {n}, Computing {nlags} lags")
        
        # Method 1: statsmodels ACF
        acf_values_sm = acf(data_clean, nlags=nlags, alpha=alpha, fft=fft, adjusted=adjusted)
        
        if isinstance(acf_values_sm, tuple):
            acf_sm, confint_sm = acf_values_sm
        else:
            acf_sm = acf_values_sm
            confint_sm = None
        
        # Method 2: Manual ACF calculation
        acf_manual = self._calculate_acf_manual(data_clean, nlags, adjusted)
        
        # Method 3: FFT-based ACF
        acf_fft = self._calculate_acf_fft(data_clean, nlags)
        
        # Statistical diagnostics
        diagnostics = self._acf_diagnostics(acf_sm, n, nlags, alpha)
        
        # Store results
        self.acf_results = {
            'lags': np.arange(nlags + 1),
            'acf_statsmodels': acf_sm,
            'acf_manual': acf_manual,
            'acf_fft': acf_fft,
            'confidence_intervals': confint_sm,
            'alpha': alpha,
            'n_obs': n,
            'nlags': nlags,
            'diagnostics': diagnostics,
            'parameters': {
                'fft': fft,
                'adjusted': adjusted,
                'missing': missing
            }
        }
        
        # Print summary
        print(f"ACF calculation completed:")
        print(f"  • Method comparison: Max diff = {np.max(np.abs(acf_sm - acf_manual)):.6f}")
        print(f"  • Significant lags: {diagnostics['significant_lags_count']}")
        print(f"  • Ljung-Box p-value: {diagnostics['ljung_box_pvalue']:.4f}")
        
        return self.acf_results
    
    def calculate_pacf_comprehensive(self, data, nlags=40, alpha=0.05, method='yule_walker'):
        """
        Comprehensive PACF calculation with multiple methods
        
        Parameters:
        -----------
        data : array-like
            Time series data
        nlags : int, default=40
            Number of lags to calculate
        alpha : float, default=0.05
            Significance level for confidence intervals
        method : str, default='yule_walker'
            Method for PACF computation ('yule_walker', 'ols', 'burg')
        """
        
        print(f"\n=== Comprehensive PACF Analysis ===")
        
        # Preprocess data
        data_clean = self._preprocess_data(data)
        n = len(data_clean)
        nlags = min(nlags, n // 4)
        
        print(f"Method: {method}, Computing {nlags} lags")
        
        # statsmodels PACF
        pacf_values_sm = pacf(data_clean, nlags=nlags, alpha=alpha, method=method)
        
        if isinstance(pacf_values_sm, tuple):
            pacf_sm, confint_pacf = pacf_values_sm
        else:
            pacf_sm = pacf_values_sm
            confint_pacf = None
        
        # Manual PACF calculation using Yule-Walker
        pacf_manual = self._calculate_pacf_manual(data_clean, nlags)
        
        # Burg algorithm PACF
        if method != 'burg':
            try:
                pacf_burg = pacf(data_clean, nlags=nlags, method='burg', alpha=None)
                if isinstance(pacf_burg, tuple):
                    pacf_burg = pacf_burg[0]
            except:
                pacf_burg = np.full(nlags + 1, np.nan)
        else:
            pacf_burg = pacf_sm
        
        # PACF diagnostics
        pacf_diagnostics = self._pacf_diagnostics(pacf_sm, n, nlags, alpha)
        
        # Store results
        self.pacf_results = {
            'lags': np.arange(nlags + 1),
            'pacf_statsmodels': pacf_sm,
            'pacf_manual': pacf_manual,
            'pacf_burg': pacf_burg,
            'confidence_intervals': confint_pacf,
            'alpha': alpha,
            'n_obs': n,
            'nlags': nlags,
            'method': method,
            'diagnostics': pacf_diagnostics
        }
        
        print(f"PACF calculation completed:")
        print(f"  • Method comparison: Max diff = {np.max(np.abs(pacf_sm - pacf_manual)):.6f}")
        print(f"  • Significant lags: {pacf_diagnostics['significant_lags_count']}")
        print(f"  • Suggested AR order: {pacf_diagnostics['suggested_ar_order']}")
        
        return self.pacf_results
    
    def _preprocess_data(self, data, missing='none'):
        """Preprocess data for ACF/PACF analysis"""
        
        # Convert to numpy array
        if isinstance(data, (pd.Series, pd.DataFrame)):
            if isinstance(data, pd.DataFrame):
                data = data.iloc[:, 0]  # Take first column
            data_array = data.values
        else:
            data_array = np.asarray(data)
        
        # Handle missing data
        if missing == 'drop':
            data_array = data_array[~np.isnan(data_array)]
        elif missing == 'raise' and np.any(np.isnan(data_array)):
            raise ValueError("Missing values found in data")
        elif missing == 'conservative':
            # Replace with interpolation
            if np.any(np.isnan(data_array)):
                data_series = pd.Series(data_array)
                data_array = data_series.interpolate().fillna(method='bfill').fillna(method='ffill').values
        
        if len(data_array) == 0:
            raise ValueError("No valid data after preprocessing")
        
        return data_array
    
    def _calculate_acf_manual(self, data, nlags, adjusted=False):
        """Manual ACF calculation for verification"""
        
        n = len(data)
        mean_val = np.mean(data)
        
        # Center the data
        centered_data = data - mean_val
        
        acf_manual = np.zeros(nlags + 1)
        acf_manual[0] = 1.0  # ACF at lag 0 is always 1
        
        # Calculate variance (lag 0 autocovariance)
        c0 = np.sum(centered_data**2) / (n if not adjusted else n)
        
        for k in range(1, nlags + 1):
            if k < n:
                # Autocovariance at lag k
                if adjusted:
                    ck = np.sum(centered_data[:-k] * centered_data[k:]) / (n - k)
                else:
                    ck = np.sum(centered_data[:-k] * centered_data[k:]) / n
                
                # Autocorrelation
                acf_manual[k] = ck / c0
            else:
                acf_manual[k] = 0
        
        return acf_manual
    
    def _calculate_acf_fft(self, data, nlags):
        """FFT-based ACF calculation"""
        
        n = len(data)
        
        # Center the data
        centered_data = data - np.mean(data)
        
        # Pad with zeros to avoid circular correlation
        padded_data = np.concatenate([centered_data, np.zeros(n)])
        
        # Compute FFT
        fft_data = np.fft.fft(padded_data)
        
        # Compute power spectral density
        psd = fft_data * np.conj(fft_data)
        
        # Inverse FFT to get autocovariance
        autocov = np.fft.ifft(psd).real
        
        # Take first n lags and normalize
        autocov = autocov[:n]
        acf_fft = autocov / autocov[0]
        
        # Return requested number of lags
        return acf_fft[:nlags + 1]
    
    def _calculate_pacf_manual(self, data, nlags):
        """Manual PACF calculation using Yule-Walker equations"""
        
        # First calculate ACF
        acf_vals = self._calculate_acf_manual(data, nlags)
        
        pacf_manual = np.zeros(nlags + 1)
        pacf_manual[0] = 1.0  # PACF at lag 0 is always 1
        
        if nlags == 0:
            return pacf_manual
        
        # PACF at lag 1 equals ACF at lag 1
        pacf_manual[1] = acf_vals[1]
        
        # For higher lags, use Yule-Walker recursion
        for k in range(2, nlags + 1):
            # Build correlation matrix R and vector r
            R = np.zeros((k-1, k-1))
            r = np.zeros(k-1)
            
            for i in range(k-1):
                for j in range(k-1):
                    R[i, j] = acf_vals[abs(i - j)]
                r[i] = acf_vals[k - 1 - i]
            
            try:
                # Solve Yule-Walker equations
                phi = np.linalg.solve(R, r)
                
                # PACF at lag k
                numerator = acf_vals[k] - np.sum(phi * acf_vals[k-1:0:-1])
                denominator = 1 - np.sum(phi * acf_vals[1:k])
                
                if abs(denominator) > 1e-10:
                    pacf_manual[k] = numerator / denominator
                else:
                    pacf_manual[k] = 0
                    
            except np.linalg.LinAlgError:
                # Singular matrix - set to 0
                pacf_manual[k] = 0
        
        return pacf_manual
    
    def _acf_diagnostics(self, acf_values, n, nlags, alpha):
        """Comprehensive ACF diagnostics"""
        
        # Confidence bounds for white noise
        confidence_bound = stats.norm.ppf(1 - alpha/2) / np.sqrt(n)
        
        # Count significant lags
        significant_lags = []
        for k in range(1, len(acf_values)):
            if abs(acf_values[k]) > confidence_bound:
                significant_lags.append(k)
        
        # Ljung-Box test
        ljung_box_stat, ljung_box_pvalue = self._ljung_box_test(acf_values[1:], n, nlags)
        
        # Box-Pierce test
        box_pierce_stat = n * np.sum(acf_values[1:]**2)
        box_pierce_pvalue = 1 - stats.chi2.cdf(box_pierce_stat, nlags)
        
        # Decay pattern analysis
        decay_pattern = self._analyze_acf_decay(acf_values)
        
        diagnostics = {
            'confidence_bound': confidence_bound,
            'significant_lags': significant_lags,
            'significant_lags_count': len(significant_lags),
            'ljung_box_statistic': ljung_box_stat,
            'ljung_box_pvalue': ljung_box_pvalue,
            'box_pierce_statistic': box_pierce_stat,
            'box_pierce_pvalue': box_pierce_pvalue,
            'decay_pattern': decay_pattern,
            'max_acf_lag': int(np.argmax(np.abs(acf_values[1:]))) + 1 if nlags > 0 else 0
        }
        
        return diagnostics
    
    def _pacf_diagnostics(self, pacf_values, n, nlags, alpha):
        """Comprehensive PACF diagnostics"""
        
        # Confidence bounds
        confidence_bound = stats.norm.ppf(1 - alpha/2) / np.sqrt(n)
        
        # Count significant lags
        significant_lags = []
        for k in range(1, len(pacf_values)):
            if abs(pacf_values[k]) > confidence_bound:
                significant_lags.append(k)
        
        # Suggest AR order (last significant lag)
        if significant_lags:
            suggested_ar_order = max(significant_lags)
        else:
            suggested_ar_order = 0
        
        # Alternative suggestion: first non-significant lag after initial significant lags
        alternative_ar_order = 0
        for k in range(1, len(pacf_values)):
            if abs(pacf_values[k]) > confidence_bound:
                alternative_ar_order = k
            else:
                break
        
        diagnostics = {
            'confidence_bound': confidence_bound,
            'significant_lags': significant_lags,
            'significant_lags_count': len(significant_lags),
            'suggested_ar_order': suggested_ar_order,
            'alternative_ar_order': alternative_ar_order,
            'max_pacf_lag': int(np.argmax(np.abs(pacf_values[1:]))) + 1 if nlags > 0 else 0
        }
        
        return diagnostics
    
    def _ljung_box_test(self, acf_values, n, nlags):
        """Ljung-Box test for joint significance of autocorrelations"""
        
        # Ljung-Box statistic
        lb_stat = n * (n + 2) * np.sum([(acf_values[k-1]**2) / (n - k) 
                                        for k in range(1, min(len(acf_values) + 1, nlags + 1))])
        
        # Degrees of freedom
        df = min(len(acf_values), nlags)
        
        # P-value
        lb_pvalue = 1 - stats.chi2.cdf(lb_stat, df)
        
        return lb_stat, lb_pvalue
    
    def _analyze_acf_decay(self, acf_values):
        """Analyze ACF decay pattern for model identification"""
        
        if len(acf_values) < 3:
            return "insufficient_data"
        
        # Check for exponential decay
        ratios = []
        for k in range(2, min(len(acf_values), 10)):
            if abs(acf_values[k-1]) > 1e-6:
                ratio = acf_values[k] / acf_values[k-1]
                ratios.append(ratio)
        
        if ratios:
            mean_ratio = np.mean(ratios)
            ratio_stability = np.std(ratios) / (abs(mean_ratio) + 1e-6)
            
            if ratio_stability < 0.5 and abs(mean_ratio) < 0.95:
                if mean_ratio > 0:
                    return "exponential_decay"
                else:
                    return "alternating_decay"
        
        # Check for cutoff pattern
        cutoff_point = None
        for k in range(1, len(acf_values)):
            if abs(acf_values[k]) < 0.1:
                cutoff_point = k
                break
        
        if cutoff_point and cutoff_point <= 5:
            return f"cutoff_at_lag_{cutoff_point}"
        
        # Check for sinusoidal pattern
        if len(acf_values) >= 8:
            sign_changes = sum(1 for k in range(1, len(acf_values)-1) 
                             if (acf_values[k] * acf_values[k+1] < 0))
            if sign_changes >= 3:
                return "oscillating"
        
        return "no_clear_pattern"
    
    def plot_acf_pacf_comprehensive(self, figsize=(16, 12), title_prefix=""):
        """Create comprehensive ACF and PACF plots with diagnostics"""
        
        if not self.acf_results or not self.pacf_results:
            raise ValueError("Must calculate ACF and PACF first")
        
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        fig.suptitle(f'{title_prefix}Comprehensive ACF and PACF Analysis', 
                     fontsize=16, fontweight='bold')
        
        lags = self.acf_results['lags']
        acf_vals = self.acf_results['acf_statsmodels']
        pacf_vals = self.pacf_results['pacf_statsmodels']
        
        # ACF plot
        axes[0, 0].stem(lags, acf_vals, linefmt='b-', markerfmt='bo', basefmt=' ')
        
        # Confidence intervals
        conf_bound = self.acf_results['diagnostics']['confidence_bound']
        axes[0, 0].axhline(y=conf_bound, color='r', linestyle='--', alpha=0.7, label='95% CI')
        axes[0, 0].axhline(y=-conf_bound, color='r', linestyle='--', alpha=0.7)
        axes[0, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        axes[0, 0].set_title('Autocorrelation Function (ACF)')
        axes[0, 0].set_xlabel('Lag')
        axes[0, 0].set_ylabel('ACF')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # PACF plot
        axes[0, 1].stem(lags, pacf_vals, linefmt='g-', markerfmt='go', basefmt=' ')
        
        conf_bound_pacf = self.pacf_results['diagnostics']['confidence_bound']
        axes[0, 1].axhline(y=conf_bound_pacf, color='r', linestyle='--', alpha=0.7, label='95% CI')
        axes[0, 1].axhline(y=-conf_bound_pacf, color='r', linestyle='--', alpha=0.7)
        axes[0, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        axes[0, 1].set_title('Partial Autocorrelation Function (PACF)')
        axes[0, 1].set_xlabel('Lag')
        axes[0, 1].set_ylabel('PACF')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Method comparison for ACF
        acf_manual = self.acf_results['acf_manual']
        acf_fft = self.acf_results['acf_fft']
        
        axes[1, 0].plot(lags, acf_vals, 'b-', label='statsmodels', linewidth=2)
        axes[1, 0].plot(lags, acf_manual, 'r--', label='manual', linewidth=1.5, alpha=0.8)
        axes[1, 0].plot(lags, acf_fft, 'g:', label='FFT', linewidth=1.5, alpha=0.8)
        
        axes[1, 0].set_title('ACF Method Comparison')
        axes[1, 0].set_xlabel('Lag')
        axes[1, 0].set_ylabel('ACF')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Method comparison for PACF
        pacf_manual = self.pacf_results['pacf_manual']
        pacf_burg = self.pacf_results['pacf_burg']
        
        axes[1, 1].plot(lags, pacf_vals, 'b-', label='statsmodels', linewidth=2)
        axes[1, 1].plot(lags, pacf_manual, 'r--', label='manual', linewidth=1.5, alpha=0.8)
        if not np.all(np.isnan(pacf_burg)):
            axes[1, 1].plot(lags, pacf_burg, 'g:', label='Burg', linewidth=1.5, alpha=0.8)
        
        axes[1, 1].set_title('PACF Method Comparison')
        axes[1, 1].set_xlabel('Lag')
        axes[1, 1].set_ylabel('PACF')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Significance analysis
        acf_sig_lags = self.acf_results['diagnostics']['significant_lags']
        pacf_sig_lags = self.pacf_results['diagnostics']['significant_lags']
        
        # ACF significance heatmap
        sig_matrix_acf = np.zeros((1, len(lags)))
        for lag in acf_sig_lags:
            if lag < len(lags):
                sig_matrix_acf[0, lag] = 1
        
        im1 = axes[2, 0].imshow(sig_matrix_acf, cmap='RdBu_r', aspect='auto', 
                               extent=[0, len(lags)-1, -0.5, 0.5])
        axes[2, 0].set_title(f'ACF Significant Lags (Count: {len(acf_sig_lags)})')
        axes[2, 0].set_xlabel('Lag')
        axes[2, 0].set_yticks([])
        
        # PACF significance heatmap
        sig_matrix_pacf = np.zeros((1, len(lags)))
        for lag in pacf_sig_lags:
            if lag < len(lags):
                sig_matrix_pacf[0, lag] = 1
        
        im2 = axes[2, 1].imshow(sig_matrix_pacf, cmap='RdBu_r', aspect='auto',
                               extent=[0, len(lags)-1, -0.5, 0.5])
        axes[2, 1].set_title(f'PACF Significant Lags (Suggested AR: {self.pacf_results["diagnostics"]["suggested_ar_order"]})')
        axes[2, 1].set_xlabel('Lag')
        axes[2, 1].set_yticks([])
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def interpret_acf_pacf_patterns(self):
        """Provide interpretation of ACF and PACF patterns for model identification"""
        
        if not self.acf_results or not self.pacf_results:
            raise ValueError("Must calculate ACF and PACF first")
        
        print(f"\n=== ACF/PACF Pattern Interpretation ===")
        
        # Extract key information
        acf_decay = self.acf_results['diagnostics']['decay_pattern']
        acf_sig_count = self.acf_results['diagnostics']['significant_lags_count']
        pacf_sig_count = self.pacf_results['diagnostics']['significant_lags_count']
        suggested_ar_order = self.pacf_results['diagnostics']['suggested_ar_order']
        
        acf_vals = self.acf_results['acf_statsmodels']
        pacf_vals = self.pacf_results['pacf_statsmodels']
        
        print(f"Pattern Analysis:")
        print(f"  • ACF decay pattern: {acf_decay}")
        print(f"  • ACF significant lags: {acf_sig_count}")
        print(f"  • PACF significant lags: {pacf_sig_count}")
        print(f"  • Suggested AR order: {suggested_ar_order}")
        
        # Model identification logic
        model_suggestions = []
        
        # AR model identification
        if pacf_sig_count <= 3 and suggested_ar_order > 0:
            if acf_decay in ['exponential_decay', 'alternating_decay']:
                model_suggestions.append(f"AR({suggested_ar_order}) - PACF cuts off, ACF decays")
        
        # MA model identification
        if acf_sig_count <= 3 and acf_decay.startswith('cutoff_at_lag'):
            cutoff_lag = int(acf_decay.split('_')[-1])
            if pacf_sig_count > cutoff_lag:
                model_suggestions.append(f"MA({cutoff_lag}) - ACF cuts off, PACF decays")
        
        # ARMA model identification
        if acf_sig_count > 3 and pacf_sig_count > 3:
            if acf_decay in ['exponential_decay', 'alternating_decay']:
                max_ar = min(suggested_ar_order, 2)
                max_ma = min(acf_sig_count, 2)
                model_suggestions.append(f"ARMA({max_ar},{max_ma}) - Both ACF and PACF tail off")
        
        # White noise check
        ljung_box_pvalue = self.acf_results['diagnostics']['ljung_box_pvalue']
        if ljung_box_pvalue > 0.05 and acf_sig_count == 0:
            model_suggestions.append("White Noise - No significant autocorrelations")
        
        # Seasonal patterns
        seasonal_lags = []
        for period in [7, 12, 24, 52]:  # Common seasonal periods
            if period < len(acf_vals) and abs(acf_vals[period]) > 0.2:
                seasonal_lags.append(period)
        
        if seasonal_lags:
            model_suggestions.append(f"Potential seasonality at lags: {seasonal_lags}")
        
        print(f"\nModel Identification Suggestions:")
        if model_suggestions:
            for i, suggestion in enumerate(model_suggestions, 1):
                print(f"  {i}. {suggestion}")
        else:
            print("  • No clear model pattern identified")
            print("  • Consider data transformations or longer series")
        
        # Statistical test results
        print(f"\nStatistical Test Results:")
        print(f"  • Ljung-Box test p-value: {ljung_box_pvalue:.4f}")
        print(f"    {'Reject H₀: Series is not white noise' if ljung_box_pvalue < 0.05 else 'Fail to reject H₀: Series may be white noise'}")
        
        box_pierce_pvalue = self.acf_results['diagnostics']['box_pierce_pvalue']
        print(f"  • Box-Pierce test p-value: {box_pierce_pvalue:.4f}")
        
        return {
            'model_suggestions': model_suggestions,
            'pattern_analysis': {
                'acf_decay': acf_decay,
                'acf_significant_lags': acf_sig_count,
                'pacf_significant_lags': pacf_sig_count,
                'suggested_ar_order': suggested_ar_order
            },
            'statistical_tests': {
                'ljung_box_pvalue': ljung_box_pvalue,
                'box_pierce_pvalue': box_pierce_pvalue
            }
        }

def demonstrate_acf_pacf_analysis():
    """Comprehensive demonstration of ACF and PACF analysis"""
    
    print("=== ACF and PACF Analysis Demonstration ===")
    
    # Initialize analyzer
    analyzer = ACFPACFAnalysis()
    
    # Generate different types of synthetic data for demonstration
    np.random.seed(42)
    n = 200
    
    datasets = {}
    
    # 1. AR(2) process
    print(f"\n=== Analyzing AR(2) Process ===")
    ar_coeffs = [0.6, -0.2]
    ar_data = np.zeros(n)
    errors = np.random.normal(0, 1, n)
    
    for t in range(2, n):
        ar_data[t] = ar_coeffs[0] * ar_data[t-1] + ar_coeffs[1] * ar_data[t-2] + errors[t]
    
    datasets['AR(2)'] = ar_data
    
    # Analyze AR(2) process
    acf_results_ar = analyzer.calculate_acf_comprehensive(ar_data, nlags=30)
    pacf_results_ar = analyzer.calculate_pacf_comprehensive(ar_data, nlags=30)
    
    # Plot and interpret
    analyzer.plot_acf_pacf_comprehensive(title_prefix="AR(2) Process - ")
    interpretation_ar = analyzer.interpret_acf_pacf_patterns()
    
    # 2. MA(1) process
    print(f"\n=== Analyzing MA(1) Process ===")
    ma_coeff = 0.8
    ma_data = np.zeros(n)
    errors = np.random.normal(0, 1, n)
    
    for t in range(1, n):
        ma_data[t] = errors[t] + ma_coeff * errors[t-1]
    
    datasets['MA(1)'] = ma_data
    
    # Analyze MA(1) process
    acf_results_ma = analyzer.calculate_acf_comprehensive(ma_data, nlags=30)
    pacf_results_ma = analyzer.calculate_pacf_comprehensive(ma_data, nlags=30)
    
    analyzer.plot_acf_pacf_comprehensive(title_prefix="MA(1) Process - ")
    interpretation_ma = analyzer.interpret_acf_pacf_patterns()
    
    # 3. White noise
    print(f"\n=== Analyzing White Noise ===")
    white_noise = np.random.normal(0, 1, n)
    datasets['White Noise'] = white_noise
    
    acf_results_wn = analyzer.calculate_acf_comprehensive(white_noise, nlags=30)
    pacf_results_wn = analyzer.calculate_pacf_comprehensive(white_noise, nlags=30)
    
    analyzer.plot_acf_pacf_comprehensive(title_prefix="White Noise - ")
    interpretation_wn = analyzer.interpret_acf_pacf_patterns()
    
    # 4. Seasonal data
    print(f"\n=== Analyzing Seasonal Data ===")
    t = np.arange(n)
    seasonal_data = (10 * np.sin(2 * np.pi * t / 12) +  # Monthly seasonality
                    5 * np.cos(2 * np.pi * t / 7) +    # Weekly seasonality
                    np.random.normal(0, 2, n))         # Noise
    
    datasets['Seasonal'] = seasonal_data
    
    acf_results_seasonal = analyzer.calculate_acf_comprehensive(seasonal_data, nlags=50)
    pacf_results_seasonal = analyzer.calculate_pacf_comprehensive(seasonal_data, nlags=50)
    
    analyzer.plot_acf_pacf_comprehensive(title_prefix="Seasonal Data - ")
    interpretation_seasonal = analyzer.interpret_acf_pacf_patterns()
    
    # Summary comparison
    print(f"\n=== Summary Comparison ===")
    
    comparison_data = {
        'Process': [],
        'ACF Pattern': [],
        'PACF Pattern': [],
        'Model Suggestion': [],
        'Ljung-Box p-value': []
    }
    
    analyses = [
        ('AR(2)', interpretation_ar),
        ('MA(1)', interpretation_ma),
        ('White Noise', interpretation_wn),
        ('Seasonal', interpretation_seasonal)
    ]
    
    for process_name, interpretation in analyses:
        comparison_data['Process'].append(process_name)
        comparison_data['ACF Pattern'].append(interpretation['pattern_analysis']['acf_decay'])
        comparison_data['PACF Pattern'].append(f"{interpretation['pattern_analysis']['pacf_significant_lags']} sig lags")
        
        if interpretation['model_suggestions']:
            comparison_data['Model Suggestion'].append(interpretation['model_suggestions'][0])
        else:
            comparison_data['Model Suggestion'].append('None')
        
        comparison_data['Ljung-Box p-value'].append(f"{interpretation['statistical_tests']['ljung_box_pvalue']:.4f}")
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Theoretical insights
    print(f"\n=== Theoretical Insights ===")
    print("Key Patterns for Model Identification:")
    print("  • AR(p): ACF decays exponentially, PACF cuts off after lag p")
    print("  • MA(q): ACF cuts off after lag q, PACF decays exponentially")  
    print("  • ARMA(p,q): Both ACF and PACF tail off exponentially")
    print("  • White Noise: Both ACF and PACF are insignificant")
    print("  • Seasonal: Spikes at seasonal lags in ACF")
    
    print(f"\nStatistical Considerations:")
    print("  • Confidence bounds: ±1.96/√n for white noise null hypothesis")
    print("  • Ljung-Box test: Joint significance of multiple autocorrelations")
    print("  • Sample size effects: Larger samples give more reliable estimates")
    print("  • Overfitting risk: Don't rely solely on automated suggestions")
    
    return analyzer, datasets, comparison_df

# Execute demonstration
if __name__ == "__main__":
    analyzer, data, comparison = demonstrate_acf_pacf_analysis()
```

**Key Theoretical Insights:**

1. **ACF Properties**: Measures linear dependence between observations separated by k time periods; bounded between -1 and 1

2. **PACF Interpretation**: Captures direct correlation after removing intermediate variable effects; essential for AR order identification

3. **Model Identification**: Classic Box-Jenkins patterns provide systematic approach to model selection

4. **Statistical Inference**: Confidence bounds and portmanteau tests enable rigorous hypothesis testing

5. **Computational Methods**: Multiple algorithms (Yule-Walker, Burg, FFT) offer different trade-offs in accuracy and efficiency

6. **Pattern Recognition**: Exponential decay vs. sharp cutoffs distinguish between AR and MA components

7. **Practical Considerations**: Sample size, missing data, and outliers can significantly impact estimates

8. **Diagnostic Value**: Combined ACF/PACF analysis provides comprehensive view of temporal dependence structure

**Applications:**
- **ARIMA Modeling**: Primary tool for model order identification
- **Forecasting**: Understanding autocorrelation structure improves predictions
- **Signal Processing**: Detecting periodic patterns and dependencies
- **Quality Control**: Monitoring for autocorrelation in residuals
- **Economic Analysis**: Studying persistence in economic indicators

This completes all 6 coding questions in the time series folder with comprehensive theoretical treatments and practical implementations.

---

