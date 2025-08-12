# Time Series Interview Questions - Scenario_Based Questions

## Question 1

**Discuss the importance oflag selectioninARMA/ARIMAmodels.**

**Answer:**

**Theoretical Foundation:**

**Lag selection** in ARMA/ARIMA models is a **critical model specification decision** that fundamentally determines the **model's ability to capture temporal dependencies**, **forecast accuracy**, and **statistical validity**. The theoretical importance stems from the **mathematical representation** of autoregressive and moving average components and their **optimal parameterization**.

**Mathematical Framework:**

**ARIMA(p,d,q) Model Structure:**
```
∇^d X_t = φ₁∇^d X_{t-1} + ... + φₚ∇^d X_{t-p} + θ₁ε_{t-1} + ... + θₑε_{t-q} + ε_t

Where:
- p: Number of autoregressive lags
- d: Degree of differencing
- q: Number of moving average lags
- ∇: Difference operator
```

**Lag Selection Theoretical Criteria:**

1. **Information Criteria Framework:**
```
AIC(p,q) = log(σ̂²) + 2(p+q)/n
BIC(p,q) = log(σ̂²) + (p+q)log(n)/n
HQIC(p,q) = log(σ̂²) + 2(p+q)log(log(n))/n

Optimal order: argmin_{p,q} IC(p,q)
```

2. **Parsimony Principle:**
```
Occam's Razor: Prefer simpler models with fewer parameters
Balance: Model Complexity ↔ Goodness of Fit
Trade-off: Bias-Variance Decomposition
```

3. **Statistical Significance:**
```
t-statistic: t_j = φ̂_j / SE(φ̂_j)
Null hypothesis: H₀: φ_j = 0
Critical value: |t_j| > t_{α/2,n-p-q}
```

**Comprehensive Lag Selection Framework:**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.api import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from itertools import product
import warnings
warnings.filterwarnings('ignore')

class ARIMALagSelection:
    """
    Comprehensive ARIMA lag selection framework with theoretical validation
    and multiple selection criteria
    """
    
    def __init__(self):
        self.results = {}
        self.best_models = {}
        self.selection_summary = {}
        
    def theoretical_foundations(self):
        """Theoretical principles of lag selection"""
        
        principles = {
            'identification_methods': {
                'box_jenkins': 'ACF/PACF pattern recognition',
                'information_criteria': 'AIC, BIC, HQIC minimization',
                'cross_validation': 'Out-of-sample performance',
                'statistical_tests': 'Parameter significance testing'
            },
            'theoretical_guidance': {
                'ar_identification': 'PACF cuts off at lag p',
                'ma_identification': 'ACF cuts off at lag q',
                'mixed_arma': 'Both ACF and PACF tail off',
                'seasonal_patterns': 'Seasonal lags in ACF/PACF'
            },
            'model_selection_theory': {
                'underfitting': 'High bias, low variance, poor fit',
                'overfitting': 'Low bias, high variance, poor generalization',
                'optimal_complexity': 'Minimizes prediction error',
                'sample_size_effects': 'Larger samples support more parameters'
            },
            'statistical_properties': {
                'consistency': 'IC → optimal order as n → ∞',
                'efficiency': 'Asymptotic optimality properties',
                'robustness': 'Performance under model misspecification',
                'finite_sample': 'Small sample behavior'
            }
        }
        
        return principles
    
    def comprehensive_lag_selection(self, data, max_p=5, max_q=5, max_d=2, 
                                  seasonal=False, m=12, information_criteria=['aic', 'bic', 'hqic'],
                                  significance_level=0.05):
        """
        Comprehensive lag selection using multiple criteria
        
        Parameters:
        -----------
        data : array-like
            Time series data
        max_p : int, default=5
            Maximum AR order to consider
        max_q : int, default=5
            Maximum MA order to consider
        max_d : int, default=2
            Maximum differencing order
        seasonal : bool, default=False
            Include seasonal components
        m : int, default=12
            Seasonal period
        information_criteria : list
            Criteria to use for selection
        significance_level : float
            Significance level for parameter tests
        """
        
        print(f"=== Comprehensive ARIMA Lag Selection ===")
        print(f"Max AR order: {max_p}, Max MA order: {max_q}, Max d: {max_d}")
        
        # Determine optimal differencing order
        d_opt = self._determine_differencing_order(data, max_d)
        print(f"Optimal differencing order: {d_opt}")
        
        # Grid search over p and q
        results_df = []
        best_models = {}
        
        for p, q in product(range(max_p + 1), range(max_q + 1)):
            try:
                # Fit ARIMA model
                if seasonal:
                    model = ARIMA(data, order=(p, d_opt, q), 
                                seasonal_order=(1, 1, 1, m))
                else:
                    model = ARIMA(data, order=(p, d_opt, q))
                
                fitted_model = model.fit()
                
                # Extract information criteria
                result_dict = {
                    'p': p, 'q': q, 'd': d_opt,
                    'aic': fitted_model.aic,
                    'bic': fitted_model.bic,
                    'hqic': fitted_model.hqic,
                    'llf': fitted_model.llf,
                    'params': fitted_model.params,
                    'pvalues': fitted_model.pvalues,
                    'model': fitted_model
                }
                
                # Parameter significance test
                significant_params = np.sum(fitted_model.pvalues < significance_level)
                result_dict['significant_params'] = significant_params
                result_dict['param_significance_ratio'] = significant_params / len(fitted_model.params)
                
                # Residual diagnostics
                residuals = fitted_model.resid
                ljung_box_stat, ljung_box_pvalue = acorr_ljungbox(residuals, lags=10, return_df=False)[-2:]
                result_dict['ljung_box_pvalue'] = ljung_box_pvalue[-1]
                
                # Forecast accuracy (in-sample)
                result_dict['mae'] = np.mean(np.abs(residuals))
                result_dict['rmse'] = np.sqrt(np.mean(residuals**2))
                
                results_df.append(result_dict)
                
            except Exception as e:
                continue
        
        if not results_df:
            raise ValueError("No models could be fitted successfully")
        
        results_df = pd.DataFrame(results_df)
        
        # Find best models by each criterion
        for criterion in information_criteria:
            if criterion in results_df.columns:
                best_idx = results_df[criterion].idxmin()
                best_models[criterion] = {
                    'order': (results_df.loc[best_idx, 'p'], 
                             results_df.loc[best_idx, 'd'], 
                             results_df.loc[best_idx, 'q']),
                    'value': results_df.loc[best_idx, criterion],
                    'model': results_df.loc[best_idx, 'model']
                }
        
        # Consensus selection
        consensus_model = self._consensus_selection(results_df, information_criteria)
        
        # Store results
        self.results = results_df
        self.best_models = best_models
        self.selection_summary = {
            'consensus_model': consensus_model,
            'total_models_tested': len(results_df),
            'best_by_criterion': {k: v['order'] for k, v in best_models.items()}
        }
        
        # Print summary
        print(f"\nLag Selection Results:")
        print(f"  • Total models tested: {len(results_df)}")
        print(f"  • Best by AIC: {best_models.get('aic', {}).get('order', 'N/A')}")
        print(f"  • Best by BIC: {best_models.get('bic', {}).get('order', 'N/A')}")
        print(f"  • Best by HQIC: {best_models.get('hqic', {}).get('order', 'N/A')}")
        print(f"  • Consensus model: {consensus_model['order']}")
        
        return self.results, self.best_models, self.selection_summary
    
    def _determine_differencing_order(self, data, max_d):
        """Determine optimal differencing order using statistical tests"""
        
        print(f"\n--- Determining Differencing Order ---")
        
        current_data = data.copy()
        
        for d in range(max_d + 1):
            # ADF test for stationarity
            adf_stat, adf_pvalue, _, _, adf_critical, _ = adfuller(current_data, autolag='AIC')
            
            # KPSS test for trend stationarity
            try:
                kpss_stat, kpss_pvalue, _, kpss_critical = kpss(current_data, regression='ct')
            except:
                kpss_pvalue = np.nan
            
            print(f"  d={d}: ADF p-value={adf_pvalue:.4f}, KPSS p-value={kpss_pvalue:.4f}")
            
            # Check stationarity conditions
            if adf_pvalue < 0.05 and (np.isnan(kpss_pvalue) or kpss_pvalue > 0.05):
                print(f"  → Series is stationary at d={d}")
                return d
            
            # Apply one more difference
            if d < max_d:
                current_data = np.diff(current_data)
        
        print(f"  → Using maximum differencing order: {max_d}")
        return max_d
    
    def _consensus_selection(self, results_df, criteria):
        """Select consensus model based on multiple criteria"""
        
        # Rank models by each criterion
        ranks = pd.DataFrame()
        
        for criterion in criteria:
            if criterion in results_df.columns:
                ranks[f'{criterion}_rank'] = results_df[criterion].rank()
        
        # Calculate average rank
        ranks['avg_rank'] = ranks.mean(axis=1)
        
        # Find model with best average rank
        best_idx = ranks['avg_rank'].idxmin()
        
        consensus_model = {
            'order': (results_df.loc[best_idx, 'p'], 
                     results_df.loc[best_idx, 'd'], 
                     results_df.loc[best_idx, 'q']),
            'avg_rank': ranks.loc[best_idx, 'avg_rank'],
            'model': results_df.loc[best_idx, 'model']
        }
        
        return consensus_model
    
    def validate_lag_selection(self, data, train_ratio=0.8):
        """Validate lag selection using time series cross-validation"""
        
        print(f"\n=== Lag Selection Validation ===")
        
        if not self.best_models:
            raise ValueError("Must run lag selection first")
        
        # Split data
        split_point = int(len(data) * train_ratio)
        train_data = data[:split_point]
        test_data = data[split_point:]
        
        validation_results = {}
        
        for criterion, model_info in self.best_models.items():
            try:
                order = model_info['order']
                
                # Refit model on training data
                model = ARIMA(train_data, order=order)
                fitted_model = model.fit()
                
                # Generate forecasts
                forecast_steps = len(test_data)
                forecast = fitted_model.forecast(steps=forecast_steps)
                forecast_ci = fitted_model.get_forecast(steps=forecast_steps).conf_int()
                
                # Calculate validation metrics
                mae = np.mean(np.abs(test_data - forecast))
                rmse = np.sqrt(np.mean((test_data - forecast)**2))
                mape = np.mean(np.abs((test_data - forecast) / test_data)) * 100
                
                # Coverage probability for prediction intervals
                coverage = np.mean((test_data >= forecast_ci.iloc[:, 0]) & 
                                 (test_data <= forecast_ci.iloc[:, 1]))
                
                validation_results[criterion] = {
                    'order': order,
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'coverage_prob': coverage,
                    'forecast': forecast,
                    'forecast_ci': forecast_ci
                }
                
                print(f"{criterion.upper()} {order}:")
                print(f"  • MAE: {mae:.4f}")
                print(f"  • RMSE: {rmse:.4f}")
                print(f"  • MAPE: {mape:.2f}%")
                print(f"  • Coverage: {coverage:.2f}")
                
            except Exception as e:
                print(f"Validation failed for {criterion}: {e}")
                continue
        
        return validation_results
    
    def plot_lag_selection_analysis(self, figsize=(16, 12)):
        """Create comprehensive visualization of lag selection analysis"""
        
        if self.results.empty:
            raise ValueError("Must run lag selection first")
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('ARIMA Lag Selection Analysis', fontsize=16, fontweight='bold')
        
        # Information criteria comparison
        criteria = ['aic', 'bic', 'hqic']
        colors = ['blue', 'red', 'green']
        
        for i, (criterion, color) in enumerate(zip(criteria, colors)):
            if criterion in self.results.columns:
                pivot_data = self.results.pivot(index='p', columns='q', values=criterion)
                
                im = axes[0, i].imshow(pivot_data, cmap='viridis', aspect='auto')
                axes[0, i].set_title(f'{criterion.upper()} Values')
                axes[0, i].set_xlabel('MA Order (q)')
                axes[0, i].set_ylabel('AR Order (p)')
                
                # Mark best model
                best_p, best_q = self.best_models[criterion]['order'][0], self.best_models[criterion]['order'][2]
                axes[0, i].plot(best_q, best_p, 'r*', markersize=15, label='Best')
                axes[0, i].legend()
                
                plt.colorbar(im, ax=axes[0, i])
        
        # Model comparison
        axes[1, 0].plot(self.results['aic'], 'b-', label='AIC', alpha=0.7)
        axes[1, 0].plot(self.results['bic'], 'r-', label='BIC', alpha=0.7)
        axes[1, 0].plot(self.results['hqic'], 'g-', label='HQIC', alpha=0.7)
        axes[1, 0].set_title('Information Criteria Comparison')
        axes[1, 0].set_xlabel('Model Index')
        axes[1, 0].set_ylabel('IC Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Parameter significance
        axes[1, 1].scatter(self.results['p'] + self.results['q'], 
                          self.results['param_significance_ratio'], 
                          c=self.results['aic'], cmap='viridis', alpha=0.7)
        axes[1, 1].set_title('Parameter Significance vs Model Complexity')
        axes[1, 1].set_xlabel('Total Parameters (p+q)')
        axes[1, 1].set_ylabel('Significance Ratio')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Residual diagnostics
        axes[1, 2].scatter(self.results['rmse'], self.results['ljung_box_pvalue'], 
                          c=self.results['aic'], cmap='viridis', alpha=0.7)
        axes[1, 2].axhline(y=0.05, color='r', linestyle='--', alpha=0.7, label='α=0.05')
        axes[1, 2].set_title('Residual Quality vs Fit Quality')
        axes[1, 2].set_xlabel('RMSE')
        axes[1, 2].set_ylabel('Ljung-Box p-value')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig

def demonstrate_lag_selection():
    """Demonstrate comprehensive lag selection methodology"""
    
    print("=== ARIMA Lag Selection Demonstration ===")
    
    # Generate synthetic ARIMA data
    np.random.seed(42)
    n = 200
    
    # True ARIMA(2,1,1) process
    true_order = (2, 1, 1)
    phi = [0.6, -0.2]  # AR coefficients
    theta = [0.4]      # MA coefficients
    
    # Generate non-stationary series
    errors = np.random.normal(0, 1, n + 50)
    y = np.zeros(n + 50)
    
    # Apply ARIMA structure
    for t in range(2, n + 50):
        y[t] = phi[0] * (y[t-1] - y[t-2]) + phi[1] * (y[t-2] - y[t-3]) + \
               errors[t] + theta[0] * errors[t-1]
    
    # Add trend to make it I(1)
    trend = np.cumsum(np.random.normal(0, 0.1, n + 50))
    y = np.cumsum(y) + trend
    
    # Take final n observations
    data = y[-n:]
    
    print(f"Generated ARIMA{true_order} series with {n} observations")
    
    # Initialize lag selection framework
    selector = ARIMALagSelection()
    
    # Display theoretical foundations
    foundations = selector.theoretical_foundations()
    print(f"\nTheoretical Foundations:")
    for category, methods in foundations.items():
        print(f"  {category}:")
        for method, description in methods.items():
            print(f"    • {method}: {description}")
    
    # Perform comprehensive lag selection
    results, best_models, summary = selector.comprehensive_lag_selection(
        data, max_p=4, max_q=4, max_d=2, 
        information_criteria=['aic', 'bic', 'hqic']
    )
    
    # Validate selection
    validation_results = selector.validate_lag_selection(data, train_ratio=0.8)
    
    # Create visualizations
    selector.plot_lag_selection_analysis()
    
    # Summary analysis
    print(f"\n=== Lag Selection Summary ===")
    print(f"True model: ARIMA{true_order}")
    print(f"Selected models:")
    for criterion, order in summary['best_by_criterion'].items():
        print(f"  • {criterion.upper()}: ARIMA{order}")
    print(f"Consensus: ARIMA{summary['consensus_model']['order']}")
    
    # Theoretical insights
    print(f"\n=== Theoretical Insights ===")
    print(f"Lag Selection Principles:")
    print(f"  • Parsimony: Simpler models preferred (BIC penalty)")
    print(f"  • Fit Quality: Better fit preferred (AIC preference)")
    print(f"  • Statistical Significance: Parameters should be significant")
    print(f"  • Diagnostic Quality: Residuals should be white noise")
    print(f"  • Out-of-Sample Performance: Validation crucial")
    
    print(f"\nModel Selection Theory:")
    print(f"  • AIC: Asymptotically efficient, may overfit")
    print(f"  • BIC: Consistent, prefers parsimonious models")
    print(f"  • HQIC: Intermediate between AIC and BIC")
    print(f"  • Cross-validation: Most reliable for forecasting")
    
    return selector, data, validation_results

# Execute demonstration
if __name__ == "__main__":
    selector, data, validation = demonstrate_lag_selection()
```

**Critical Importance of Lag Selection:**

1. **Model Identification**: Correct lag specification ensures capturing all relevant temporal dependencies while avoiding overfitting

2. **Forecasting Accuracy**: Optimal lags directly impact prediction quality and confidence intervals

3. **Statistical Validity**: Proper specification ensures residuals satisfy white noise assumptions

4. **Economic Interpretation**: Lag structure often has meaningful economic or business interpretation

5. **Computational Efficiency**: Avoiding unnecessary parameters reduces estimation variance and computational cost

6. **Robustness**: Well-selected models perform better under structural changes and outliers

**Practical Guidelines:**
- Use multiple criteria (AIC, BIC, cross-validation)
- Consider theoretical constraints and domain knowledge  
- Validate using out-of-sample performance
- Test parameter significance and residual properties
- Balance model complexity with interpretability

---

## Question 2

**Discuss the use and considerations ofrolling-window analysisintime series.**

**Answer:**

**Theoretical Foundation:**

**Rolling-window analysis** represents a fundamental approach in time series analysis for handling **time-varying parameters**, **structural breaks**, and **non-stationary behavior**. The theoretical framework addresses the trade-off between **statistical efficiency** and **adaptability** in dynamic environments where model parameters evolve over time.

**Mathematical Framework:**

**Rolling Window Estimator:**
```
θ̂_t(w) = argmin Σ_{i=t-w+1}^t L(y_i, f(x_i; θ))

Where:
- w: Window size
- θ̂_t(w): Parameter estimate at time t using window size w
- L: Loss function
- f: Model function
```

**Window Size Selection Theory:**
```
Bias-Variance Trade-off:
- Large w: Lower variance, higher bias (if parameters change)
- Small w: Higher variance, lower bias (more adaptive)

Optimal Window Size: w* = argmin MSE(w) = Bias²(w) + Variance(w)
```

**Comprehensive Rolling Window Framework:**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class RollingWindowAnalysis:
    """
    Comprehensive rolling-window analysis framework for time series
    with theoretical foundations and adaptive methodologies
    """
    
    def __init__(self):
        self.results = {}
        self.window_performance = {}
        self.adaptive_results = {}
        
    def theoretical_foundations(self):
        """Theoretical foundations of rolling window analysis"""
        
        foundations = {
            'core_concepts': {
                'temporal_adaptation': 'Parameters adjust to local data patterns',
                'structural_breaks': 'Detect and adapt to regime changes',
                'non_stationarity': 'Handle evolving statistical properties',
                'forecasting_relevance': 'Use most recent relevant information'
            },
            'mathematical_principles': {
                'bias_variance_tradeoff': 'Balance between stability and adaptability',
                'estimation_efficiency': 'Trade sample size for temporal relevance',
                'convergence_properties': 'Asymptotic behavior under changing parameters',
                'robustness': 'Performance under model misspecification'
            },
            'window_selection_theory': {
                'fixed_window': 'Constant window size throughout analysis',
                'expanding_window': 'Growing window from initial point',
                'shrinking_window': 'Decreasing window approaching forecast',
                'adaptive_window': 'Data-driven window size selection'
            },
            'applications': {
                'parameter_estimation': 'Time-varying coefficient models',
                'forecasting': 'Adaptive prediction models',
                'risk_management': 'Dynamic risk measures',
                'change_detection': 'Structural break identification'
            }
        }
        
        return foundations
    
    def rolling_regression_analysis(self, y, X, window_sizes=[30, 60, 120], 
                                   min_window=20, forecast_horizon=1):
        """
        Comprehensive rolling regression analysis with multiple window sizes
        
        Parameters:
        -----------
        y : array-like
            Dependent variable
        X : array-like
            Independent variables
        window_sizes : list
            Different window sizes to evaluate
        min_window : int
            Minimum window size for analysis
        forecast_horizon : int
            Steps ahead for forecasting
        """
        
        print(f"=== Rolling Regression Analysis ===")
        print(f"Series length: {len(y)}, Window sizes: {window_sizes}")
        
        n = len(y)
        results = {}
        
        for window_size in window_sizes:
            print(f"\nAnalyzing window size: {window_size}")
            
            if window_size >= n - forecast_horizon:
                print(f"  Skipping: window too large for series")
                continue
            
            # Initialize storage
            coefficients = []
            predictions = []
            prediction_errors = []
            r_squared_values = []
            timestamps = []
            
            # Rolling window estimation
            for t in range(window_size, n - forecast_horizon + 1):
                try:
                    # Extract window data
                    y_window = y[t-window_size:t]
                    X_window = X[t-window_size:t]
                    
                    # Fit regression model
                    # Using simple linear regression for demonstration
                    if X_window.ndim == 1:
                        X_window = X_window.reshape(-1, 1)
                    
                    # Add intercept
                    X_window_int = np.column_stack([np.ones(len(X_window)), X_window])
                    
                    # OLS estimation
                    try:
                        beta = np.linalg.lstsq(X_window_int, y_window, rcond=None)[0]
                    except np.linalg.LinAlgError:
                        beta = np.full(X_window_int.shape[1], np.nan)
                    
                    coefficients.append(beta)
                    
                    # Forecast
                    if t + forecast_horizon - 1 < len(X):
                        X_forecast = X[t + forecast_horizon - 1]
                        if np.isscalar(X_forecast):
                            X_forecast = np.array([X_forecast])
                        X_forecast_int = np.concatenate([[1], X_forecast])
                        
                        y_pred = np.dot(X_forecast_int, beta)
                        y_actual = y[t + forecast_horizon - 1]
                        
                        predictions.append(y_pred)
                        prediction_errors.append(y_actual - y_pred)
                        
                        # Calculate R-squared for window
                        y_pred_insample = X_window_int @ beta
                        ss_res = np.sum((y_window - y_pred_insample)**2)
                        ss_tot = np.sum((y_window - np.mean(y_window))**2)
                        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                        r_squared_values.append(r_squared)
                    
                    timestamps.append(t)
                    
                except Exception as e:
                    continue
            
            # Store results
            results[window_size] = {
                'coefficients': np.array(coefficients),
                'predictions': np.array(predictions),
                'errors': np.array(prediction_errors),
                'r_squared': np.array(r_squared_values),
                'timestamps': np.array(timestamps),
                'rmse': np.sqrt(np.mean(np.array(prediction_errors)**2)) if prediction_errors else np.nan,
                'mae': np.mean(np.abs(prediction_errors)) if prediction_errors else np.nan
            }
            
            print(f"  • RMSE: {results[window_size]['rmse']:.4f}")
            print(f"  • MAE: {results[window_size]['mae']:.4f}")
            print(f"  • Mean R²: {np.mean(r_squared_values):.4f}")
        
        self.results['rolling_regression'] = results
        return results
    
    def rolling_arima_analysis(self, data, order=(1,1,1), window_sizes=[50, 100, 150]):
        """
        Rolling ARIMA analysis with multiple window sizes
        """
        
        print(f"\n=== Rolling ARIMA Analysis ===")
        print(f"ARIMA order: {order}, Window sizes: {window_sizes}")
        
        results = {}
        
        for window_size in window_sizes:
            print(f"\nAnalyzing ARIMA with window size: {window_size}")
            
            if window_size >= len(data) - 10:
                continue
            
            forecasts = []
            forecast_errors = []
            model_params = []
            aic_values = []
            
            for t in range(window_size, len(data)):
                try:
                    # Extract window
                    window_data = data[t-window_size:t]
                    
                    # Fit ARIMA model
                    model = ARIMA(window_data, order=order)
                    fitted_model = model.fit()
                    
                    # One-step forecast
                    forecast = fitted_model.forecast(steps=1)[0]
                    actual = data[t]
                    error = actual - forecast
                    
                    forecasts.append(forecast)
                    forecast_errors.append(error)
                    model_params.append(fitted_model.params)
                    aic_values.append(fitted_model.aic)
                    
                except Exception as e:
                    continue
            
            results[window_size] = {
                'forecasts': np.array(forecasts),
                'errors': np.array(forecast_errors),
                'parameters': model_params,
                'aic': np.array(aic_values),
                'rmse': np.sqrt(np.mean(np.array(forecast_errors)**2)),
                'mae': np.mean(np.abs(forecast_errors))
            }
            
            print(f"  • RMSE: {results[window_size]['rmse']:.4f}")
            print(f"  • MAE: {results[window_size]['mae']:.4f}")
            print(f"  • Mean AIC: {np.mean(aic_values):.2f}")
        
        self.results['rolling_arima'] = results
        return results
    
    def adaptive_window_selection(self, data, model_type='arima', criterion='aic',
                                 min_window=30, max_window=200, step=10):
        """
        Adaptive window size selection based on information criteria
        """
        
        print(f"\n=== Adaptive Window Selection ===")
        print(f"Model: {model_type}, Criterion: {criterion}")
        
        window_sizes = range(min_window, min(max_window, len(data)//2), step)
        performance_scores = []
        
        for window_size in window_sizes:
            scores = []
            
            # Cross-validation-like approach
            test_points = range(window_size, len(data), max(1, len(data)//20))
            
            for t in test_points:
                try:
                    if model_type == 'arima':
                        window_data = data[t-window_size:t]
                        model = ARIMA(window_data, order=(1,1,1))
                        fitted_model = model.fit()
                        
                        if criterion == 'aic':
                            score = fitted_model.aic
                        elif criterion == 'bic':
                            score = fitted_model.bic
                        else:  # forecast error
                            if t < len(data):
                                forecast = fitted_model.forecast(steps=1)[0]
                                score = (data[t] - forecast)**2
                    
                    scores.append(score)
                    
                except Exception as e:
                    continue
            
            if scores:
                avg_score = np.mean(scores)
                performance_scores.append(avg_score)
            else:
                performance_scores.append(np.inf)
        
        # Find optimal window size
        if criterion in ['aic', 'bic']:
            optimal_idx = np.argmin(performance_scores)
        else:  # forecast error - minimize
            optimal_idx = np.argmin(performance_scores)
        
        optimal_window = list(window_sizes)[optimal_idx]
        
        self.adaptive_results = {
            'window_sizes': list(window_sizes),
            'performance_scores': performance_scores,
            'optimal_window': optimal_window,
            'criterion': criterion
        }
        
        print(f"Optimal window size: {optimal_window}")
        return optimal_window, performance_scores
    
    def structural_break_detection(self, data, window_size=50, threshold=2.0):
        """
        Detect structural breaks using rolling window analysis
        """
        
        print(f"\n=== Structural Break Detection ===")
        print(f"Window size: {window_size}, Threshold: {threshold}")
        
        n = len(data)
        break_statistics = []
        break_points = []
        
        for t in range(window_size, n - window_size):
            try:
                # Pre-break and post-break windows
                pre_window = data[t-window_size:t]
                post_window = data[t:t+window_size]
                
                # Statistical tests for break
                # Welch's t-test for mean difference
                t_stat, p_value = stats.ttest_ind(pre_window, post_window, equal_var=False)
                
                # F-test for variance difference
                f_stat = np.var(post_window, ddof=1) / np.var(pre_window, ddof=1)
                
                break_statistics.append({
                    'time': t,
                    't_statistic': abs(t_stat),
                    'p_value': p_value,
                    'f_statistic': f_stat,
                    'break_indicator': abs(t_stat) > threshold
                })
                
                if abs(t_stat) > threshold:
                    break_points.append(t)
                    
            except Exception as e:
                continue
        
        print(f"Detected {len(break_points)} potential structural breaks")
        if break_points:
            print(f"Break points at: {break_points}")
        
        self.results['structural_breaks'] = {
            'statistics': break_statistics,
            'break_points': break_points,
            'threshold': threshold
        }
        
        return break_points, break_statistics
    
    def plot_rolling_analysis(self, figsize=(16, 12)):
        """Create comprehensive visualization of rolling window analysis"""
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Rolling Window Analysis Results', fontsize=16, fontweight='bold')
        
        # Rolling regression coefficients
        if 'rolling_regression' in self.results:
            for i, (window_size, results) in enumerate(self.results['rolling_regression'].items()):
                if i < 3:  # Plot first 3 window sizes
                    coeffs = results['coefficients']
                    timestamps = results['timestamps']
                    
                    if coeffs.shape[1] > 1:  # Plot first coefficient (excluding intercept)
                        axes[0, i].plot(timestamps, coeffs[:, 1], label=f'Window {window_size}')
                        axes[0, i].set_title(f'Rolling Coefficient (Window={window_size})')
                        axes[0, i].set_xlabel('Time')
                        axes[0, i].set_ylabel('Coefficient')
                        axes[0, i].grid(True, alpha=0.3)
        
        # ARIMA parameter evolution
        if 'rolling_arima' in self.results:
            for i, (window_size, results) in enumerate(self.results['rolling_arima'].items()):
                if i < 3:
                    aic_values = results['aic']
                    axes[1, i].plot(aic_values, label=f'AIC (Window {window_size})')
                    axes[1, i].set_title(f'AIC Evolution (Window={window_size})')
                    axes[1, i].set_xlabel('Time')
                    axes[1, i].set_ylabel('AIC')
                    axes[1, i].grid(True, alpha=0.3)
        
        # Adaptive window selection
        if hasattr(self, 'adaptive_results') and self.adaptive_results:
            window_sizes = self.adaptive_results['window_sizes']
            scores = self.adaptive_results['performance_scores']
            optimal_window = self.adaptive_results['optimal_window']
            
            axes[1, 2].plot(window_sizes, scores, 'b-', linewidth=2)
            axes[1, 2].axvline(x=optimal_window, color='r', linestyle='--', 
                              label=f'Optimal: {optimal_window}')
            axes[1, 2].set_title('Adaptive Window Selection')
            axes[1, 2].set_xlabel('Window Size')
            axes[1, 2].set_ylabel('Performance Score')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig

def demonstrate_rolling_window_analysis():
    """Demonstrate comprehensive rolling window methodologies"""
    
    print("=== Rolling Window Analysis Demonstration ===")
    
    # Generate synthetic data with structural breaks
    np.random.seed(42)
    n = 300
    
    # Create time series with regime changes
    t = np.arange(n)
    
    # Base trend with structural breaks
    trend1 = 0.02 * t[:100]  # Low growth regime
    trend2 = 0.02 * t[100:200] + 0.05 * (t[100:200] - 100)  # High growth regime
    trend3 = 0.02 * t[200:] + 0.05 * 100 - 0.01 * (t[200:] - 200)  # Decline regime
    trend = np.concatenate([trend1, trend2, trend3])
    
    # Add autoregressive component with changing parameters
    ar_coeff1 = 0.7  # First regime
    ar_coeff2 = 0.3  # Second regime
    ar_coeff3 = 0.9  # Third regime
    
    y = np.zeros(n)
    errors = np.random.normal(0, 1, n)
    
    for i in range(1, n):
        if i < 100:
            ar_coeff = ar_coeff1
        elif i < 200:
            ar_coeff = ar_coeff2
        else:
            ar_coeff = ar_coeff3
        
        y[i] = ar_coeff * y[i-1] + trend[i] + errors[i]
    
    # Create exogenous variable
    X = 0.5 * y[:-1] + np.random.normal(0, 0.5, n-1)
    y_reg = y[1:]  # Dependent variable for regression
    
    print(f"Generated time series with {n} observations and 3 regimes")
    
    # Initialize rolling window analyzer
    analyzer = RollingWindowAnalysis()
    
    # Display theoretical foundations
    foundations = analyzer.theoretical_foundations()
    print(f"\nTheoretical Foundations:")
    for category, concepts in foundations.items():
        print(f"  {category}:")
        for concept, description in concepts.items():
            print(f"    • {concept}: {description}")
    
    # Rolling regression analysis
    reg_results = analyzer.rolling_regression_analysis(
        y_reg, X, window_sizes=[30, 60, 90], forecast_horizon=1
    )
    
    # Rolling ARIMA analysis
    arima_results = analyzer.rolling_arima_analysis(
        y, order=(1,0,0), window_sizes=[50, 75, 100]
    )
    
    # Adaptive window selection
    optimal_window, scores = analyzer.adaptive_window_selection(
        y, model_type='arima', criterion='aic', min_window=30, max_window=150
    )
    
    # Structural break detection
    break_points, break_stats = analyzer.structural_break_detection(
        y, window_size=40, threshold=2.0
    )
    
    # Create visualizations
    analyzer.plot_rolling_analysis()
    
    # Performance comparison
    print(f"\n=== Performance Comparison ===")
    print(f"Rolling Regression Results:")
    for window_size, results in reg_results.items():
        print(f"  Window {window_size}: RMSE={results['rmse']:.4f}, MAE={results['mae']:.4f}")
    
    print(f"\nRolling ARIMA Results:")
    for window_size, results in arima_results.items():
        print(f"  Window {window_size}: RMSE={results['rmse']:.4f}, MAE={results['mae']:.4f}")
    
    # Theoretical insights
    print(f"\n=== Theoretical Insights ===")
    print(f"Rolling Window Considerations:")
    print(f"  • Window Size Trade-off: Stability vs Adaptability")
    print(f"  • Structural Break Detection: Parameter instability signals")
    print(f"  • Forecast Performance: Recent data more relevant")
    print(f"  • Computational Efficiency: Balance accuracy and speed")
    
    print(f"\nPractical Guidelines:")
    print(f"  • Financial data: Shorter windows (20-60 observations)")
    print(f"  • Economic data: Longer windows (60-120 observations)")
    print(f"  • High-frequency data: Very short windows (10-30)")
    print(f"  • Structural instability: Adaptive window selection")
    
    return analyzer, y, X, break_points

# Execute demonstration
if __name__ == "__main__":
    analyzer, data, X, breaks = demonstrate_rolling_window_analysis()
```

**Key Considerations in Rolling-Window Analysis:**

1. **Window Size Selection**: Critical trade-off between bias and variance; smaller windows adapt faster but with higher uncertainty

2. **Structural Stability**: Essential for detecting regime changes and parameter instability

3. **Forecasting Relevance**: Recent observations often more informative for prediction

4. **Computational Efficiency**: Balance between accuracy and computational cost

5. **Statistical Properties**: Need sufficient observations for reliable parameter estimation

6. **Model Specification**: Window size may affect optimal model complexity

---

## Question 3

**Discuss the advantage of usingstate-space modelsand theKalman filterfortime series analysis.**

**Answer:**

**Theoretical Foundation:**

**State-space models** and the **Kalman filter** represent a unified framework for handling **unobserved components**, **missing data**, **parameter estimation**, and **optimal filtering** in time series analysis. The theoretical advantages stem from the **recursive estimation structure**, **optimal statistical properties**, and **flexibility in model specification**.

**Mathematical Framework:**

**State-Space Representation:**
```
State Equation:    α_{t+1} = T_t α_t + R_t η_t     (System dynamics)
Observation Eq:    y_t = Z_t α_t + ε_t              (Measurement relation)

Where:
- α_t: State vector (unobserved)
- y_t: Observation vector
- T_t: Transition matrix
- Z_t: Design matrix  
- η_t ~ N(0, Q_t): State noise
- ε_t ~ N(0, H_t): Observation noise
```

**Kalman Filter Recursions:**
```
Prediction Step:
α_{t|t-1} = T_t α_{t-1|t-1}
P_{t|t-1} = T_t P_{t-1|t-1} T_t' + R_t Q_t R_t'

Update Step:
K_t = P_{t|t-1} Z_t' (Z_t P_{t|t-1} Z_t' + H_t)^{-1}
α_{t|t} = α_{t|t-1} + K_t (y_t - Z_t α_{t|t-1})
P_{t|t} = (I - K_t Z_t) P_{t|t-1}
```

**Comprehensive State-Space Framework:**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg
from statsmodels.tsa.statespace import MLEModel, sarimax
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

class StateSpaceFramework:
    """
    Comprehensive state-space modeling framework with Kalman filtering
    and advanced estimation techniques
    """
    
    def __init__(self):
        self.models = {}
        self.filter_results = {}
        self.smoother_results = {}
        
    def theoretical_foundations(self):
        """Theoretical advantages of state-space models"""
        
        advantages = {
            'unobserved_components': {
                'trend_extraction': 'Separate trends from cycles and noise',
                'seasonal_modeling': 'Time-varying seasonal patterns',
                'missing_data': 'Optimal handling of irregular observations',
                'latent_factors': 'Model hidden driving forces'
            },
            'statistical_optimality': {
                'mse_optimality': 'Minimum mean squared error estimation',
                'likelihood_methods': 'Maximum likelihood parameter estimation',
                'recursive_structure': 'Sequential processing of observations',
                'prediction_intervals': 'Exact confidence bounds'
            },
            'flexibility': {
                'nonlinear_extensions': 'Extended and Unscented Kalman filters',
                'multivariate_models': 'Vector state-space systems',
                'time_varying_parameters': 'Adaptive coefficient estimation',
                'regime_switching': 'Multiple model frameworks'
            },
            'computational_advantages': {
                'recursive_estimation': 'O(n) complexity vs O(n³) for batch',
                'real_time_processing': 'Online estimation and forecasting',
                'memory_efficiency': 'Fixed memory requirements',
                'parallel_processing': 'Independent state updates'
            }
        }
        
        return advantages
    
    def local_level_model(self, data, initial_variance=1.0):
        """
        Local level model (random walk with noise)
        
        State equation: α_{t+1} = α_t + η_t
        Obs equation: y_t = α_t + ε_t
        """
        
        print(f"=== Local Level Model ===")
        
        n = len(data)
        
        # Model specification
        # State transition matrix
        T = np.array([[1.0]])
        
        # Design matrix
        Z = np.array([[1.0]])
        
        # State noise covariance
        Q = np.array([[initial_variance]])
        
        # Observation noise covariance
        H = np.array([[initial_variance]])
        
        # Initial state
        a0 = np.array([data[0]])
        P0 = np.array([[initial_variance * 10]])
        
        # Run Kalman filter
        filter_results = self._kalman_filter(data, T, Z, Q, H, a0, P0)
        
        # Run Kalman smoother
        smoother_results = self._kalman_smoother(filter_results, T, Z, Q)
        
        # Estimate parameters using MLE
        optimal_params = self._estimate_local_level_mle(data)
        
        print(f"Estimated signal variance: {optimal_params['signal_var']:.6f}")
        print(f"Estimated noise variance: {optimal_params['noise_var']:.6f}")
        print(f"Signal-to-noise ratio: {optimal_params['signal_var']/optimal_params['noise_var']:.3f}")
        
        self.models['local_level'] = {
            'filter': filter_results,
            'smoother': smoother_results,
            'parameters': optimal_params,
            'specification': {'T': T, 'Z': Z, 'Q': Q, 'H': H}
        }
        
        return filter_results, smoother_results, optimal_params
    
    def local_trend_model(self, data):
        """
        Local linear trend model
        
        State: [level, slope]'
        α_{t+1} = [1 1; 0 1] α_t + η_t
        y_t = [1 0] α_t + ε_t
        """
        
        print(f"\n=== Local Linear Trend Model ===")
        
        n = len(data)
        
        # State transition matrix
        T = np.array([[1.0, 1.0],
                     [0.0, 1.0]])
        
        # Design matrix
        Z = np.array([[1.0, 0.0]])
        
        # State noise covariance (level and slope disturbances)
        Q = np.array([[1.0, 0.0],
                     [0.0, 0.1]])
        
        # Observation noise covariance
        H = np.array([[1.0]])
        
        # Initial state [level, slope]
        # Estimate initial slope using first difference
        initial_slope = data[1] - data[0] if len(data) > 1 else 0
        a0 = np.array([data[0], initial_slope])
        P0 = np.array([[10.0, 0.0],
                      [0.0, 1.0]])
        
        # Apply Kalman filter and smoother
        filter_results = self._kalman_filter(data, T, Z, Q, H, a0, P0)
        smoother_results = self._kalman_smoother(filter_results, T, Z, Q)
        
        # Extract trend and slope estimates
        trend_estimates = smoother_results['states'][:, 0]
        slope_estimates = smoother_results['states'][:, 1]
        
        print(f"Final trend level: {trend_estimates[-1]:.4f}")
        print(f"Final slope: {slope_estimates[-1]:.6f}")
        print(f"Average absolute slope change: {np.mean(np.abs(np.diff(slope_estimates))):.6f}")
        
        self.models['local_trend'] = {
            'filter': filter_results,
            'smoother': smoother_results,
            'trend': trend_estimates,
            'slope': slope_estimates,
            'specification': {'T': T, 'Z': Z, 'Q': Q, 'H': H}
        }
        
        return trend_estimates, slope_estimates
    
    def seasonal_state_space_model(self, data, seasonal_periods=12):
        """
        Seasonal state-space model with trend and seasonal components
        """
        
        print(f"\n=== Seasonal State-Space Model ===")
        print(f"Seasonal periods: {seasonal_periods}")
        
        # State dimension: trend (2) + seasonal (s-1)
        state_dim = 2 + seasonal_periods - 1
        
        # Build transition matrix
        T = np.zeros((state_dim, state_dim))
        
        # Trend component [level, slope]
        T[0, 0] = 1.0  # level
        T[0, 1] = 1.0  # add slope to level
        T[1, 1] = 1.0  # slope persistence
        
        # Seasonal component (sum to zero constraint)
        for i in range(seasonal_periods - 1):
            if i == 0:
                T[2, 2:2+seasonal_periods-1] = -1.0
            else:
                T[2+i, 2+i-1] = 1.0
        
        # Design matrix
        Z = np.zeros((1, state_dim))
        Z[0, 0] = 1.0  # level
        Z[0, 2] = 1.0  # first seasonal state
        
        # State noise covariance
        Q = np.eye(state_dim) * 0.1
        Q[0, 0] = 1.0    # level variance
        Q[1, 1] = 0.01   # slope variance
        Q[2, 2] = 0.5    # seasonal variance
        
        # Observation noise
        H = np.array([[1.0]])
        
        # Initial state
        a0 = np.zeros(state_dim)
        a0[0] = data[0]  # initial level
        
        # Initial state covariance
        P0 = np.eye(state_dim) * 10.0
        
        # Apply filters
        filter_results = self._kalman_filter(data, T, Z, Q, H, a0, P0)
        smoother_results = self._kalman_smoother(filter_results, T, Z, Q)
        
        # Extract components
        trend_component = smoother_results['states'][:, 0]
        seasonal_component = smoother_results['states'][:, 2]
        slope_component = smoother_results['states'][:, 1]
        
        self.models['seasonal'] = {
            'filter': filter_results,
            'smoother': smoother_results,
            'trend': trend_component,
            'seasonal': seasonal_component,
            'slope': slope_component,
            'specification': {'T': T, 'Z': Z, 'Q': Q, 'H': H}
        }
        
        return trend_component, seasonal_component, slope_component
    
    def _kalman_filter(self, y, T, Z, Q, H, a0, P0):
        """
        Kalman filter implementation
        """
        
        n = len(y)
        state_dim = T.shape[0]
        
        # Storage
        a_pred = np.zeros((n, state_dim))    # Predicted states
        P_pred = np.zeros((n, state_dim, state_dim))  # Predicted covariances
        a_upd = np.zeros((n, state_dim))     # Updated states
        P_upd = np.zeros((n, state_dim, state_dim))   # Updated covariances
        v = np.zeros(n)                      # Prediction errors
        F = np.zeros(n)                      # Prediction error variances
        K = np.zeros((n, state_dim))         # Kalman gains
        
        # Initial values
        a_upd_prev = a0
        P_upd_prev = P0
        
        for t in range(n):
            # Prediction step
            a_pred[t] = T @ a_upd_prev
            P_pred[t] = T @ P_upd_prev @ T.T + Q
            
            # Prediction error
            v[t] = y[t] - Z @ a_pred[t]
            F[t] = Z @ P_pred[t] @ Z.T + H[0, 0]
            
            # Kalman gain
            if F[t] > 1e-8:
                K[t] = (P_pred[t] @ Z.T) / F[t]
            else:
                K[t] = np.zeros(state_dim)
            
            # Update step
            a_upd[t] = a_pred[t] + K[t] * v[t]
            P_upd[t] = P_pred[t] - np.outer(K[t], Z @ P_pred[t])
            
            # Store for next iteration
            a_upd_prev = a_upd[t]
            P_upd_prev = P_upd[t]
        
        return {
            'states_pred': a_pred,
            'states_upd': a_upd,
            'covariances_pred': P_pred,
            'covariances_upd': P_upd,
            'innovations': v,
            'innovation_variances': F,
            'kalman_gains': K,
            'loglikelihood': -0.5 * np.sum(np.log(2 * np.pi * F) + v**2 / F)
        }
    
    def _kalman_smoother(self, filter_results, T, Z, Q):
        """
        Kalman smoother (RTS smoother)
        """
        
        n = filter_results['states_upd'].shape[0]
        state_dim = T.shape[0]
        
        # Initialize with filtered estimates
        a_smooth = filter_results['states_upd'].copy()
        P_smooth = filter_results['covariances_upd'].copy()
        
        # Backward pass
        for t in range(n - 2, -1, -1):
            # Smoother gain
            if np.linalg.det(filter_results['covariances_pred'][t + 1]) > 1e-10:
                A = filter_results['covariances_upd'][t] @ T.T @ \
                    np.linalg.inv(filter_results['covariances_pred'][t + 1])
            else:
                A = np.zeros((state_dim, state_dim))
            
            # Smoothed state
            a_smooth[t] = filter_results['states_upd'][t] + \
                         A @ (a_smooth[t + 1] - filter_results['states_pred'][t + 1])
            
            # Smoothed covariance
            P_smooth[t] = filter_results['covariances_upd'][t] + \
                         A @ (P_smooth[t + 1] - filter_results['covariances_pred'][t + 1]) @ A.T
        
        return {
            'states': a_smooth,
            'covariances': P_smooth
        }
    
    def _estimate_local_level_mle(self, data):
        """
        Maximum likelihood estimation for local level model
        """
        
        def negative_loglikelihood(params):
            signal_var, noise_var = np.exp(params)  # Ensure positivity
            
            if signal_var <= 0 or noise_var <= 0:
                return 1e10
            
            T = np.array([[1.0]])
            Z = np.array([[1.0]])
            Q = np.array([[signal_var]])
            H = np.array([[noise_var]])
            a0 = np.array([data[0]])
            P0 = np.array([[signal_var * 10]])
            
            try:
                filter_results = self._kalman_filter(data, T, Z, Q, H, a0, P0)
                return -filter_results['loglikelihood']
            except:
                return 1e10
        
        # Optimize
        from scipy.optimize import minimize
        initial_params = [0.0, 0.0]  # log(1.0) for both variances
        result = minimize(negative_loglikelihood, initial_params, method='BFGS')
        
        optimal_signal_var, optimal_noise_var = np.exp(result.x)
        
        return {
            'signal_var': optimal_signal_var,
            'noise_var': optimal_noise_var,
            'loglikelihood': -result.fun,
            'convergence': result.success
        }
    
    def forecast_state_space(self, model_name, horizon=10):
        """
        Generate forecasts using state-space model
        """
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        filter_results = model['filter']
        spec = model['specification']
        
        # Get final state and covariance
        final_state = filter_results['states_upd'][-1]
        final_cov = filter_results['covariances_upd'][-1]
        
        # Initialize forecast arrays
        forecasts = np.zeros(horizon)
        forecast_vars = np.zeros(horizon)
        
        # Forward simulation
        state = final_state.copy()
        cov = final_cov.copy()
        
        for h in range(horizon):
            # Predict state
            state = spec['T'] @ state
            cov = spec['T'] @ cov @ spec['T'].T + spec['Q']
            
            # Predict observation
            forecasts[h] = spec['Z'] @ state
            forecast_vars[h] = spec['Z'] @ cov @ spec['Z'].T + spec['H'][0, 0]
        
        # Confidence intervals
        forecast_se = np.sqrt(forecast_vars)
        lower_ci = forecasts - 1.96 * forecast_se
        upper_ci = forecasts + 1.96 * forecast_se
        
        return {
            'forecasts': forecasts,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci,
            'forecast_se': forecast_se
        }
    
    def plot_state_space_results(self, data, figsize=(16, 12)):
        """
        Comprehensive visualization of state-space results
        """
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('State-Space Model Results', fontsize=16, fontweight='bold')
        
        time_index = np.arange(len(data))
        
        # Local level model
        if 'local_level' in self.models:
            smooth_states = self.models['local_level']['smoother']['states'][:, 0]
            axes[0, 0].plot(time_index, data, 'b-', alpha=0.7, label='Observed')
            axes[0, 0].plot(time_index, smooth_states, 'r-', linewidth=2, label='Smoothed Level')
            axes[0, 0].set_title('Local Level Model')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Local trend model
        if 'local_trend' in self.models:
            trend = self.models['local_trend']['trend']
            slope = self.models['local_trend']['slope']
            axes[0, 1].plot(time_index, data, 'b-', alpha=0.7, label='Observed')
            axes[0, 1].plot(time_index, trend, 'r-', linewidth=2, label='Trend')
            axes[0, 1].set_title('Local Linear Trend Model')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot slope
            ax_slope = axes[0, 1].twinx()
            ax_slope.plot(time_index, slope, 'g--', alpha=0.7, label='Slope')
            ax_slope.set_ylabel('Slope', color='g')
        
        # Seasonal model
        if 'seasonal' in self.models:
            trend = self.models['seasonal']['trend']
            seasonal = self.models['seasonal']['seasonal']
            
            axes[1, 0].plot(time_index, data, 'b-', alpha=0.7, label='Observed')
            axes[1, 0].plot(time_index, trend, 'r-', linewidth=2, label='Trend')
            axes[1, 0].plot(time_index, seasonal, 'g-', alpha=0.7, label='Seasonal')
            axes[1, 0].set_title('Seasonal Decomposition')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Innovation diagnostics
        if 'local_level' in self.models:
            innovations = self.models['local_level']['filter']['innovations']
            axes[1, 1].plot(time_index, innovations, 'b-', alpha=0.7)
            axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.7)
            axes[1, 1].set_title('Filter Innovations')
            axes[1, 1].set_xlabel('Time')
            axes[1, 1].set_ylabel('Innovation')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig

def demonstrate_state_space_advantages():
    """
    Demonstrate advantages of state-space models and Kalman filtering
    """
    
    print("=== State-Space Models and Kalman Filter Demonstration ===")
    
    # Generate complex synthetic data
    np.random.seed(42)
    n = 200
    
    # True components
    t = np.arange(n)
    
    # Trend component (local linear trend)
    true_trend = 5 + 0.1 * t + 0.02 * t**1.5 / 100
    
    # Seasonal component
    seasonal_pattern = 2 * np.sin(2 * np.pi * t / 12) + 1.5 * np.cos(2 * np.pi * t / 12)
    
    # Irregular component
    irregular = np.random.normal(0, 1, n)
    
    # Missing data pattern
    missing_indices = np.random.choice(n, size=n//10, replace=False)
    
    # Combine components
    observed = true_trend + seasonal_pattern + irregular
    observed_with_missing = observed.copy()
    observed_with_missing[missing_indices] = np.nan
    
    print(f"Generated time series: n={n}, missing={len(missing_indices)} observations")
    
    # Initialize state-space framework
    framework = StateSpaceFramework()
    
    # Display theoretical advantages
    advantages = framework.theoretical_foundations()
    print(f"\nTheoretical Advantages:")
    for category, items in advantages.items():
        print(f"  {category}:")
        for advantage, description in items.items():
            print(f"    • {advantage}: {description}")
    
    # Use complete data for initial demonstration
    data_complete = observed[~np.isnan(observed_with_missing)]
    
    # Fit different state-space models
    print(f"\n=== Model Fitting ===")
    
    # Local level model
    filter_ll, smoother_ll, params_ll = framework.local_level_model(data_complete)
    
    # Local trend model
    trend_estimates, slope_estimates = framework.local_trend_model(data_complete)
    
    # Seasonal model
    trend_seasonal, seasonal_comp, slope_seasonal = framework.seasonal_state_space_model(
        data_complete, seasonal_periods=12
    )
    
    # Generate forecasts
    print(f"\n=== Forecasting ===")
    
    for model_name in ['local_level', 'local_trend', 'seasonal']:
        if model_name in framework.models:
            forecast_results = framework.forecast_state_space(model_name, horizon=12)
            print(f"{model_name.title()} Model Forecast:")
            print(f"  • Next period: {forecast_results['forecasts'][0]:.3f} ± {1.96*forecast_results['forecast_se'][0]:.3f}")
            print(f"  • 12-step ahead: {forecast_results['forecasts'][-1]:.3f} ± {1.96*forecast_results['forecast_se'][-1]:.3f}")
    
    # Create comprehensive visualizations
    framework.plot_state_space_results(data_complete)
    
    # Compare with traditional methods
    print(f"\n=== Comparison with Traditional Methods ===")
    
    # Classical decomposition
    ts_data = pd.Series(data_complete, index=pd.date_range('2000-01-01', periods=len(data_complete), freq='M'))
    classical_decomp = seasonal_decompose(ts_data, model='additive', period=12)
    
    # Compare trend estimates
    ss_trend = framework.models['seasonal']['trend']
    classical_trend = classical_decomp.trend.dropna().values
    
    # Align lengths for comparison
    min_len = min(len(ss_trend), len(classical_trend))
    trend_corr = np.corrcoef(ss_trend[:min_len], classical_trend[:min_len])[0, 1]
    
    print(f"Trend Correlation (State-Space vs Classical): {trend_corr:.4f}")
    
    # Compare seasonal estimates
    ss_seasonal = framework.models['seasonal']['seasonal']
    classical_seasonal = classical_decomp.seasonal.dropna().values
    
    min_len_seasonal = min(len(ss_seasonal), len(classical_seasonal))
    seasonal_corr = np.corrcoef(ss_seasonal[:min_len_seasonal], 
                               classical_seasonal[:min_len_seasonal])[0, 1]
    
    print(f"Seasonal Correlation (State-Space vs Classical): {seasonal_corr:.4f}")
    
    # Demonstrate missing data handling
    print(f"\n=== Missing Data Handling ===")
    
    # Handle missing data with Kalman filter (conceptual demonstration)
    print(f"State-space models naturally handle missing observations")
    print(f"Classical methods require interpolation or complete data")
    print(f"Missing data points: {len(missing_indices)} out of {n}")
    
    # Theoretical insights
    print(f"\n=== Key Theoretical Insights ===")
    print(f"State-Space Advantages:")
    print(f"  • Optimal Filtering: MSE-optimal state estimates")
    print(f"  • Uncertainty Quantification: Exact confidence intervals")
    print(f"  • Missing Data: Natural handling without interpolation")
    print(f"  • Forecasting: Theoretically grounded prediction intervals")
    print(f"  • Model Selection: Likelihood-based comparison")
    print(f"  • Computational Efficiency: O(n) recursive algorithms")
    
    print(f"\nPractical Applications:")
    print(f"  • Economic Indicators: GDP, inflation trend extraction")
    print(f"  • Financial Markets: Volatility modeling, risk factors")
    print(f"  • Engineering Systems: Signal processing, control theory")
    print(f"  • Epidemiology: Disease spread modeling")
    print(f"  • Climate Science: Temperature trend analysis")
    
    return framework, data_complete, {
        'true_trend': true_trend,
        'true_seasonal': seasonal_pattern,
        'missing_indices': missing_indices
    }

# Execute demonstration
if __name__ == "__main__":
    framework, data, components = demonstrate_state_space_advantages()
```

**Key Advantages of State-Space Models and Kalman Filter:**

1. **Unobserved Components**: Systematically handle latent variables like trends, cycles, and structural factors

2. **Statistical Optimality**: Provide minimum mean squared error estimates under normality assumptions

3. **Missing Data**: Naturally accommodate irregular observations without requiring interpolation

4. **Recursive Computation**: Efficient O(n) algorithms suitable for real-time applications

5. **Uncertainty Quantification**: Exact confidence intervals for all estimates and forecasts

6. **Model Flexibility**: Easily accommodate time-varying parameters and complex dynamics

7. **Forecasting Framework**: Theoretically grounded prediction intervals with proper uncertainty propagation

8. **Likelihood-Based Inference**: Enable rigorous model comparison and parameter testing

---

## Question 4

**How would you approach building atime series modelto forecaststock prices?**

**Answer:**

**Theoretical Framework:**

Stock price forecasting represents one of the most challenging applications in time series analysis due to **market efficiency**, **non-linear dynamics**, and **regime changes**. The theoretical approach must account for **volatility clustering**, **fat-tailed distributions**, and **structural breaks**.

**Mathematical Foundation:**

**Efficient Market Hypothesis:**
```
P_t = E[P_t | I_{t-1}] + ε_t

Where:
- P_t: Stock price at time t
- I_{t-1}: Information set at time t-1
- ε_t: Unpredictable innovation
```

**Log-Return Modeling:**
```
r_t = log(P_t/P_{t-1}) = μ + σ_t ε_t

Where:
- r_t: Log return
- μ: Expected return
- σ_t: Time-varying volatility
- ε_t ~ iid(0,1)
```

**Comprehensive Implementation:**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

class StockPriceForecastingFramework:
    """
    Comprehensive stock price forecasting using multiple approaches
    """
    
    def __init__(self):
        self.models = {}
        self.forecasts = {}
        self.performance = {}
        
    def fetch_stock_data(self, symbol, start_date='2020-01-01', end_date='2024-01-01'):
        """Fetch stock data"""
        try:
            stock = yf.download(symbol, start=start_date, end=end_date)
            return stock['Adj Close']
        except:
            # Generate synthetic data if yfinance fails
            np.random.seed(42)
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            returns = np.random.normal(0.001, 0.02, len(dates))
            prices = 100 * np.exp(np.cumsum(returns))
            return pd.Series(prices, index=dates)
    
    def prepare_data(self, prices):
        """Prepare data for modeling"""
        
        # Calculate returns
        returns = np.log(prices / prices.shift(1)).dropna()
        
        # Basic statistics
        stats = {
            'mean_return': returns.mean(),
            'volatility': returns.std(),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'jarque_bera': self._jarque_bera_test(returns)
        }
        
        print(f"Data Statistics:")
        print(f"  • Mean return: {stats['mean_return']:.6f}")
        print(f"  • Volatility: {stats['volatility']:.4f}")
        print(f"  • Skewness: {stats['skewness']:.3f}")
        print(f"  • Kurtosis: {stats['kurtosis']:.3f}")
        
        return returns, stats
    
    def _jarque_bera_test(self, data):
        """Jarque-Bera normality test"""
        from scipy.stats import jarque_bera
        statistic, pvalue = jarque_bera(data)
        return {'statistic': statistic, 'pvalue': pvalue}
    
    def arima_modeling(self, returns, order=(1,0,1)):
        """ARIMA model for returns"""
        
        print(f"\n=== ARIMA Modeling ===")
        
        # Split data
        train_size = int(0.8 * len(returns))
        train, test = returns[:train_size], returns[train_size:]
        
        # Fit ARIMA model
        model = ARIMA(train, order=order)
        fitted_model = model.fit()
        
        # Forecasts
        forecast = fitted_model.forecast(steps=len(test))
        
        # Performance
        mse = mean_squared_error(test, forecast)
        
        self.models['ARIMA'] = fitted_model
        self.forecasts['ARIMA'] = forecast
        self.performance['ARIMA'] = {'MSE': mse}
        
        print(f"ARIMA{order} Results:")
        print(f"  • MSE: {mse:.6f}")
        print(f"  • AIC: {fitted_model.aic:.2f}")
        
        return fitted_model, forecast
    
    def garch_modeling(self, returns):
        """GARCH model for volatility"""
        
        print(f"\n=== GARCH Modeling ===")
        
        # Split data
        train_size = int(0.8 * len(returns))
        train, test = returns[:train_size], returns[train_size:]
        
        # Fit GARCH(1,1) model
        garch_model = arch_model(train * 100, vol='Garch', p=1, q=1)
        fitted_garch = garch_model.fit(disp='off')
        
        # Volatility forecasts
        vol_forecast = fitted_garch.forecast(horizon=len(test))
        predicted_vol = np.sqrt(vol_forecast.variance.iloc[-1].values) / 100
        
        self.models['GARCH'] = fitted_garch
        self.forecasts['GARCH'] = predicted_vol
        
        print(f"GARCH(1,1) Results:")
        print(f"  • AIC: {fitted_garch.aic:.2f}")
        print(f"  • Mean predicted volatility: {predicted_vol.mean():.4f}")
        
        return fitted_garch, predicted_vol
    
    def ensemble_modeling(self, returns):
        """Ensemble approach using multiple models"""
        
        print(f"\n=== Ensemble Modeling ===")
        
        # Prepare features
        features = self._create_features(returns)
        
        # Split data
        train_size = int(0.8 * len(features))
        X_train, X_test = features[:train_size], features[train_size:]
        y_train, y_test = returns[1:train_size+1], returns[train_size+1:]
        
        # Random Forest model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Predictions
        rf_forecast = rf_model.predict(X_test)
        
        # Performance
        mse = mean_squared_error(y_test, rf_forecast)
        
        self.models['Ensemble'] = rf_model
        self.forecasts['Ensemble'] = rf_forecast
        self.performance['Ensemble'] = {'MSE': mse}
        
        print(f"Ensemble Results:")
        print(f"  • MSE: {mse:.6f}")
        print(f"  • Feature importance: {rf_model.feature_importances_[:5]}")
        
        return rf_model, rf_forecast
    
    def _create_features(self, returns):
        """Create technical features"""
        
        features = pd.DataFrame(index=returns.index[:-1])
        
        # Lagged returns
        for lag in range(1, 6):
            features[f'return_lag_{lag}'] = returns.shift(lag)[:-1]
        
        # Moving averages
        features['ma_5'] = returns.rolling(5).mean()[:-1]
        features['ma_20'] = returns.rolling(20).mean()[:-1]
        
        # Volatility measures
        features['vol_5'] = returns.rolling(5).std()[:-1]
        features['vol_20'] = returns.rolling(20).std()[:-1]
        
        return features.dropna()
    
    def model_diagnostics(self, returns):
        """Comprehensive model diagnostics"""
        
        print(f"\n=== Model Diagnostics ===")
        
        for model_name, model in self.models.items():
            if model_name == 'ARIMA':
                residuals = model.resid
                
                # Ljung-Box test
                from statsmodels.stats.diagnostic import acorr_ljungbox
                lb_result = acorr_ljungbox(residuals, lags=10, return_df=True)
                lb_pvalue = lb_result['lb_pvalue'].iloc[-1]
                
                print(f"{model_name} Diagnostics:")
                print(f"  • Ljung-Box p-value: {lb_pvalue:.4f}")
                print(f"  • Residual normality: {'Pass' if self._jarque_bera_test(residuals)['pvalue'] > 0.05 else 'Fail'}")
    
    def plot_results(self, prices, returns):
        """Visualization of results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Stock Price Forecasting Results', fontsize=16, fontweight='bold')
        
        # Price series
        axes[0, 0].plot(prices.index, prices.values)
        axes[0, 0].set_title('Stock Price History')
        axes[0, 0].set_ylabel('Price')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Returns
        axes[0, 1].plot(returns.index, returns.values, alpha=0.7)
        axes[0, 1].set_title('Log Returns')
        axes[0, 1].set_ylabel('Returns')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Forecast comparison
        if 'ARIMA' in self.forecasts and 'Ensemble' in self.forecasts:
            test_size = len(self.forecasts['ARIMA'])
            test_returns = returns[-test_size:]
            
            axes[1, 0].plot(test_returns.values, label='Actual', alpha=0.7)
            axes[1, 0].plot(self.forecasts['ARIMA'], label='ARIMA', alpha=0.7)
            axes[1, 0].plot(self.forecasts['Ensemble'], label='Ensemble', alpha=0.7)
            axes[1, 0].set_title('Forecast Comparison')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Performance comparison
        if self.performance:
            models = list(self.performance.keys())
            mse_values = [self.performance[model]['MSE'] for model in models]
            
            axes[1, 1].bar(models, mse_values)
            axes[1, 1].set_title('Model Performance (MSE)')
            axes[1, 1].set_ylabel('MSE')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig

def demonstrate_stock_forecasting():
    """Demonstrate stock price forecasting framework"""
    
    print("=== Stock Price Forecasting Demonstration ===")
    
    # Initialize framework
    forecaster = StockPriceForecastingFramework()
    
    # Fetch data
    print("\nFetching stock data...")
    prices = forecaster.fetch_stock_data('AAPL', '2022-01-01', '2024-01-01')
    
    # Prepare data
    returns, stats = forecaster.prepare_data(prices)
    
    # Apply different models
    arima_model, arima_forecast = forecaster.arima_modeling(returns)
    garch_model, vol_forecast = forecaster.garch_modeling(returns)
    ensemble_model, ensemble_forecast = forecaster.ensemble_modeling(returns)
    
    # Diagnostics
    forecaster.model_diagnostics(returns)
    
    # Visualization
    forecaster.plot_results(prices, returns)
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"Best performing model: {min(forecaster.performance.keys(), key=lambda x: forecaster.performance[x]['MSE'])}")
    
    print(f"\nKey Insights:")
    print(f"  • Stock returns show volatility clustering")
    print(f"  • GARCH models capture time-varying volatility")
    print(f"  • Ensemble methods often outperform single models")
    print(f"  • Out-of-sample validation is crucial")
    
    return forecaster

# Execute demonstration
if __name__ == "__main__":
    forecaster = demonstrate_stock_forecasting()
```

**Strategic Approach:**

1. **Data Preparation**: Log returns, outlier detection, statistical testing
2. **Model Selection**: ARIMA for returns, GARCH for volatility, ML for non-linearity
3. **Validation**: Out-of-sample testing, walk-forward analysis
4. **Risk Management**: Volatility forecasting, VaR estimation
5. **Ensemble Methods**: Combine multiple models for robustness

**Key Challenges:**
- Market efficiency limits predictability
- Structural breaks require adaptive models
- High-frequency noise vs. signal
- Model overfitting risks

---

## Question 5

**Discuss the challenges and strategies of usingtime series analysisinanomaly detectionforsystem monitoring.**

**Answer:**

**Theoretical Foundation:**

Anomaly detection in time series for system monitoring involves identifying **unusual patterns** that deviate from **normal system behavior**. This requires understanding **baseline establishment**, **threshold setting**, and **real-time detection** under dynamic conditions.

**Mathematical Framework:**

**Anomaly Definition:**
```
Anomaly_t = |X_t - E[X_t|I_{t-1}]| > k × σ_t

Where:
- X_t: Observed value
- E[X_t|I_{t-1}]: Expected value given history
- σ_t: Adaptive standard deviation
- k: Threshold multiplier
```

**Implementation:**

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import IsolationForest
from scipy import stats

class TimeSeriesAnomalyDetector:
    """Real-time anomaly detection for system monitoring"""
    
    def __init__(self, window_size=100, sensitivity=3.0):
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.baseline_model = None
        
    def detect_anomalies(self, data, method='statistical'):
        """Detect anomalies using multiple approaches"""
        
        if method == 'statistical':
            return self._statistical_detection(data)
        elif method == 'ml':
            return self._isolation_forest_detection(data)
        elif method == 'decomposition':
            return self._decomposition_based_detection(data)
    
    def _statistical_detection(self, data):
        """Statistical anomaly detection"""
        
        # Rolling statistics
        rolling_mean = data.rolling(self.window_size).mean()
        rolling_std = data.rolling(self.window_size).std()
        
        # Z-score based detection
        z_scores = np.abs((data - rolling_mean) / rolling_std)
        anomalies = z_scores > self.sensitivity
        
        return anomalies, z_scores
    
    def _isolation_forest_detection(self, data):
        """Machine learning based detection"""
        
        # Feature engineering
        features = self._create_features(data)
        
        # Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomalies = iso_forest.fit_predict(features) == -1
        
        return pd.Series(anomalies, index=data.index), None
    
    def _decomposition_based_detection(self, data):
        """Decomposition-based anomaly detection"""
        
        # Seasonal decomposition
        decomposition = seasonal_decompose(data, model='additive', period=24)
        
        # Anomalies in residuals
        residuals = decomposition.resid.dropna()
        threshold = self.sensitivity * residuals.std()
        anomalies = np.abs(residuals) > threshold
        
        return anomalies, residuals
    
    def _create_features(self, data):
        """Create features for ML detection"""
        
        features = pd.DataFrame()
        
        # Statistical features
        features['value'] = data
        features['rolling_mean'] = data.rolling(10).mean()
        features['rolling_std'] = data.rolling(10).std()
        
        # Time-based features
        features['hour'] = data.index.hour
        features['day_of_week'] = data.index.dayofweek
        
        return features.dropna()

def demonstrate_anomaly_detection():
    """Demonstrate anomaly detection for system monitoring"""
    
    # Generate synthetic system data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='H')
    
    # Normal pattern with daily seasonality
    normal_pattern = 50 + 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 24)
    noise = np.random.normal(0, 2, len(dates))
    
    # Inject anomalies
    data = normal_pattern + noise
    anomaly_indices = np.random.choice(len(dates), 50, replace=False)
    data[anomaly_indices] += np.random.normal(0, 20, 50)  # Spike anomalies
    
    ts_data = pd.Series(data, index=dates)
    
    # Initialize detector
    detector = TimeSeriesAnomalyDetector(sensitivity=2.5)
    
    # Detect anomalies
    stat_anomalies, z_scores = detector.detect_anomalies(ts_data, 'statistical')
    ml_anomalies, _ = detector.detect_anomalies(ts_data, 'ml')
    decomp_anomalies, residuals = detector.detect_anomalies(ts_data, 'decomposition')
    
    print(f"Anomaly Detection Results:")
    print(f"  • Statistical method: {stat_anomalies.sum()} anomalies")
    print(f"  • ML method: {ml_anomalies.sum()} anomalies")
    print(f"  • Decomposition method: {decomp_anomalies.sum()} anomalies")
    
    return detector, ts_data, stat_anomalies

# Execute demonstration
if __name__ == "__main__":
    detector, data, anomalies = demonstrate_anomaly_detection()
```

**Key Challenges:**

1. **Baseline Establishment**: Defining normal behavior in dynamic systems
2. **Concept Drift**: System behavior changes over time
3. **False Positives**: Balancing sensitivity vs. specificity
4. **Scalability**: Real-time processing of multiple metrics
5. **Interpretability**: Understanding why anomalies occurred

**Strategic Solutions:**

1. **Adaptive Baselines**: Use rolling windows and online learning
2. **Multi-Method Ensemble**: Combine statistical and ML approaches
3. **Context Awareness**: Consider operational context and metadata
4. **Hierarchical Detection**: Monitor at multiple system levels
5. **Feedback Loops**: Incorporate domain expert validation

---

## Question 6

**How would you usetime series analysisto predictelectricity consumption patterns?**

**Answer:**

**Theoretical Foundation:**

Electricity consumption forecasting requires understanding **multiple seasonality**, **weather dependencies**, and **demand response patterns**. The challenge involves **hierarchical forecasting** across different time horizons and **external factor integration**.

**Mathematical Framework:**

**Multi-Seasonal Model:**
```
E_t = T_t + S_daily(t) + S_weekly(t) + S_annual(t) + W_t + ε_t

Where:
- E_t: Electricity consumption at time t
- T_t: Trend component
- S_*: Seasonal components at different frequencies
- W_t: Weather effect
- ε_t: Random component
```

**Implementation Strategy:**

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor

class ElectricityDemandForecaster:
    """Electricity consumption forecasting framework"""
    
    def __init__(self):
        self.models = {}
        self.forecasts = {}
        
    def prepare_features(self, consumption, weather_data=None):
        """Create comprehensive feature set"""
        
        features = pd.DataFrame(index=consumption.index)
        
        # Time-based features
        features['hour'] = consumption.index.hour
        features['day_of_week'] = consumption.index.dayofweek
        features['month'] = consumption.index.month
        features['is_weekend'] = (consumption.index.dayofweek >= 5).astype(int)
        
        # Cyclical encoding
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        
        # Lagged consumption
        for lag in [1, 24, 168]:  # 1h, 1day, 1week
            features[f'consumption_lag_{lag}'] = consumption.shift(lag)
        
        # Weather features (if available)
        if weather_data is not None:
            features['temperature'] = weather_data['temp']
            features['cooling_degree_days'] = np.maximum(weather_data['temp'] - 18, 0)
            features['heating_degree_days'] = np.maximum(18 - weather_data['temp'], 0)
        
        return features.dropna()
    
    def forecast_consumption(self, consumption, horizon=24):
        """Generate consumption forecasts"""
        
        # Method 1: Triple Exponential Smoothing
        model_ets = ExponentialSmoothing(
            consumption, 
            trend='add', 
            seasonal='add', 
            seasonal_periods=24
        ).fit()
        
        forecast_ets = model_ets.forecast(horizon)
        
        # Method 2: Machine Learning with features
        features = self.prepare_features(consumption)
        train_size = len(features) - horizon
        
        X_train = features[:train_size]
        y_train = consumption[features.index[:train_size]]
        
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Generate future features (simplified)
        future_features = self._generate_future_features(features, horizon)
        forecast_ml = rf_model.predict(future_features)
        
        self.models = {'ETS': model_ets, 'ML': rf_model}
        self.forecasts = {'ETS': forecast_ets, 'ML': forecast_ml}
        
        return forecast_ets, forecast_ml
    
    def _generate_future_features(self, features, horizon):
        """Generate features for future time periods"""
        
        last_time = features.index[-1]
        future_index = pd.date_range(
            start=last_time + pd.Timedelta(hours=1), 
            periods=horizon, 
            freq='H'
        )
        
        future_features = pd.DataFrame(index=future_index)
        future_features['hour'] = future_index.hour
        future_features['day_of_week'] = future_index.dayofweek
        future_features['month'] = future_index.month
        future_features['is_weekend'] = (future_index.dayofweek >= 5).astype(int)
        
        # Cyclical encoding
        future_features['hour_sin'] = np.sin(2 * np.pi * future_features['hour'] / 24)
        future_features['hour_cos'] = np.cos(2 * np.pi * future_features['hour'] / 24)
        
        # Fill lagged features with last known values (simplified)
        for col in ['consumption_lag_1', 'consumption_lag_24', 'consumption_lag_168']:
            if col in features.columns:
                future_features[col] = features[col].iloc[-1]
        
        return future_features

# Demo
def demonstrate_electricity_forecasting():
    """Demonstrate electricity consumption forecasting"""
    
    # Generate synthetic hourly electricity data
    np.random.seed(42)
    hours = pd.date_range('2023-01-01', '2023-12-31', freq='H')
    
    # Base consumption with multiple seasonalities
    base_consumption = 1000
    daily_pattern = 200 * np.sin(2 * np.pi * hours.hour / 24)
    weekly_pattern = 100 * np.sin(2 * np.pi * hours.dayofweek / 7)
    annual_pattern = 150 * np.sin(2 * np.pi * hours.dayofyear / 365)
    
    consumption = (base_consumption + daily_pattern + 
                  weekly_pattern + annual_pattern + 
                  np.random.normal(0, 50, len(hours)))
    
    consumption_ts = pd.Series(consumption, index=hours)
    
    # Initialize forecaster
    forecaster = ElectricityDemandForecaster()
    
    # Generate forecasts
    forecast_ets, forecast_ml = forecaster.forecast_consumption(consumption_ts[:-48], horizon=48)
    
    print("Electricity Demand Forecasting Results:")
    print(f"  • ETS forecast range: {forecast_ets.min():.0f} - {forecast_ets.max():.0f} kWh")
    print(f"  • ML forecast range: {forecast_ml.min():.0f} - {forecast_ml.max():.0f} kWh")
    
    return forecaster, consumption_ts

if __name__ == "__main__":
    forecaster, data = demonstrate_electricity_forecasting()
```

**Key Considerations:**
- **Multiple Seasonality**: Daily, weekly, seasonal patterns
- **Weather Integration**: Temperature, humidity effects
- **Load Profiling**: Different customer segments
- **Real-time Updates**: Continuous model refinement

---

## Question 7

**Propose a strategy for forecastingtourist arrivalsusingtime series data.**

**Answer:**

**Theoretical Foundation:**

Tourist arrival forecasting involves **seasonal tourism patterns**, **economic indicators**, **external events**, and **marketing effects**. The approach requires **regime-aware modeling** and **external factor integration**.

**Strategic Framework:**

```python
class TourismForecastingStrategy:
    """Comprehensive tourism forecasting framework"""
    
    def __init__(self):
        self.base_models = {}
        self.external_factors = {}
        
    def tourism_specific_features(self, arrivals, events=None, economic_data=None):
        """Create tourism-specific features"""
        
        features = pd.DataFrame(index=arrivals.index)
        
        # Seasonal indicators
        features['peak_season'] = arrivals.index.month.isin([6, 7, 8]).astype(int)
        features['shoulder_season'] = arrivals.index.month.isin([4, 5, 9, 10]).astype(int)
        
        # Holiday effects
        features['school_holidays'] = self._identify_school_holidays(arrivals.index)
        features['public_holidays'] = self._identify_public_holidays(arrivals.index)
        
        # Economic indicators
        if economic_data is not None:
            features['gdp_growth'] = economic_data['gdp_growth']
            features['exchange_rate'] = economic_data['exchange_rate']
            features['fuel_prices'] = economic_data['fuel_prices']
        
        # Event effects
        if events is not None:
            features['major_events'] = self._encode_events(arrivals.index, events)
        
        return features
    
    def forecast_tourism(self, arrivals, horizon=12):
        """Generate tourism forecasts with uncertainty quantification"""
        
        # Seasonal decomposition
        from statsmodels.tsa.seasonal import seasonal_decompose
        decomposition = seasonal_decompose(arrivals, model='multiplicative', period=12)
        
        # Trend forecasting
        trend = decomposition.trend.dropna()
        trend_forecast = self._forecast_trend(trend, horizon)
        
        # Seasonal component
        seasonal = decomposition.seasonal[:12]  # One year of seasonal pattern
        seasonal_forecast = np.tile(seasonal.values, horizon // 12 + 1)[:horizon]
        
        # Combine forecasts
        base_forecast = trend_forecast * seasonal_forecast
        
        # Confidence intervals
        residuals = decomposition.resid.dropna()
        forecast_std = residuals.std()
        
        lower_ci = base_forecast * (1 - 1.96 * forecast_std / arrivals.mean())
        upper_ci = base_forecast * (1 + 1.96 * forecast_std / arrivals.mean())
        
        return {
            'forecast': base_forecast,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci,
            'components': {
                'trend': trend_forecast,
                'seasonal': seasonal_forecast
            }
        }
    
    def _forecast_trend(self, trend, horizon):
        """Forecast trend component"""
        
        # Simple linear extrapolation
        x = np.arange(len(trend))
        coeffs = np.polyfit(x, trend, 1)
        
        future_x = np.arange(len(trend), len(trend) + horizon)
        trend_forecast = np.polyval(coeffs, future_x)
        
        return trend_forecast
    
    def _identify_school_holidays(self, dates):
        """Identify school holiday periods"""
        
        # Simplified school holiday identification
        school_holidays = np.zeros(len(dates))
        
        for i, date in enumerate(dates):
            if date.month in [7, 8, 12]:  # Summer and winter holidays
                school_holidays[i] = 1
        
        return school_holidays
    
    def _identify_public_holidays(self, dates):
        """Identify public holidays"""
        
        # Simplified public holiday identification
        public_holidays = np.zeros(len(dates))
        
        # Add major holidays (simplified)
        for i, date in enumerate(dates):
            if (date.month == 12 and date.day == 25) or \
               (date.month == 1 and date.day == 1):
                public_holidays[i] = 1
        
        return public_holidays
    
    def _encode_events(self, dates, events):
        """Encode special events"""
        
        event_effects = np.zeros(len(dates))
        
        for event_date, effect_size in events.items():
            for i, date in enumerate(dates):
                if abs((date - event_date).days) <= 30:  # Event effect window
                    event_effects[i] = effect_size
        
        return event_effects

# Demo
def demonstrate_tourism_forecasting():
    """Demonstrate tourism forecasting strategy"""
    
    # Generate synthetic monthly tourism data
    np.random.seed(42)
    months = pd.date_range('2020-01-01', '2023-12-01', freq='MS')
    
    # Tourism pattern with seasonality
    base_arrivals = 10000
    seasonal_effect = 3000 * np.sin(2 * np.pi * months.month / 12)
    trend = 50 * np.arange(len(months))
    noise = np.random.normal(0, 500, len(months))
    
    arrivals = base_arrivals + seasonal_effect + trend + noise
    arrivals_ts = pd.Series(arrivals, index=months)
    
    # Initialize strategy
    strategy = TourismForecastingStrategy()
    
    # Generate forecasts
    forecast_result = strategy.forecast_tourism(arrivals_ts, horizon=12)
    
    print("Tourism Forecasting Results:")
    print(f"  • Forecast range: {forecast_result['forecast'].min():.0f} - {forecast_result['forecast'].max():.0f}")
    print(f"  • Peak season forecast: {forecast_result['forecast'][5:8].mean():.0f}")
    print(f"  • Off-season forecast: {forecast_result['forecast'][0:3].mean():.0f}")
    
    return strategy, arrivals_ts

if __name__ == "__main__":
    strategy, data = demonstrate_tourism_forecasting()
```

**Strategic Elements:**
- **Seasonal Modeling**: Peak/off-season patterns
- **External Integration**: Economic, weather, event data
- **Scenario Planning**: Multiple forecast scenarios
- **Stakeholder Communication**: Clear uncertainty bands

---

## Question 8

**How would you analyze and predict theload on a serverusingtime series?**

**Answer:**

**Theoretical Foundation:**

Server load prediction requires understanding **traffic patterns**, **capacity constraints**, **auto-scaling triggers**, and **performance degradation**. The approach involves **real-time monitoring** and **predictive scaling**.

**Implementation Framework:**

```python
class ServerLoadPredictor:
    """Server load prediction and capacity planning"""
    
    def __init__(self, capacity_threshold=80):
        self.capacity_threshold = capacity_threshold
        self.models = {}
        self.alerts = []
        
    def analyze_load_patterns(self, load_data):
        """Analyze server load patterns"""
        
        analysis = {
            'basic_stats': {
                'mean_load': load_data.mean(),
                'max_load': load_data.max(),
                'p95_load': load_data.quantile(0.95),
                'peak_hours': load_data.groupby(load_data.index.hour).mean().idxmax()
            },
            'seasonality': {
                'daily_pattern': load_data.groupby(load_data.index.hour).mean(),
                'weekly_pattern': load_data.groupby(load_data.index.dayofweek).mean()
            },
            'anomalies': self._detect_load_anomalies(load_data)
        }
        
        return analysis
    
    def predict_load(self, load_data, horizon_minutes=60):
        """Predict server load for capacity planning"""
        
        # Short-term prediction using ARIMA
        from statsmodels.tsa.arima.model import ARIMA
        
        model = ARIMA(load_data, order=(2, 1, 1))
        fitted_model = model.fit()
        
        # Generate forecasts
        forecast = fitted_model.forecast(steps=horizon_minutes)
        forecast_ci = fitted_model.get_forecast(steps=horizon_minutes).conf_int()
        
        # Capacity alerts
        capacity_alerts = self._generate_capacity_alerts(forecast, forecast_ci)
        
        return {
            'forecast': forecast,
            'confidence_interval': forecast_ci,
            'capacity_alerts': capacity_alerts,
            'scaling_recommendations': self._scaling_recommendations(forecast)
        }
    
    def _detect_load_anomalies(self, load_data):
        """Detect anomalous load patterns"""
        
        # Statistical anomaly detection
        rolling_mean = load_data.rolling(window=60).mean()
        rolling_std = load_data.rolling(window=60).std()
        
        z_scores = np.abs((load_data - rolling_mean) / rolling_std)
        anomalies = z_scores > 3
        
        return {
            'anomaly_count': anomalies.sum(),
            'anomaly_times': load_data[anomalies].index.tolist(),
            'max_z_score': z_scores.max()
        }
    
    def _generate_capacity_alerts(self, forecast, confidence_interval):
        """Generate capacity planning alerts"""
        
        alerts = []
        
        # Check if forecast exceeds capacity threshold
        high_load_periods = forecast > self.capacity_threshold
        
        if high_load_periods.any():
            alerts.append({
                'type': 'capacity_warning',
                'message': f'Load expected to exceed {self.capacity_threshold}% threshold',
                'time_to_threshold': high_load_periods.idxmax(),
                'peak_forecast': forecast.max()
            })
        
        # Check confidence interval upper bound
        upper_bound = confidence_interval.iloc[:, 1]
        if (upper_bound > self.capacity_threshold).any():
            alerts.append({
                'type': 'capacity_risk',
                'message': 'High probability of capacity breach',
                'risk_level': 'HIGH' if upper_bound.max() > 90 else 'MEDIUM'
            })
        
        return alerts
    
    def _scaling_recommendations(self, forecast):
        """Generate auto-scaling recommendations"""
        
        recommendations = []
        
        max_forecast = forecast.max()
        
        if max_forecast > 90:
            recommendations.append({
                'action': 'scale_up',
                'urgency': 'immediate',
                'suggested_capacity': int(max_forecast * 1.2)
            })
        elif max_forecast > 75:
            recommendations.append({
                'action': 'prepare_scale_up',
                'urgency': 'within_30min',
                'suggested_capacity': int(max_forecast * 1.1)
            })
        elif max_forecast < 30:
            recommendations.append({
                'action': 'consider_scale_down',
                'urgency': 'low',
                'suggested_capacity': int(max_forecast * 1.5)
            })
        
        return recommendations

# Demo
def demonstrate_server_load_prediction():
    """Demonstrate server load prediction"""
    
    # Generate synthetic server load data
    np.random.seed(42)
    minutes = pd.date_range('2024-01-01', '2024-01-02', freq='T')
    
    # Server load pattern
    base_load = 40
    hourly_pattern = 30 * np.sin(2 * np.pi * minutes.hour / 24)
    business_hours_boost = 20 * ((minutes.hour >= 9) & (minutes.hour <= 17)).astype(int)
    noise = np.random.normal(0, 5, len(minutes))
    
    # Simulate load spikes
    spike_indices = np.random.choice(len(minutes), 20, replace=False)
    spike_effect = np.zeros(len(minutes))
    spike_effect[spike_indices] = np.random.uniform(20, 40, 20)
    
    load = base_load + hourly_pattern + business_hours_boost + noise + spike_effect
    load = np.clip(load, 0, 100)  # Keep within 0-100% range
    
    load_ts = pd.Series(load, index=minutes)
    
    # Initialize predictor
    predictor = ServerLoadPredictor(capacity_threshold=80)
    
    # Analyze patterns
    analysis = predictor.analyze_load_patterns(load_ts)
    
    # Predict load
    prediction = predictor.predict_load(load_ts[:-60], horizon_minutes=60)
    
    print("Server Load Analysis:")
    print(f"  • Mean load: {analysis['basic_stats']['mean_load']:.1f}%")
    print(f"  • Peak load: {analysis['basic_stats']['max_load']:.1f}%")
    print(f"  • P95 load: {analysis['basic_stats']['p95_load']:.1f}%")
    print(f"  • Peak hour: {analysis['basic_stats']['peak_hours']}:00")
    print(f"  • Anomalies detected: {analysis['anomalies']['anomaly_count']}")
    
    print(f"\nLoad Prediction:")
    print(f"  • Next hour max forecast: {prediction['forecast'].max():.1f}%")
    print(f"  • Capacity alerts: {len(prediction['capacity_alerts'])}")
    print(f"  • Scaling recommendations: {len(prediction['scaling_recommendations'])}")
    
    return predictor, load_ts

if __name__ == "__main__":
    predictor, data = demonstrate_server_load_prediction()
```

**Key Components:**
- **Pattern Recognition**: Daily/weekly load cycles
- **Anomaly Detection**: Unusual traffic spikes
- **Capacity Planning**: Proactive scaling decisions
- **Alert System**: Real-time threshold monitoring

---

## Question 9

**Discuss your approach to evaluating the impact ofpromotional campaignsonsalesusingtime series analysis.**

**Answer:**

**Theoretical Foundation:**

Promotional campaign impact evaluation requires **causal inference** techniques to separate campaign effects from **baseline trends**, **seasonality**, and **external factors**. The approach involves **intervention analysis** and **counterfactual estimation**.

**Mathematical Framework:**

**Intervention Model:**
```
Y_t = μ_t + β × I_t + ε_t

Where:
- Y_t: Sales at time t
- μ_t: Baseline sales (without intervention)
- β: Campaign effect size
- I_t: Intervention indicator (0/1)
- ε_t: Random error
```

**Implementation:**

```python
class PromotionalCampaignAnalyzer:
    """Promotional campaign impact analysis framework"""
    
    def __init__(self):
        self.baseline_model = None
        self.intervention_results = {}
        
    def analyze_campaign_impact(self, sales_data, campaign_periods, 
                               control_variables=None):
        """Comprehensive campaign impact analysis"""
        
        # 1. Baseline modeling (pre-campaign)
        baseline_model = self._build_baseline_model(sales_data, campaign_periods)
        
        # 2. Intervention analysis
        intervention_results = self._intervention_analysis(
            sales_data, campaign_periods, baseline_model
        )
        
        # 3. Causal impact estimation
        causal_impact = self._estimate_causal_impact(
            sales_data, campaign_periods, baseline_model
        )
        
        return {
            'baseline_model': baseline_model,
            'intervention_results': intervention_results,
            'causal_impact': causal_impact,
            'statistical_significance': self._test_significance(intervention_results)
        }
    
    def _build_baseline_model(self, sales_data, campaign_periods):
        """Build baseline sales model"""
        
        # Exclude campaign periods for baseline
        baseline_mask = np.ones(len(sales_data), dtype=bool)
        for start, end in campaign_periods:
            baseline_mask[start:end+1] = False
        
        baseline_sales = sales_data[baseline_mask]
        
        # Fit baseline model (seasonal + trend)
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        baseline_model = ExponentialSmoothing(
            baseline_sales, 
            trend='add', 
            seasonal='add', 
            seasonal_periods=7
        ).fit()
        
        return baseline_model
    
    def _intervention_analysis(self, sales_data, campaign_periods, baseline_model):
        """Analyze intervention effects"""
        
        results = {}
        
        for i, (start, end) in enumerate(campaign_periods):
            campaign_name = f"Campaign_{i+1}"
            
            # Actual sales during campaign
            actual_sales = sales_data[start:end+1]
            
            # Predicted baseline (counterfactual)
            baseline_prediction = baseline_model.forecast(len(actual_sales))
            
            # Calculate impact
            absolute_impact = actual_sales.sum() - baseline_prediction.sum()
            relative_impact = (actual_sales.sum() / baseline_prediction.sum() - 1) * 100
            
            # Daily impact profile
            daily_impact = actual_sales.values - baseline_prediction
            
            results[campaign_name] = {
                'period': (start, end),
                'actual_sales': actual_sales.sum(),
                'predicted_baseline': baseline_prediction.sum(),
                'absolute_impact': absolute_impact,
                'relative_impact': relative_impact,
                'daily_impact': daily_impact,
                'peak_impact_day': np.argmax(daily_impact),
                'impact_consistency': np.std(daily_impact) / np.mean(daily_impact)
            }
        
        return results
    
    def _estimate_causal_impact(self, sales_data, campaign_periods, baseline_model):
        """Estimate causal impact using synthetic control approach"""
        
        causal_results = {}
        
        for i, (start, end) in enumerate(campaign_periods):
            # Create intervention indicator
            intervention = np.zeros(len(sales_data))
            intervention[start:end+1] = 1
            
            # Regression with intervention
            from sklearn.linear_model import LinearRegression
            
            # Features: time trend, seasonality, intervention
            X = np.column_stack([
                np.arange(len(sales_data)),  # Trend
                np.sin(2 * np.pi * np.arange(len(sales_data)) / 7),  # Weekly seasonality
                np.cos(2 * np.pi * np.arange(len(sales_data)) / 7),
                intervention  # Intervention effect
            ])
            
            reg_model = LinearRegression().fit(X, sales_data)
            intervention_effect = reg_model.coef_[-1]  # Last coefficient
            
            # Statistical significance
            predictions = reg_model.predict(X)
            residuals = sales_data - predictions
            se_intervention = np.sqrt(np.var(residuals) * 
                                    np.linalg.inv(X.T @ X)[-1, -1])
            t_stat = intervention_effect / se_intervention
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(sales_data) - X.shape[1]))
            
            causal_results[f"Campaign_{i+1}"] = {
                'causal_effect': intervention_effect,
                'standard_error': se_intervention,
                't_statistic': t_stat,
                'p_value': p_value,
                'is_significant': p_value < 0.05
            }
        
        return causal_results
    
    def _test_significance(self, intervention_results):
        """Test statistical significance of interventions"""
        
        significance_tests = {}
        
        for campaign, results in intervention_results.items():
            daily_impact = results['daily_impact']
            
            # T-test against zero (no effect)
            t_stat, p_value = stats.ttest_1samp(daily_impact, 0)
            
            significance_tests[campaign] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'is_significant': p_value < 0.05,
                'effect_size': np.mean(daily_impact) / np.std(daily_impact)  # Cohen's d
            }
        
        return significance_tests

# Demo
def demonstrate_campaign_analysis():
    """Demonstrate promotional campaign impact analysis"""
    
    # Generate synthetic sales data
    np.random.seed(42)
    days = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    
    # Baseline sales with trend and seasonality
    trend = 1000 + 2 * np.arange(len(days))
    weekly_seasonality = 200 * np.sin(2 * np.pi * np.arange(len(days)) / 7)
    noise = np.random.normal(0, 100, len(days))
    
    baseline_sales = trend + weekly_seasonality + noise
    
    # Add campaign effects
    campaign_periods = [(50, 56), (150, 159), (250, 256)]  # 3 campaigns
    campaign_effects = [300, 500, 400]  # Different effect sizes
    
    sales = baseline_sales.copy()
    for (start, end), effect in zip(campaign_periods, campaign_effects):
        sales[start:end+1] += effect + np.random.normal(0, 50, end-start+1)
    
    sales_ts = pd.Series(sales, index=days)
    
    # Initialize analyzer
    analyzer = PromotionalCampaignAnalyzer()
    
    # Analyze campaign impact
    analysis_results = analyzer.analyze_campaign_impact(sales_ts, campaign_periods)
    
    print("Promotional Campaign Impact Analysis:")
    print("\nCampaign Results:")
    for campaign, results in analysis_results['intervention_results'].items():
        print(f"\n{campaign}:")
        print(f"  • Absolute impact: {results['absolute_impact']:.0f} units")
        print(f"  • Relative impact: {results['relative_impact']:.1f}%")
        print(f"  • Peak impact day: Day {results['peak_impact_day'] + 1}")
    
    print(f"\nStatistical Significance:")
    for campaign, sig_test in analysis_results['statistical_significance'].items():
        print(f"  • {campaign}: {'Significant' if sig_test['is_significant'] else 'Not significant'} (p={sig_test['p_value']:.4f})")
    
    return analyzer, sales_ts

if __name__ == "__main__":
    analyzer, data = demonstrate_campaign_analysis()
```

**Key Methodologies:**
- **Baseline Modeling**: Pre-campaign sales patterns
- **Intervention Analysis**: Before/after comparison with counterfactuals
- **Causal Inference**: Statistical significance testing
- **Attribution**: Isolating campaign effects from other factors

---

## Question 10

**Discuss the potential ofrecurrent neural networks (RNNs)intime series forecasting.**

**Answer:**

**Theoretical Foundation:**

RNNs represent a paradigm shift in time series forecasting by enabling **end-to-end learning** of **complex temporal dependencies** without explicit feature engineering. The theoretical advantage lies in **universal approximation** of non-linear temporal functions and **automatic pattern discovery**.

**Mathematical Framework:**

**RNN Architecture:**
```
h_t = f(W_hh × h_{t-1} + W_xh × x_t + b_h)
y_t = W_hy × h_t + b_y

Where:
- h_t: Hidden state at time t
- x_t: Input at time t
- W: Weight matrices
- f: Activation function (tanh, ReLU)
```

**LSTM Enhancement:**
```
f_t = σ(W_f × [h_{t-1}, x_t] + b_f)  # Forget gate
i_t = σ(W_i × [h_{t-1}, x_t] + b_i)  # Input gate
C̃_t = tanh(W_C × [h_{t-1}, x_t] + b_C)  # Candidate values
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t  # Cell state
o_t = σ(W_o × [h_{t-1}, x_t] + b_o)  # Output gate
h_t = o_t ⊙ tanh(C_t)  # Hidden state
```

**Implementation Framework:**

```python
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

class RNNTimeSeriesForecaster:
    """RNN-based time series forecasting framework"""
    
    def __init__(self, model_type='LSTM', hidden_size=50, num_layers=2):
        self.model_type = model_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model = None
        self.scaler = MinMaxScaler()
        
    def create_sequences(self, data, seq_length=10):
        """Create input-output sequences for RNN training"""
        
        sequences = []
        targets = []
        
        for i in range(len(data) - seq_length):
            seq = data[i:i+seq_length]
            target = data[i+seq_length]
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def build_model(self, input_size=1, output_size=1):
        """Build RNN model architecture"""
        
        class RNNModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, output_size, model_type):
                super(RNNModel, self).__init__()
                
                if model_type == 'LSTM':
                    self.rnn = nn.LSTM(input_size, hidden_size, num_layers, 
                                     batch_first=True, dropout=0.2)
                elif model_type == 'GRU':
                    self.rnn = nn.GRU(input_size, hidden_size, num_layers, 
                                    batch_first=True, dropout=0.2)
                else:  # Simple RNN
                    self.rnn = nn.RNN(input_size, hidden_size, num_layers, 
                                    batch_first=True, dropout=0.2)
                
                self.fc = nn.Linear(hidden_size, output_size)
                self.dropout = nn.Dropout(0.2)
                
            def forward(self, x):
                out, _ = self.rnn(x)
                out = self.dropout(out[:, -1, :])  # Take last output
                out = self.fc(out)
                return out
        
        self.model = RNNModel(input_size, self.hidden_size, self.num_layers, 
                             output_size, self.model_type)
        return self.model
    
    def train_model(self, data, seq_length=10, epochs=100, learning_rate=0.001):
        """Train RNN model"""
        
        # Prepare data
        scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1)).flatten()
        X, y = self.create_sequences(scaled_data, seq_length)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).unsqueeze(-1)
        y_tensor = torch.FloatTensor(y)
        
        # Split train/validation
        train_size = int(0.8 * len(X_tensor))
        X_train, X_val = X_tensor[:train_size], X_tensor[train_size:]
        y_train, y_val = y_tensor[:train_size], y_tensor[train_size:]
        
        # Build model
        if self.model is None:
            self.build_model()
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            optimizer.zero_grad()
            
            train_pred = self.model(X_train)
            train_loss = criterion(train_pred.squeeze(), y_train)
            train_loss.backward()
            optimizer.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val)
                val_loss = criterion(val_pred.squeeze(), y_val)
            
            train_losses.append(train_loss.item())
            val_losses.append(val_loss.item())
            
            if epoch % 20 == 0:
                print(f'Epoch {epoch}: Train Loss = {train_loss.item():.6f}, Val Loss = {val_loss.item():.6f}')
        
        return train_losses, val_losses
    
    def forecast(self, data, seq_length=10, horizon=10):
        """Generate forecasts using trained model"""
        
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        # Prepare input sequence
        scaled_data = self.scaler.transform(data.values.reshape(-1, 1)).flatten()
        input_seq = scaled_data[-seq_length:]
        
        forecasts = []
        self.model.eval()
        
        with torch.no_grad():
            current_seq = torch.FloatTensor(input_seq).unsqueeze(0).unsqueeze(-1)
            
            for _ in range(horizon):
                # Predict next value
                pred = self.model(current_seq)
                forecasts.append(pred.item())
                
                # Update sequence for next prediction
                new_seq = torch.cat([current_seq[:, 1:, :], 
                                   pred.unsqueeze(0).unsqueeze(-1)], dim=1)
                current_seq = new_seq
        
        # Inverse transform forecasts
        forecasts_scaled = np.array(forecasts).reshape(-1, 1)
        forecasts_original = self.scaler.inverse_transform(forecasts_scaled).flatten()
        
        return forecasts_original
    
    def evaluate_model(self, test_data, seq_length=10):
        """Evaluate model performance"""
        
        scaled_test = self.scaler.transform(test_data.values.reshape(-1, 1)).flatten()
        X_test, y_test = self.create_sequences(scaled_test, seq_length)
        
        self.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).unsqueeze(-1)
            predictions = self.model(X_test_tensor).squeeze().numpy()
        
        # Inverse transform
        y_test_original = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        pred_original = self.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        mse = mean_squared_error(y_test_original, pred_original)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test_original - pred_original))
        mape = np.mean(np.abs((y_test_original - pred_original) / y_test_original)) * 100
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }

# Demo
def demonstrate_rnn_forecasting():
    """Demonstrate RNN time series forecasting"""
    
    # Generate synthetic time series
    np.random.seed(42)
    t = np.arange(1000)
    trend = 0.02 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 50) + 5 * np.sin(2 * np.pi * t / 10)
    noise = np.random.normal(0, 2, len(t))
    data = 100 + trend + seasonal + noise
    
    ts_data = pd.Series(data)
    
    # Initialize RNN forecaster
    forecaster = RNNTimeSeriesForecaster(model_type='LSTM', hidden_size=64, num_layers=2)
    
    # Split data
    train_size = int(0.8 * len(ts_data))
    train_data = ts_data[:train_size]
    test_data = ts_data[train_size:]
    
    print("=== RNN Time Series Forecasting ===")
    print(f"Data size: {len(ts_data)}")
    print(f"Training size: {len(train_data)}")
    print(f"Test size: {len(test_data)}")
    
    # Train model
    print(f"\nTraining {forecaster.model_type} model...")
    train_losses, val_losses = forecaster.train_model(train_data, epochs=50)
    
    # Evaluate on test set
    test_metrics = forecaster.evaluate_model(test_data)
    print(f"\nTest Performance:")
    for metric, value in test_metrics.items():
        print(f"  • {metric}: {value:.4f}")
    
    # Generate forecasts
    forecasts = forecaster.forecast(train_data, horizon=20)
    print(f"\nForecasts generated: {len(forecasts)} points")
    print(f"Forecast range: {forecasts.min():.2f} - {forecasts.max():.2f}")
    
    return forecaster, ts_data, forecasts

# Execute demonstration
if __name__ == "__main__":
    forecaster, data, forecasts = demonstrate_rnn_forecasting()
```

**RNN Advantages:**

1. **Automatic Feature Learning**: No manual feature engineering required
2. **Non-linear Modeling**: Capture complex temporal patterns
3. **Long-term Dependencies**: LSTM/GRU handle vanishing gradients
4. **Multivariate Capability**: Handle multiple input variables naturally
5. **End-to-End Training**: Optimize entire pipeline jointly

**Challenges and Solutions:**

1. **Data Requirements**: Need large datasets; use transfer learning
2. **Overfitting**: Apply dropout, regularization, early stopping
3. **Interpretability**: Use attention mechanisms, feature importance
4. **Computational Cost**: Use GPU acceleration, model compression
5. **Hyperparameter Tuning**: Automated search, Bayesian optimization

**Best Practices:**
- **Architecture Selection**: LSTM for long sequences, GRU for efficiency
- **Sequence Length**: Balance memory vs. computational cost
- **Data Preprocessing**: Normalization, outlier handling critical
- **Validation Strategy**: Time-aware cross-validation
- **Ensemble Methods**: Combine multiple RNN models

---

