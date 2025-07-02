"""
Temporal Analysis Framework for Change Detection
===============================================
Advanced temporal analysis to distinguish anthropogenic changes from natural/seasonal variations
using multi-year time series data and statistical baselines.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union
import pickle
import os
import logging
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TimeSeriesPoint:
    """Single time series observation"""
    date: datetime
    values: Dict[str, float]  # Spectral indices and metrics
    cloud_cover: float
    satellite: str
    quality_flag: str = "good"


@dataclass
class SeasonalBaseline:
    """Seasonal baseline statistics"""
    month: int
    mean_values: Dict[str, float]
    std_values: Dict[str, float]
    percentiles: Dict[str, Dict[int, float]]  # 5th, 25th, 75th, 95th percentiles
    sample_count: int
    confidence_interval: Dict[str, Tuple[float, float]]


class TemporalAnalyzer:
    """
    Advanced temporal analysis for change detection
    
    This class implements state-of-the-art methods for:
    - Building robust seasonal baselines from multi-year data
    - Detecting anthropogenic anomalies vs natural variations
    - Statistical significance testing for changes
    - Trend analysis and forecasting
    """
    
    def __init__(self, 
                 baseline_years: int = 3,
                 min_observations_per_month: int = 5,
                 anomaly_threshold: float = 2.0,
                 trend_significance_level: float = 0.05):
        """
        Initialize temporal analyzer
        
        Args:
            baseline_years: Years of historical data for baseline
            min_observations_per_month: Minimum observations needed per month
            anomaly_threshold: Z-score threshold for anomaly detection
            trend_significance_level: P-value threshold for trend significance
        """
        self.baseline_years = baseline_years
        self.min_observations_per_month = min_observations_per_month
        self.anomaly_threshold = anomaly_threshold
        self.trend_significance_level = trend_significance_level
        
        self.time_series_data: List[TimeSeriesPoint] = []
        self.seasonal_baselines: Dict[int, SeasonalBaseline] = {}
        self.anomaly_detector = IsolationForest(
            contamination=0.1, 
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def add_observation(self, observation: TimeSeriesPoint):
        """Add new time series observation"""
        self.time_series_data.append(observation)
        logger.debug(f"Added observation for {observation.date}")
    
    def load_historical_data(self, data_path: str):
        """Load historical time series data from file"""
        try:
            if data_path.endswith('.pkl'):
                with open(data_path, 'rb') as f:
                    self.time_series_data = pickle.load(f)
            elif data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
                self.time_series_data = self._dataframe_to_timeseries(df)
            else:
                raise ValueError("Unsupported file format. Use .pkl or .csv")
            
            logger.info(f"Loaded {len(self.time_series_data)} observations from {data_path}")
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            raise
    
    def _dataframe_to_timeseries(self, df: pd.DataFrame) -> List[TimeSeriesPoint]:
        """Convert DataFrame to TimeSeriesPoint list"""
        time_series = []
        for _, row in df.iterrows():
            values = {col: row[col] for col in df.columns 
                     if col not in ['date', 'cloud_cover', 'satellite', 'quality_flag']}
            
            point = TimeSeriesPoint(
                date=pd.to_datetime(row['date']),
                values=values,
                cloud_cover=row.get('cloud_cover', 0.0),
                satellite=row.get('satellite', 'unknown'),
                quality_flag=row.get('quality_flag', 'good')
            )
            time_series.append(point)
        return time_series
    
    def build_seasonal_baselines(self):
        """
        Build robust seasonal baselines using multi-year historical data
        
        Uses statistical methods to establish normal ranges for each month,
        accounting for inter-annual variability and outliers.
        """
        if len(self.time_series_data) < 12:
            raise ValueError("Need at least 12 months of data to build baselines")
        
        # Group observations by month
        monthly_data = {month: [] for month in range(1, 13)}
        
        for obs in self.time_series_data:
            if obs.quality_flag == 'good':  # Only use high-quality observations
                monthly_data[obs.date.month].append(obs)
        
        # Build baseline for each month
        for month in range(1, 13):
            month_obs = monthly_data[month]
            
            if len(month_obs) < self.min_observations_per_month:
                logger.warning(f"Insufficient data for month {month}: {len(month_obs)} observations")
                continue
            
            # Extract values for all indices
            if not month_obs:
                continue
                
            indices = list(month_obs[0].values.keys())
            month_values = {idx: [] for idx in indices}
            
            for obs in month_obs:
                for idx in indices:
                    if idx in obs.values:
                        month_values[idx].append(obs.values[idx])
            
            # Calculate robust statistics for each index
            mean_values = {}
            std_values = {}
            percentiles = {}
            confidence_intervals = {}
            
            for idx in indices:
                if month_values[idx]:
                    values = np.array(month_values[idx])
                    
                    # Remove outliers using IQR method
                    Q1 = np.percentile(values, 25)
                    Q3 = np.percentile(values, 75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    clean_values = values[(values >= lower_bound) & (values <= upper_bound)]
                    
                    if len(clean_values) > 0:
                        mean_values[idx] = np.mean(clean_values)
                        std_values[idx] = np.std(clean_values)
                        
                        # Calculate percentiles
                        percentiles[idx] = {
                            5: np.percentile(clean_values, 5),
                            25: np.percentile(clean_values, 25),
                            50: np.percentile(clean_values, 50),
                            75: np.percentile(clean_values, 75),
                            95: np.percentile(clean_values, 95)
                        }
                        
                        # 95% confidence interval
                        sem = stats.sem(clean_values)
                        ci = stats.t.interval(0.95, len(clean_values)-1, 
                                            loc=mean_values[idx], scale=sem)
                        confidence_intervals[idx] = ci
            
            # Create seasonal baseline
            baseline = SeasonalBaseline(
                month=month,
                mean_values=mean_values,
                std_values=std_values,
                percentiles=percentiles,
                sample_count=len(month_obs),
                confidence_interval=confidence_intervals
            )
            
            self.seasonal_baselines[month] = baseline
            logger.info(f"Built baseline for month {month} with {len(month_obs)} observations")
        
        logger.info(f"Completed seasonal baseline construction for {len(self.seasonal_baselines)} months")
    
    def fit_anomaly_detector(self):
        """Fit anomaly detection model on historical data"""
        if not self.seasonal_baselines:
            self.build_seasonal_baselines()
        
        # Prepare feature matrix
        features = []
        for obs in self.time_series_data:
            if obs.quality_flag == 'good':
                # Get seasonal baseline for this month
                month = obs.date.month
                if month in self.seasonal_baselines:
                    baseline = self.seasonal_baselines[month]
                    
                    # Calculate normalized deviations from baseline
                    normalized_features = []
                    for idx, value in obs.values.items():
                        if idx in baseline.mean_values and baseline.std_values[idx] > 0:
                            z_score = (value - baseline.mean_values[idx]) / baseline.std_values[idx]
                            normalized_features.append(z_score)
                    
                    if normalized_features:
                        features.append(normalized_features)
        
        if features:
            features_array = np.array(features)
            self.scaler.fit(features_array)
            scaled_features = self.scaler.transform(features_array)
            self.anomaly_detector.fit(scaled_features)
            self.is_fitted = True
            logger.info(f"Fitted anomaly detector on {len(features)} observations")
        else:
            logger.warning("No valid features found for anomaly detection")
    
    def detect_anthropogenic_change(self, 
                                  new_observation: TimeSeriesPoint,
                                  return_detailed_analysis: bool = True) -> Dict:
        """
        Detect if a new observation represents anthropogenic change
        
        Args:
            new_observation: New observation to analyze
            return_detailed_analysis: Return detailed statistical analysis
            
        Returns:
            Dictionary with change detection results and analysis
        """
        if not self.is_fitted:
            raise ValueError("Analyzer not fitted. Call fit_anomaly_detector() first")
        
        month = new_observation.date.month
        if month not in self.seasonal_baselines:
            return {
                'is_anthropogenic': False,
                'confidence': 0.0,
                'reason': 'No baseline available for this month'
            }
        
        baseline = self.seasonal_baselines[month]
        result = {
            'is_anthropogenic': False,
            'confidence': 0.0,
            'anomaly_score': 0.0,
            'statistical_significance': {},
            'change_magnitude': {},
            'change_direction': {},
            'analysis_details': {}
        }
        
        # Calculate statistical deviations
        significant_changes = []
        z_scores = []
        
        for idx, value in new_observation.values.items():
            if idx in baseline.mean_values and baseline.std_values[idx] > 0:
                mean_val = baseline.mean_values[idx]
                std_val = baseline.std_values[idx]
                z_score = (value - mean_val) / std_val
                z_scores.append(z_score)
                
                # Statistical significance test
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                is_significant = p_value < self.trend_significance_level
                
                result['statistical_significance'][idx] = {
                    'z_score': z_score,
                    'p_value': p_value,
                    'is_significant': is_significant
                }
                
                result['change_magnitude'][idx] = abs(value - mean_val)
                result['change_direction'][idx] = 'increase' if value > mean_val else 'decrease'
                
                # Check if change exceeds threshold
                if abs(z_score) > self.anomaly_threshold:
                    significant_changes.append({
                        'index': idx,
                        'z_score': z_score,
                        'magnitude': abs(value - mean_val),
                        'direction': 'increase' if value > mean_val else 'decrease'
                    })
        
        # Use ML anomaly detector
        if z_scores:
            feature_vector = np.array(z_scores).reshape(1, -1)
            if feature_vector.shape[1] == self.scaler.n_features_in_:
                scaled_features = self.scaler.transform(feature_vector)
                anomaly_score = self.anomaly_detector.decision_function(scaled_features)[0]
                is_anomaly = self.anomaly_detector.predict(scaled_features)[0] == -1
                
                result['anomaly_score'] = anomaly_score
                
                # Combine statistical and ML approaches
                statistical_evidence = len(significant_changes) > 0
                ml_evidence = is_anomaly
                
                # Determine if change is anthropogenic
                if statistical_evidence and ml_evidence:
                    result['is_anthropogenic'] = True
                    result['confidence'] = min(0.95, 0.5 + abs(anomaly_score) * 0.45)
                elif statistical_evidence or ml_evidence:
                    result['is_anthropogenic'] = True
                    result['confidence'] = min(0.75, 0.3 + abs(anomaly_score) * 0.45)
                else:
                    result['confidence'] = max(0.05, 0.5 - abs(anomaly_score) * 0.45)
        
        # Add detailed analysis
        if return_detailed_analysis:
            result['analysis_details'] = {
                'baseline_month': month,
                'baseline_sample_count': baseline.sample_count,
                'significant_changes': significant_changes,
                'overall_deviation': np.mean([abs(z) for z in z_scores]) if z_scores else 0,
                'recommendation': self._generate_recommendation(result, significant_changes)
            }
        
        return result
    
    def _generate_recommendation(self, result: Dict, significant_changes: List) -> str:
        """Generate human-readable recommendation based on analysis"""
        if result['is_anthropogenic']:
            if result['confidence'] > 0.8:
                return "High confidence anthropogenic change detected. Recommend immediate investigation."
            elif result['confidence'] > 0.5:
                return "Moderate confidence anthropogenic change. Consider further monitoring."
            else:
                return "Possible anthropogenic change. Continue monitoring for confirmation."
        else:
            return "Change appears consistent with natural/seasonal variation."
    
    def analyze_trends(self, 
                      index_name: str, 
                      window_months: int = 12) -> Dict:
        """
        Analyze long-term trends in spectral indices
        
        Args:
            index_name: Name of spectral index to analyze
            window_months: Moving window size for trend analysis
            
        Returns:
            Trend analysis results
        """
        # Extract time series for specific index
        dates = []
        values = []
        
        for obs in self.time_series_data:
            if index_name in obs.values and obs.quality_flag == 'good':
                dates.append(obs.date)
                values.append(obs.values[index_name])
        
        if len(values) < window_months:
            return {'error': 'Insufficient data for trend analysis'}
        
        # Create DataFrame for analysis
        df = pd.DataFrame({'date': dates, 'value': values})
        df = df.sort_values('date')
        df['timestamp'] = df['date'].map(datetime.timestamp)
        
        # Linear trend analysis
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df['timestamp'], df['value']
        )
        
        # Seasonal decomposition
        df = df.set_index('date')
        df = df.resample('M').mean()  # Monthly aggregation
        
        # Mann-Kendall trend test (non-parametric)
        mk_trend = self._mann_kendall_test(df['value'].values)
        
        # Moving average trend
        df['moving_avg'] = df['value'].rolling(window=min(window_months, len(df))).mean()
        
        return {
            'linear_trend': {
                'slope': slope * 365.25 * 24 * 3600,  # Convert to per year
                'r_squared': r_value**2,
                'p_value': p_value,
                'is_significant': p_value < self.trend_significance_level
            },
            'mann_kendall': mk_trend,
            'data_points': len(values),
            'time_span_years': (max(dates) - min(dates)).days / 365.25,
            'recent_trend': self._calculate_recent_trend(df['value'].values, window_months//2)
        }
    
    def _mann_kendall_test(self, data: np.ndarray) -> Dict:
        """Mann-Kendall trend test implementation"""
        n = len(data)
        s = 0
        
        for i in range(n-1):
            for j in range(i+1, n):
                if data[j] > data[i]:
                    s += 1
                elif data[j] < data[i]:
                    s -= 1
        
        # Variance calculation
        var_s = n * (n - 1) * (2 * n + 5) / 18
        
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
        
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        return {
            'statistic': s,
            'z_score': z,
            'p_value': p_value,
            'trend': 'increasing' if s > 0 else 'decreasing' if s < 0 else 'no trend',
            'is_significant': p_value < self.trend_significance_level
        }
    
    def _calculate_recent_trend(self, data: np.ndarray, window: int) -> Dict:
        """Calculate trend for recent data window"""
        if len(data) < window:
            window = len(data)
        
        recent_data = data[-window:]
        x = np.arange(len(recent_data))
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent_data)
        
        return {
            'slope': slope,
            'r_squared': r_value**2,
            'p_value': p_value,
            'direction': 'increasing' if slope > 0 else 'decreasing',
            'window_size': window
        }
    
    def save_baselines(self, filepath: str):
        """Save seasonal baselines to file"""
        baseline_data = {
            'baselines': self.seasonal_baselines,
            'metadata': {
                'baseline_years': self.baseline_years,
                'min_observations_per_month': self.min_observations_per_month,
                'anomaly_threshold': self.anomaly_threshold,
                'created_date': datetime.now().isoformat(),
                'data_points': len(self.time_series_data)
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(baseline_data, f)
        
        logger.info(f"Saved baselines to {filepath}")
    
    def load_baselines(self, filepath: str):
        """Load seasonal baselines from file"""
        with open(filepath, 'rb') as f:
            baseline_data = pickle.load(f)
        
        self.seasonal_baselines = baseline_data['baselines']
        metadata = baseline_data.get('metadata', {})
        
        logger.info(f"Loaded baselines from {filepath}")
        logger.info(f"Baselines created: {metadata.get('created_date', 'Unknown')}")
        logger.info(f"Data points used: {metadata.get('data_points', 'Unknown')}")
    
    def generate_report(self, output_dir: str = "temporal_analysis_report"):
        """Generate comprehensive temporal analysis report"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create visualizations
        self._plot_seasonal_baselines(os.path.join(output_dir, "seasonal_baselines.png"))
        self._plot_time_series(os.path.join(output_dir, "time_series.png"))
        self._plot_anomaly_distribution(os.path.join(output_dir, "anomaly_distribution.png"))
        
        # Generate text report
        report_path = os.path.join(output_dir, "temporal_analysis_report.txt")
        with open(report_path, 'w') as f:
            f.write(self._generate_text_report())
        
        logger.info(f"Generated temporal analysis report in {output_dir}")
    
    def _plot_seasonal_baselines(self, filepath: str):
        """Plot seasonal baselines for all indices"""
        if not self.seasonal_baselines:
            return
        
        # Get all indices
        indices = set()
        for baseline in self.seasonal_baselines.values():
            indices.update(baseline.mean_values.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, idx in enumerate(list(indices)[:4]):  # Plot first 4 indices
            if i >= len(axes):
                break
                
            months = []
            means = []
            stds = []
            
            for month in range(1, 13):
                if month in self.seasonal_baselines and idx in self.seasonal_baselines[month].mean_values:
                    months.append(month)
                    means.append(self.seasonal_baselines[month].mean_values[idx])
                    stds.append(self.seasonal_baselines[month].std_values[idx])
            
            if means:
                axes[i].errorbar(months, means, yerr=stds, marker='o', capsize=5)
                axes[i].set_title(f'Seasonal Baseline: {idx}')
                axes[i].set_xlabel('Month')
                axes[i].set_ylabel(f'{idx} Value')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_time_series(self, filepath: str):
        """Plot time series data"""
        if not self.time_series_data:
            return
        
        # Extract NDVI time series (if available)
        dates = []
        ndvi_values = []
        
        for obs in self.time_series_data:
            if 'ndvi' in obs.values:
                dates.append(obs.date)
                ndvi_values.append(obs.values['ndvi'])
        
        if dates:
            plt.figure(figsize=(12, 6))
            plt.plot(dates, ndvi_values, 'b-', alpha=0.7, linewidth=1)
            plt.title('NDVI Time Series')
            plt.xlabel('Date')
            plt.ylabel('NDVI')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_anomaly_distribution(self, filepath: str):
        """Plot anomaly score distribution"""
        if not self.is_fitted:
            return
        
        # Calculate anomaly scores for all historical data
        scores = []
        for obs in self.time_series_data:
            if obs.quality_flag == 'good':
                month = obs.date.month
                if month in self.seasonal_baselines:
                    baseline = self.seasonal_baselines[month]
                    z_scores = []
                    
                    for idx, value in obs.values.items():
                        if idx in baseline.mean_values and baseline.std_values[idx] > 0:
                            z_score = (value - baseline.mean_values[idx]) / baseline.std_values[idx]
                            z_scores.append(z_score)
                    
                    if z_scores and len(z_scores) == self.scaler.n_features_in_:
                        feature_vector = np.array(z_scores).reshape(1, -1)
                        scaled_features = self.scaler.transform(feature_vector)
                        score = self.anomaly_detector.decision_function(scaled_features)[0]
                        scores.append(score)
        
        if scores:
            plt.figure(figsize=(10, 6))
            plt.hist(scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            plt.axvline(x=0, color='red', linestyle='--', label='Decision Boundary')
            plt.title('Anomaly Score Distribution')
            plt.xlabel('Anomaly Score')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
    
    def _generate_text_report(self) -> str:
        """Generate comprehensive text report"""
        report = []
        report.append("TEMPORAL ANALYSIS REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Data summary
        report.append("DATA SUMMARY")
        report.append("-" * 20)
        report.append(f"Total observations: {len(self.time_series_data)}")
        report.append(f"Date range: {min(obs.date for obs in self.time_series_data)} to {max(obs.date for obs in self.time_series_data)}")
        report.append(f"Baseline years: {self.baseline_years}")
        report.append("")
        
        # Baseline summary
        report.append("SEASONAL BASELINES")
        report.append("-" * 20)
        for month, baseline in self.seasonal_baselines.items():
            report.append(f"Month {month}: {baseline.sample_count} observations")
            for idx, mean_val in baseline.mean_values.items():
                std_val = baseline.std_values[idx]
                report.append(f"  {idx}: {mean_val:.4f} ± {std_val:.4f}")
        report.append("")
        
        # Model status
        report.append("MODEL STATUS")
        report.append("-" * 20)
        report.append(f"Anomaly detector fitted: {self.is_fitted}")
        report.append(f"Anomaly threshold: {self.anomaly_threshold}")
        report.append(f"Significance level: {self.trend_significance_level}")
        
        return "\n".join(report)


class ChangeTypeClassifier:
    """
    Advanced classifier for different types of anthropogenic changes
    
    Classifies detected changes into specific categories:
    - Urban development
    - Deforestation
    - Mining
    - Agriculture expansion
    - Infrastructure development
    """
    
    def __init__(self):
        self.change_signatures = {
            'urban_development': {
                'ndvi': ('decrease', 'high'),
                'ndbi': ('increase', 'high'),
                'brightness': ('increase', 'medium')
            },
            'deforestation': {
                'ndvi': ('decrease', 'very_high'),
                'brightness': ('increase', 'medium'),
                'texture': ('increase', 'high')
            },
            'mining': {
                'ndvi': ('decrease', 'very_high'),
                'brightness': ('increase', 'high'),
                'red_edge': ('decrease', 'high')
            },
            'agriculture_expansion': {
                'ndvi': ('variable', 'medium'),
                'ndwi': ('decrease', 'medium'),
                'texture': ('decrease', 'medium')
            }
        }
    
    def classify_change(self, 
                       before_indices: Dict[str, float],
                       after_indices: Dict[str, float],
                       temporal_context: Dict) -> Dict:
        """
        Classify type of anthropogenic change
        
        Args:
            before_indices: Spectral indices before change
            after_indices: Spectral indices after change  
            temporal_context: Temporal analysis context
            
        Returns:
            Classification results with confidence scores
        """
        # Calculate change vectors
        changes = {}
        for idx in before_indices:
            if idx in after_indices:
                changes[idx] = after_indices[idx] - before_indices[idx]
        
        # Score each change type
        type_scores = {}
        for change_type, signatures in self.change_signatures.items():
            score = self._calculate_type_score(changes, signatures)
            type_scores[change_type] = score
        
        # Find best match
        best_type = max(type_scores.keys(), key=lambda k: type_scores[k])
        best_score = type_scores[best_type]
        
        return {
            'predicted_type': best_type,
            'confidence': best_score,
            'all_scores': type_scores,
            'change_vector': changes
        }
    
    def _calculate_type_score(self, changes: Dict, signatures: Dict) -> float:
        """Calculate similarity score for a change type"""
        score = 0.0
        total_weight = 0.0
        
        weight_map = {'low': 1.0, 'medium': 2.0, 'high': 3.0, 'very_high': 4.0}
        
        for idx, (direction, importance) in signatures.items():
            if idx in changes:
                change_val = changes[idx]
                weight = weight_map[importance]
                
                if direction == 'increase' and change_val > 0:
                    score += weight * min(1.0, abs(change_val))
                elif direction == 'decrease' and change_val < 0:
                    score += weight * min(1.0, abs(change_val))
                elif direction == 'variable':
                    score += weight * 0.5  # Neutral for variable changes
                
                total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0.0 