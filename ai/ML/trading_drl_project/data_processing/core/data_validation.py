"""
Data Validation

Comprehensive data quality validation for trading data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
import warnings


@dataclass
class ValidationRule:
    """Single validation rule"""
    name: str
    description: str
    severity: str  # 'error', 'warning', 'info'
    threshold: Optional[float] = None
    
    
@dataclass
class ValidationResult:
    """Result of validation check"""
    rule_name: str
    passed: bool
    severity: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    
    
@dataclass
class DataQualityReport:
    """Comprehensive data quality report"""
    symbol: str
    timeframe: str
    total_records: int
    date_range: Tuple[datetime, datetime]
    validation_results: List[ValidationResult]
    quality_score: float
    recommendations: List[str]
    

class TradingDataValidator:
    """
    Comprehensive validator for trading data quality
    
    Validates OHLCV data, technical indicators, and derived features
    for machine learning and trading applications.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data validator
        
        Args:
            config: Validation configuration
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Default validation rules
        self.rules = self._initialize_default_rules()
        
        # Custom rules from config
        if 'custom_rules' in self.config:
            self.rules.extend(self.config['custom_rules'])
    
    def _initialize_default_rules(self) -> List[ValidationRule]:
        """Initialize default validation rules"""
        return [
            ValidationRule("missing_data", "Check for missing values", "error", 0.05),
            ValidationRule("price_consistency", "Validate OHLC price relationships", "error"),
            ValidationRule("volume_validity", "Check volume data validity", "warning"),
            ValidationRule("datetime_continuity", "Validate datetime index continuity", "error"),
            ValidationRule("outlier_detection", "Detect price outliers", "warning", 3.0),
            ValidationRule("duplicate_timestamps", "Check for duplicate timestamps", "error"),
            ValidationRule("future_dates", "Check for future dates", "error"),
            ValidationRule("zero_prices", "Check for zero or negative prices", "error"),
            ValidationRule("extreme_moves", "Detect extreme price movements", "warning", 0.5),
            ValidationRule("volume_spikes", "Detect volume anomalies", "info", 10.0),
            ValidationRule("data_completeness", "Check data completeness", "warning", 0.95),
            ValidationRule("timestamp_gaps", "Detect gaps in time series", "warning"),
        ]
    
    def validate_ohlcv_data(
        self,
        data: pd.DataFrame,
        symbol: str = "Unknown",
        timeframe: str = "Unknown"
    ) -> DataQualityReport:
        """
        Validate OHLCV data comprehensively
        
        Args:
            data: OHLCV DataFrame
            symbol: Trading symbol
            timeframe: Data timeframe
            
        Returns:
            Data quality report
        """
        self.logger.info(f"Validating OHLCV data for {symbol} {timeframe}")
        
        validation_results = []
        
        # Basic structure validation
        validation_results.extend(self._validate_structure(data))
        
        # Data quality checks
        validation_results.extend(self._validate_missing_data(data))
        validation_results.extend(self._validate_price_consistency(data))
        validation_results.extend(self._validate_volume_data(data))
        validation_results.extend(self._validate_datetime_index(data))
        validation_results.extend(self._validate_outliers(data))
        validation_results.extend(self._validate_duplicates(data))
        validation_results.extend(self._validate_extreme_movements(data))
        validation_results.extend(self._validate_data_completeness(data, timeframe))
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(validation_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(validation_results)
        
        # Create report
        date_range = (data.index.min(), data.index.max()) if len(data) > 0 else (None, None)
        
        report = DataQualityReport(
            symbol=symbol,
            timeframe=timeframe,
            total_records=len(data),
            date_range=date_range,
            validation_results=validation_results,
            quality_score=quality_score,
            recommendations=recommendations
        )
        
        self.logger.info(f"Validation completed. Quality score: {quality_score:.2f}")
        return report
    
    def _validate_structure(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Validate basic data structure"""
        results = []
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            results.append(ValidationResult(
                rule_name="structure",
                passed=False,
                severity="error",
                message=f"Missing required columns: {missing_cols}",
                details={'missing_columns': missing_cols}
            ))
        else:
            results.append(ValidationResult(
                rule_name="structure",
                passed=True,
                severity="info",
                message="All required columns present"
            ))
        
        # Check data types
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        non_numeric = []
        
        for col in numeric_cols:
            if col in data.columns and not pd.api.types.is_numeric_dtype(data[col]):
                non_numeric.append(col)
        
        if non_numeric:
            results.append(ValidationResult(
                rule_name="data_types",
                passed=False,
                severity="warning",
                message=f"Non-numeric columns detected: {non_numeric}",
                details={'non_numeric_columns': non_numeric}
            ))
        
        return results
    
    def _validate_missing_data(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Validate missing data"""
        results = []
        
        if data.empty:
            results.append(ValidationResult(
                rule_name="missing_data",
                passed=False,
                severity="error",
                message="Dataset is empty"
            ))
            return results
        
        # Check missing values per column
        missing_stats = {}
        total_rows = len(data)
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in data.columns:
                missing_count = data[col].isnull().sum()
                missing_pct = missing_count / total_rows
                missing_stats[col] = {
                    'count': missing_count,
                    'percentage': missing_pct
                }
        
        # Find columns with excessive missing data
        threshold = self._get_rule_threshold("missing_data", 0.05)
        problematic_cols = [
            col for col, stats in missing_stats.items() 
            if stats['percentage'] > threshold
        ]
        
        if problematic_cols:
            results.append(ValidationResult(
                rule_name="missing_data",
                passed=False,
                severity="error",
                message=f"Excessive missing data in columns: {problematic_cols}",
                details=missing_stats
            ))
        else:
            total_missing = sum(stats['count'] for stats in missing_stats.values())
            results.append(ValidationResult(
                rule_name="missing_data",
                passed=True,
                severity="info",
                message=f"Missing data within acceptable limits: {total_missing} values",
                details=missing_stats
            ))
        
        return results
    
    def _validate_price_consistency(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Validate OHLC price relationships"""
        results = []
        
        price_cols = ['open', 'high', 'low', 'close']
        if not all(col in data.columns for col in price_cols):
            return results
        
        # Check high >= max(open, close, low)
        high_valid = (
            (data['high'] >= data['open']) &
            (data['high'] >= data['close']) &
            (data['high'] >= data['low'])
        )
        
        # Check low <= min(open, close, high)
        low_valid = (
            (data['low'] <= data['open']) &
            (data['low'] <= data['close']) &
            (data['low'] <= data['high'])
        )
        
        # Check for zero or negative prices
        positive_prices = (data[price_cols] > 0).all(axis=1)
        
        # Combine all checks
        price_consistency = high_valid & low_valid & positive_prices
        
        invalid_count = (~price_consistency).sum()
        invalid_pct = invalid_count / len(data)
        
        if invalid_count > 0:
            results.append(ValidationResult(
                rule_name="price_consistency",
                passed=False,
                severity="error",
                message=f"Price consistency violations: {invalid_count} records ({invalid_pct:.2%})",
                details={
                    'invalid_count': invalid_count,
                    'invalid_percentage': invalid_pct,
                    'high_violations': (~high_valid).sum(),
                    'low_violations': (~low_valid).sum(),
                    'negative_prices': (~positive_prices).sum()
                }
            ))
        else:
            results.append(ValidationResult(
                rule_name="price_consistency",
                passed=True,
                severity="info",
                message="All price relationships are valid"
            ))
        
        return results
    
    def _validate_volume_data(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Validate volume data"""
        results = []
        
        if 'volume' not in data.columns:
            return results
        
        volume = data['volume']
        
        # Check for negative volumes
        negative_volume = (volume < 0).sum()
        
        # Check for zero volumes
        zero_volume = (volume == 0).sum()
        zero_pct = zero_volume / len(volume)
        
        # Check for extremely high volumes (outliers)
        if len(volume) > 10:
            q99 = volume.quantile(0.99)
            median = volume.median()
            extreme_threshold = median * 100  # 100x median
            extreme_volumes = (volume > extreme_threshold).sum()
        else:
            extreme_volumes = 0
        
        issues = []
        if negative_volume > 0:
            issues.append(f"{negative_volume} negative volumes")
        if zero_pct > 0.1:  # More than 10% zero volumes
            issues.append(f"{zero_pct:.1%} zero volumes")
        if extreme_volumes > 0:
            issues.append(f"{extreme_volumes} extreme volume spikes")
        
        if issues:
            results.append(ValidationResult(
                rule_name="volume_validity",
                passed=False,
                severity="warning",
                message=f"Volume issues detected: {', '.join(issues)}",
                details={
                    'negative_volumes': negative_volume,
                    'zero_volumes': zero_volume,
                    'zero_percentage': zero_pct,
                    'extreme_volumes': extreme_volumes
                }
            ))
        else:
            results.append(ValidationResult(
                rule_name="volume_validity",
                passed=True,
                severity="info",
                message="Volume data appears valid"
            ))
        
        return results
    
    def _validate_datetime_index(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Validate datetime index"""
        results = []
        
        if not isinstance(data.index, pd.DatetimeIndex):
            results.append(ValidationResult(
                rule_name="datetime_index",
                passed=False,
                severity="error",
                message="Index is not DatetimeIndex"
            ))
            return results
        
        # Check for future dates
        now = pd.Timestamp.now()
        future_dates = (data.index > now).sum()
        
        if future_dates > 0:
            results.append(ValidationResult(
                rule_name="future_dates",
                passed=False,
                severity="error",
                message=f"Found {future_dates} future dates",
                details={'future_dates_count': future_dates}
            ))
        
        # Check for duplicates
        duplicates = data.index.duplicated().sum()
        
        if duplicates > 0:
            results.append(ValidationResult(
                rule_name="duplicate_timestamps",
                passed=False,
                severity="error",
                message=f"Found {duplicates} duplicate timestamps",
                details={'duplicate_count': duplicates}
            ))
        
        # Check if index is sorted
        if not data.index.is_monotonic_increasing:
            results.append(ValidationResult(
                rule_name="datetime_continuity",
                passed=False,
                severity="warning",
                message="DateTime index is not sorted"
            ))
        
        if not any(result.rule_name.startswith('datetime') and not result.passed for result in results):
            results.append(ValidationResult(
                rule_name="datetime_index",
                passed=True,
                severity="info",
                message="DateTime index is valid"
            ))
        
        return results
    
    def _validate_outliers(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Detect outliers in price data"""
        results = []
        
        if 'close' not in data.columns or len(data) < 10:
            return results
        
        # Calculate returns
        returns = data['close'].pct_change().dropna()
        
        if len(returns) == 0:
            return results
        
        # Z-score based outlier detection
        threshold = self._get_rule_threshold("outlier_detection", 3.0)
        z_scores = np.abs((returns - returns.mean()) / returns.std())
        outliers = (z_scores > threshold).sum()
        
        # IQR based outlier detection
        q1 = returns.quantile(0.25)
        q3 = returns.quantile(0.75)
        iqr = q3 - q1
        iqr_outliers = ((returns < q1 - 1.5 * iqr) | (returns > q3 + 1.5 * iqr)).sum()
        
        if outliers > len(returns) * 0.05:  # More than 5% outliers
            results.append(ValidationResult(
                rule_name="outlier_detection",
                passed=False,
                severity="warning",
                message=f"High number of outliers detected: {outliers} ({outliers/len(returns):.1%})",
                details={
                    'z_score_outliers': outliers,
                    'iqr_outliers': iqr_outliers,
                    'outlier_percentage': outliers / len(returns)
                }
            ))
        else:
            results.append(ValidationResult(
                rule_name="outlier_detection",
                passed=True,
                severity="info",
                message=f"Outlier count within normal range: {outliers}"
            ))
        
        return results
    
    def _validate_duplicates(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Check for duplicate records"""
        results = []
        
        # Check for completely duplicate rows
        total_duplicates = data.duplicated().sum()
        
        if total_duplicates > 0:
            results.append(ValidationResult(
                rule_name="duplicate_records",
                passed=False,
                severity="warning",
                message=f"Found {total_duplicates} duplicate records",
                details={'duplicate_count': total_duplicates}
            ))
        else:
            results.append(ValidationResult(
                rule_name="duplicate_records",
                passed=True,
                severity="info",
                message="No duplicate records found"
            ))
        
        return results
    
    def _validate_extreme_movements(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Detect extreme price movements"""
        results = []
        
        if 'close' not in data.columns or len(data) < 2:
            return results
        
        # Calculate price changes
        price_changes = data['close'].pct_change().abs()
        
        # Define extreme movement threshold
        threshold = self._get_rule_threshold("extreme_moves", 0.5)  # 50% default
        extreme_moves = (price_changes > threshold).sum()
        
        if extreme_moves > 0:
            max_change = price_changes.max()
            results.append(ValidationResult(
                rule_name="extreme_moves",
                passed=False,
                severity="warning",
                message=f"Found {extreme_moves} extreme price movements (>{threshold:.0%})",
                details={
                    'extreme_moves_count': extreme_moves,
                    'max_change': max_change,
                    'threshold': threshold
                }
            ))
        else:
            results.append(ValidationResult(
                rule_name="extreme_moves",
                passed=True,
                severity="info",
                message="No extreme price movements detected"
            ))
        
        return results
    
    def _validate_data_completeness(self, data: pd.DataFrame, timeframe: str) -> List[ValidationResult]:
        """Validate data completeness based on timeframe"""
        results = []
        
        if len(data) < 2:
            return results
        
        # Expected frequency based on timeframe
        freq_map = {
            '1m': '1T', '5m': '5T', '15m': '15T', '30m': '30T',
            '1h': '1H', '4h': '4H', '1d': '1D', '1w': '1W'
        }
        
        if timeframe not in freq_map:
            return results
        
        expected_freq = freq_map[timeframe]
        
        # Create expected datetime range
        start_date = data.index.min()
        end_date = data.index.max()
        expected_index = pd.date_range(start=start_date, end=end_date, freq=expected_freq)
        
        # Calculate completeness
        expected_count = len(expected_index)
        actual_count = len(data)
        completeness = actual_count / expected_count if expected_count > 0 else 0
        
        threshold = self._get_rule_threshold("data_completeness", 0.95)
        
        if completeness < threshold:
            missing_count = expected_count - actual_count
            results.append(ValidationResult(
                rule_name="data_completeness",
                passed=False,
                severity="warning",
                message=f"Data completeness below threshold: {completeness:.1%} (missing {missing_count} records)",
                details={
                    'completeness': completeness,
                    'expected_count': expected_count,
                    'actual_count': actual_count,
                    'missing_count': missing_count
                }
            ))
        else:
            results.append(ValidationResult(
                rule_name="data_completeness",
                passed=True,
                severity="info",
                message=f"Data completeness: {completeness:.1%}"
            ))
        
        return results
    
    def _get_rule_threshold(self, rule_name: str, default: float) -> float:
        """Get threshold for a specific rule"""
        rule = next((r for r in self.rules if r.name == rule_name), None)
        return rule.threshold if rule and rule.threshold is not None else default
    
    def _calculate_quality_score(self, results: List[ValidationResult]) -> float:
        """Calculate overall quality score"""
        if not results:
            return 0.0
        
        weights = {'error': 1.0, 'warning': 0.5, 'info': 0.1}
        
        total_weight = 0
        passed_weight = 0
        
        for result in results:
            weight = weights.get(result.severity, 0.1)
            total_weight += weight
            
            if result.passed:
                passed_weight += weight
        
        return (passed_weight / total_weight) if total_weight > 0 else 0.0
    
    def _generate_recommendations(self, results: List[ValidationResult]) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        for result in results:
            if not result.passed:
                if result.rule_name == "missing_data":
                    recommendations.append("Consider forward-filling or interpolating missing values")
                elif result.rule_name == "price_consistency":
                    recommendations.append("Review and clean OHLC price data for consistency")
                elif result.rule_name == "outlier_detection":
                    recommendations.append("Investigate and potentially filter extreme outliers")
                elif result.rule_name == "data_completeness":
                    recommendations.append("Fill missing time periods or adjust analysis window")
                elif result.rule_name == "duplicate_timestamps":
                    recommendations.append("Remove or consolidate duplicate timestamp records")
        
        # Add general recommendations
        quality_issues = [r for r in results if not r.passed and r.severity == 'error']
        if quality_issues:
            recommendations.append("Address critical data quality issues before using for trading")
        
        return recommendations