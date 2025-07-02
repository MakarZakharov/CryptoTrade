"""
Feature selection for market analysis.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler


class FeatureSelector:
    """
    Class for selecting relevant features for market analysis.
    """
    
    def __init__(self, target_column: str = 'close'):
        """
        Initialize the feature selector.
        
        Args:
            target_column: Column to use as target for feature selection
        """
        self.target_column = target_column
        self.selected_features = None
        self.feature_importances = None
    
    def select_features(self, df: pd.DataFrame, method: str = 'random_forest', n_features: Optional[int] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select the most relevant features from the DataFrame.
        
        Args:
            df: DataFrame with features
            method: Feature selection method ('random_forest', 'f_regression', 'mutual_info')
            n_features: Number of features to select (default: 1/3 of available features)
            
        Returns:
            Tuple of (DataFrame with selected features, list of selected feature names)
        """
        # Make a copy of the DataFrame
        data = df.copy()
        
        # Ensure the target column is in the DataFrame
        if self.target_column not in data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in DataFrame")
        
        # Remove NaN values
        data = data.dropna()
        
        # Separate features and target
        X = data.drop(columns=[self.target_column])
        y = data[self.target_column]
        
        # Set default number of features if not specified
        if n_features is None:
            n_features = max(int(X.shape[1] / 3), 5)  # At least 5 features or 1/3 of available features
        
        # Ensure n_features is not greater than the number of available features
        n_features = min(n_features, X.shape[1])
        
        # Select features based on the specified method
        if method == 'random_forest':
            selected_features, feature_importances = self._select_with_random_forest(X, y, n_features)
        elif method == 'f_regression':
            selected_features, feature_importances = self._select_with_f_regression(X, y, n_features)
        elif method == 'mutual_info':
            selected_features, feature_importances = self._select_with_mutual_info(X, y, n_features)
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        # Store the selected features and their importances
        self.selected_features = selected_features
        self.feature_importances = feature_importances
        
        # Return the DataFrame with only the selected features and the target
        selected_columns = selected_features + [self.target_column]
        return data[selected_columns], selected_features
    
    def _select_with_random_forest(self, X: pd.DataFrame, y: pd.Series, n_features: int) -> Tuple[List[str], pd.Series]:
        """
        Select features using Random Forest feature importance.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            n_features: Number of features to select
            
        Returns:
            Tuple of (list of selected feature names, Series of feature importances)
        """
        # Train a Random Forest model
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Get feature importances
        importances = pd.Series(rf.feature_importances_, index=X.columns)
        importances = importances.sort_values(ascending=False)
        
        # Select the top n_features
        selected_features = importances.index[:n_features].tolist()
        
        return selected_features, importances
    
    def _select_with_f_regression(self, X: pd.DataFrame, y: pd.Series, n_features: int) -> Tuple[List[str], pd.Series]:
        """
        Select features using F-regression.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            n_features: Number of features to select
            
        Returns:
            Tuple of (list of selected feature names, Series of feature importances)
        """
        # Apply F-regression
        selector = SelectKBest(f_regression, k=n_features)
        selector.fit(X, y)
        
        # Get feature scores
        scores = pd.Series(selector.scores_, index=X.columns)
        scores = scores.sort_values(ascending=False)
        
        # Select the top n_features
        selected_features = scores.index[:n_features].tolist()
        
        return selected_features, scores
    
    def _select_with_mutual_info(self, X: pd.DataFrame, y: pd.Series, n_features: int) -> Tuple[List[str], pd.Series]:
        """
        Select features using Mutual Information.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            n_features: Number of features to select
            
        Returns:
            Tuple of (list of selected feature names, Series of feature importances)
        """
        # Apply Mutual Information
        selector = SelectKBest(mutual_info_regression, k=n_features)
        selector.fit(X, y)
        
        # Get feature scores
        scores = pd.Series(selector.scores_, index=X.columns)
        scores = scores.sort_values(ascending=False)
        
        # Select the top n_features
        selected_features = scores.index[:n_features].tolist()
        
        return selected_features, scores
    
    def plot_feature_importances(self, top_n: int = 20) -> None:
        """
        Plot the feature importances.
        
        Args:
            top_n: Number of top features to plot
        """
        if self.feature_importances is None:
            raise ValueError("No feature importances available. Run select_features first.")
        
        # Import matplotlib only when needed
        import matplotlib.pyplot as plt
        
        # Get the top N features
        top_features = self.feature_importances.sort_values(ascending=False).head(top_n)
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        top_features.plot(kind='barh')
        plt.title(f'Top {top_n} Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()
    
    def get_feature_correlation_matrix(self, df: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
        """
        Calculate the correlation matrix for the features.
        
        Args:
            df: DataFrame with features
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Correlation matrix DataFrame
        """
        # Make a copy of the DataFrame
        data = df.copy()
        
        # Calculate the correlation matrix
        corr_matrix = data.corr(method=method)
        
        return corr_matrix
    
    def plot_correlation_matrix(self, df: pd.DataFrame, method: str = 'pearson') -> None:
        """
        Plot the correlation matrix for the features.
        
        Args:
            df: DataFrame with features
            method: Correlation method ('pearson', 'spearman', 'kendall')
        """
        # Import matplotlib and seaborn only when needed
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Calculate the correlation matrix
        corr_matrix = self.get_feature_correlation_matrix(df, method)
        
        # Create the plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
        plt.title(f'Feature Correlation Matrix ({method})')
        plt.tight_layout()
        plt.show()
    
    def remove_highly_correlated_features(self, df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """
        Remove highly correlated features from the DataFrame.
        
        Args:
            df: DataFrame with features
            threshold: Correlation threshold for removal
            
        Returns:
            DataFrame with highly correlated features removed
        """
        # Make a copy of the DataFrame
        data = df.copy()
        
        # Calculate the correlation matrix
        corr_matrix = data.corr().abs()
        
        # Create a mask for the upper triangle
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation greater than the threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        # Drop the highly correlated features
        data_reduced = data.drop(columns=to_drop)
        
        print(f"Removed {len(to_drop)} highly correlated features: {to_drop}")
        
        return data_reduced