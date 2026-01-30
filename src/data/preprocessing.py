"""
Preprocessing Module for CTR Prediction

Implements Feature Hashing (Hashing Trick) and numerical transformations.
"""
import math
import hashlib
from typing import List, Dict, Optional, Tuple
import numpy as np


class FeatureHasher:
    """
    Feature Hashing (Hashing Trick) for high-dimensional sparse features.
    
    Maps any feature to a fixed-size hash space, enabling:
    - O(1) memory per feature (no dictionary needed)
    - Handling of unseen features at test time
    - Collision-based dimensionality reduction
    
    Mathematical formulation:
        h(x_i) = hash(feature_name + ":" + feature_value) mod num_buckets
        sign(x_i) = +1 or -1 based on secondary hash (for unbiased estimation)
        
    Reference:
        Weinberger et al., "Feature Hashing for Large Scale Multitask Learning"
    
    Example:
        hasher = FeatureHasher(num_buckets=2**18)
        sparse_vector = hasher.transform(['cat1:abc', 'cat2:def', 'int1:5'])
    """
    
    def __init__(self, 
                 num_buckets: int = 2**20,
                 use_sign: bool = True):
        """
        Initialize the feature hasher.
        
        Args:
            num_buckets: Size of hash space (power of 2 recommended)
            use_sign: Whether to use signed hashing (reduces collision bias)
        """
        self.num_buckets = num_buckets
        self.use_sign = use_sign
    
    def _hash(self, value: str) -> int:
        """
        Hash a string to a bucket index.
        
        Args:
            value: String to hash
            
        Returns:
            Bucket index in [0, num_buckets)
        """
        return int(hashlib.md5(value.encode()).hexdigest(), 16) % self.num_buckets
    
    def _sign(self, value: str) -> int:
        """
        Get sign for signed hashing.
        
        Args:
            value: String to compute sign for
            
        Returns:
            +1 or -1
        """
        if not self.use_sign:
            return 1
        # Use different hash function for sign
        h = int(hashlib.sha256(value.encode()).hexdigest(), 16)
        return 1 if h % 2 == 0 else -1
    
    def hash_feature(self, name: str, value: str) -> Tuple[int, int]:
        """
        Hash a single feature (name:value pair).
        
        Args:
            name: Feature name (e.g., 'C1', 'I5')
            value: Feature value
            
        Returns:
            Tuple of (bucket_index, sign)
        """
        key = f"{name}:{value}"
        return self._hash(key), self._sign(key)
    
    def transform(self, 
                  int_features: List, 
                  cat_features: List,
                  int_feature_names: Optional[List[str]] = None,
                  cat_feature_names: Optional[List[str]] = None) -> Dict[int, float]:
        """
        Transform raw features to sparse hashed representation.
        
        Args:
            int_features: List of 13 integer values
            cat_features: List of 26 categorical values
            int_feature_names: Custom names for int features
            cat_feature_names: Custom names for cat features
            
        Returns:
            Sparse dict {bucket_index: value}
        """
        sparse = {}
        
        # Default feature names
        if int_feature_names is None:
            int_feature_names = [f"I{i+1}" for i in range(len(int_features))]
        if cat_feature_names is None:
            cat_feature_names = [f"C{i+1}" for i in range(len(cat_features))]
        
        # Hash integer features (bucket by value ranges)
        for name, value in zip(int_feature_names, int_features):
            if value == -1 or value == '':  # Missing value
                continue
            
            # Convert to log scale and bucket
            try:
                val = float(value)
                if val > 0:
                    val = math.log1p(val)
                bucket_idx, sign = self.hash_feature(name, f"log_{int(val)}")
                sparse[bucket_idx] = sparse.get(bucket_idx, 0) + sign * val
            except (ValueError, TypeError):
                continue
        
        # Hash categorical features
        for name, value in zip(cat_feature_names, cat_features):
            if value == '' or value is None:  # Missing value
                continue
            
            bucket_idx, sign = self.hash_feature(name, str(value))
            sparse[bucket_idx] = sparse.get(bucket_idx, 0) + sign * 1.0
        
        # Add bias term
        bias_idx = self._hash("__BIAS__")
        sparse[bias_idx] = 1.0
        
        return sparse


class LogTransformer:
    """
    Log transformation for numerical features.
    
    Transforms skewed distributions to more Gaussian-like:
        x' = log(1 + x) for x >= 0
        x' = -log(1 - x) for x < 0
    """
    
    @staticmethod
    def transform(value: float) -> float:
        """
        Apply log transformation.
        
        Args:
            value: Input value
            
        Returns:
            Transformed value
        """
        if value >= 0:
            return math.log1p(value)
        else:
            return -math.log1p(-value)
    
    @staticmethod
    def transform_array(values: List[float]) -> List[float]:
        """
        Transform an array of values.
        
        Args:
            values: List of input values
            
        Returns:
            List of transformed values
        """
        return [LogTransformer.transform(v) for v in values]


class MissingValueHandler:
    """
    Handle missing values in Criteo dataset.
    
    Strategies:
    - numerical: Replace with median or special marker (-1)
    - categorical: Replace with special token '__MISSING__'
    """
    
    def __init__(self, 
                 numerical_strategy: str = 'marker',
                 categorical_strategy: str = 'token'):
        """
        Initialize the handler.
        
        Args:
            numerical_strategy: 'marker' (-1) or 'zero'
            categorical_strategy: 'token' or 'drop'
        """
        self.numerical_strategy = numerical_strategy
        self.categorical_strategy = categorical_strategy
    
    def handle_numerical(self, value) -> float:
        """Handle missing numerical value."""
        if value == '' or value == -1 or value is None:
            if self.numerical_strategy == 'marker':
                return -1
            else:
                return 0.0
        return float(value)
    
    def handle_categorical(self, value: str) -> str:
        """Handle missing categorical value."""
        if value == '' or value is None:
            if self.categorical_strategy == 'token':
                return '__MISSING__'
            else:
                return ''
        return value


class Preprocessor:
    """
    Combined preprocessor for Criteo dataset.
    
    Pipeline:
    1. Handle missing values
    2. Apply log transform to numerical features
    3. Apply feature hashing
    
    Example:
        preprocessor = Preprocessor(num_buckets=2**18)
        sparse_features = preprocessor.transform(raw_features)
    """
    
    NUM_INT_FEATURES = 13
    NUM_CAT_FEATURES = 26
    
    def __init__(self, num_buckets: int = 2**20, use_sign: bool = True):
        """
        Initialize the preprocessor.
        
        Args:
            num_buckets: Hash space size
            use_sign: Use signed hashing
        """
        self.num_buckets = num_buckets
        self.hasher = FeatureHasher(num_buckets, use_sign)
        self.missing_handler = MissingValueHandler()
        self.log_transformer = LogTransformer()
    
    def transform(self, raw_features: List) -> Dict[int, float]:
        """
        Transform raw features to sparse representation.
        
        Args:
            raw_features: List of 39 feature values (13 int + 26 cat)
            
        Returns:
            Sparse feature dict
        """
        # Split features
        int_features = raw_features[:self.NUM_INT_FEATURES]
        cat_features = raw_features[self.NUM_INT_FEATURES:]
        
        # Handle missing and transform numerical
        processed_int = []
        for v in int_features:
            v = self.missing_handler.handle_numerical(v)
            if v > 0:
                v = self.log_transformer.transform(v)
            processed_int.append(v)
        
        # Handle missing categorical
        processed_cat = [self.missing_handler.handle_categorical(v) for v in cat_features]
        
        # Apply feature hashing
        return self.hasher.transform(processed_int, processed_cat)


if __name__ == '__main__':
    # Test preprocessing pipeline
    print("Testing Preprocessor:")
    
    # Sample feature vector (13 int + 26 cat)
    sample_int = [1, 5, '', 100, 50, 0, '', 3, 2, 1, 0, 8, 10]
    sample_cat = ['abc123', 'def456', '', 'ghi789'] + [''] * 22
    sample_features = sample_int + sample_cat
    
    preprocessor = Preprocessor(num_buckets=2**18)
    sparse = preprocessor.transform(sample_features)
    
    print(f"  Input: {len(sample_features)} features")
    print(f"  Output: {len(sparse)} non-zero buckets")
    print(f"  Sample buckets: {list(sparse.items())[:5]}")
