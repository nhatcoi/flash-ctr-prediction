"""
Data Loader Module for Criteo Dataset

Provides streaming data loading for large-scale CTR prediction.
"""
import os
import gzip
from typing import Iterator, Tuple, List, Optional


class CriteoDataLoader:
    """
    Data loader for Criteo Click Logs dataset.
    
    Dataset Format (TSV):
    - Column 0: Label (1 = click, 0 = no click)
    - Columns 1-13: Integer features (I1-I13) - numerical
    - Columns 14-39: Categorical features (C1-C26) - hashed strings
    
    Example:
        loader = CriteoDataLoader('data/train.txt', batch_size=1000)
        for batch in loader:
            labels, features = batch
            # Process batch
    """
    
    NUM_INT_FEATURES = 13
    NUM_CAT_FEATURES = 26
    TOTAL_FEATURES = NUM_INT_FEATURES + NUM_CAT_FEATURES
    
    def __init__(self, 
                 filepath: str,
                 batch_size: int = 1024,
                 shuffle: bool = False,
                 max_samples: Optional[int] = None):
        """
        Initialize the data loader.
        
        Args:
            filepath: Path to the data file (TSV or GZ format)
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data (not recommended for streaming)
            max_samples: Maximum number of samples to load (None = all)
        """
        self.filepath = filepath
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_samples = max_samples
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
    
    def __iter__(self) -> Iterator[Tuple[List[int], List[List]]]:
        """
        Iterate over the dataset in batches.
        
        Yields:
            Tuple of (labels, features) for each batch
        """
        labels = []
        features = []
        sample_count = 0
        
        # Determine if file is compressed
        is_gz = self.filepath.endswith('.gz')
        open_func = gzip.open if is_gz else open
        mode = 'rt' if is_gz else 'r'
        
        with open_func(self.filepath, mode, encoding='utf-8') as f:
            for line in f:
                if self.max_samples and sample_count >= self.max_samples:
                    break
                
                parsed = self._parse_line(line)
                if parsed is not None:
                    label, feat = parsed
                    labels.append(label)
                    features.append(feat)
                    sample_count += 1
                    
                    if len(labels) >= self.batch_size:
                        yield labels, features
                        labels = []
                        features = []
        
        # Yield remaining samples
        if labels:
            yield labels, features
    
    def _parse_line(self, line: str) -> Optional[Tuple[int, List]]:
        """
        Parse a single line from the dataset.
        
        Args:
            line: Raw line from TSV file
            
        Returns:
            Tuple of (label, features) or None if parsing fails
        """
        try:
            parts = line.strip().split('\t')
            if len(parts) < self.TOTAL_FEATURES + 1:
                return None
            
            label = int(parts[0])
            
            # Parse integer features (handle missing values as -1)
            int_features = []
            for i in range(1, self.NUM_INT_FEATURES + 1):
                if parts[i] == '':
                    int_features.append(-1)  # Missing value marker
                else:
                    int_features.append(int(parts[i]))
            
            # Parse categorical features (keep as strings)
            cat_features = []
            for i in range(self.NUM_INT_FEATURES + 1, len(parts)):
                cat_features.append(parts[i] if parts[i] else '')
            
            return label, int_features + cat_features
            
        except (ValueError, IndexError):
            return None


class StreamingIterator:
    """
    Memory-efficient streaming iterator for processing one sample at a time.
    
    Useful for online learning where we update the model after each sample.
    
    Example:
        iterator = StreamingIterator('data/train.txt')
        for label, features in iterator:
            model.partial_fit(features, label)
    """
    
    def __init__(self, filepath: str, max_samples: Optional[int] = None):
        """
        Initialize the streaming iterator.
        
        Args:
            filepath: Path to the data file
            max_samples: Maximum number of samples to iterate
        """
        self.filepath = filepath
        self.max_samples = max_samples
        self.loader = CriteoDataLoader(filepath, batch_size=1, max_samples=max_samples)
    
    def __iter__(self) -> Iterator[Tuple[int, List]]:
        """
        Iterate one sample at a time.
        
        Yields:
            Tuple of (label, features) for each sample
        """
        for labels, features in self.loader:
            yield labels[0], features[0]
    
    def count_lines(self) -> int:
        """
        Count total number of lines in file (for progress tracking).
        
        Returns:
            Number of lines in the file
        """
        count = 0
        is_gz = self.filepath.endswith('.gz')
        open_func = gzip.open if is_gz else open
        mode = 'rt' if is_gz else 'r'
        
        with open_func(self.filepath, mode, encoding='utf-8') as f:
            for _ in f:
                count += 1
        return count


def create_sample_data(output_path: str, num_samples: int = 1000):
    """
    Create sample data for testing (simulates Criteo format).
    
    Args:
        output_path: Path to save the sample data
        num_samples: Number of samples to generate
    """
    import random
    import hashlib
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        for _ in range(num_samples):
            # Random label (skewed towards 0, ~3% CTR like real Criteo)
            label = 1 if random.random() < 0.03 else 0
            
            # Random integer features
            int_features = []
            for _ in range(13):
                if random.random() < 0.1:  # 10% missing
                    int_features.append('')
                else:
                    int_features.append(str(random.randint(0, 1000)))
            
            # Random categorical features (hashed)
            cat_features = []
            for i in range(26):
                if random.random() < 0.1:  # 10% missing
                    cat_features.append('')
                else:
                    val = f"cat_{i}_{random.randint(0, 100)}"
                    hashed = hashlib.md5(val.encode()).hexdigest()[:8]
                    cat_features.append(hashed)
            
            line = '\t'.join([str(label)] + int_features + cat_features)
            f.write(line + '\n')
    
    print(f"Created sample data: {output_path} ({num_samples} samples)")


if __name__ == '__main__':
    # Test: Create sample data and verify loading
    sample_path = 'data/sample/train.txt'
    create_sample_data(sample_path, num_samples=100)
    
    # Test batch loading
    print("\nTesting CriteoDataLoader:")
    loader = CriteoDataLoader(sample_path, batch_size=10)
    for i, (labels, features) in enumerate(loader):
        print(f"  Batch {i}: {len(labels)} samples, label sum = {sum(labels)}")
        if i >= 2:
            break
    
    # Test streaming
    print("\nTesting StreamingIterator:")
    iterator = StreamingIterator(sample_path, max_samples=5)
    for label, features in iterator:
        print(f"  Label: {label}, Features count: {len(features)}")
