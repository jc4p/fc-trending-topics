import sys
import os
import pandas as pd
import numpy as np
import unittest
from unittest.mock import patch, MagicMock

# Add src directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the functions we want to test
from src.data_preprocessing import filter_with_embeddings, funnel_filter_posts, generate_embeddings

class TestFunnelFiltering(unittest.TestCase):
    """Tests for the funnel filtering approach using SentenceTransformer embeddings"""
    
    def setUp(self):
        """Set up test data"""
        # Create a sample DataFrame with posts
        # The posts are designed to include some near-duplicates
        self.texts = [
            "This is a unique post about something interesting",
            "This is a nearly identical post about something interesting",  # Near-duplicate of first text
            "Another completely different topic about crypto",
            "More discussion about crypto and tokens",  # Similar topic but different text
            "Completely unrelated content about cooking recipes",
            "Here is a post about crypto trading strategies",  # Similar to #3
            "Let's talk about cooking delicious meals at home",  # Similar to #5
            "The weather is really nice today",
            "Have you seen the weather forecast for tomorrow?",  # Related to #8
            "What's your opinion on the new crypto regulations?",  # Related to #3
        ] * 3  # Repeat to get enough data
        
        # Expand the DataFrame with some fields we need
        self.df = pd.DataFrame({
            'Text': self.texts,
            'cleaned_text': self.texts,
            'Hash': [f'hash{i}' for i in range(len(self.texts))],
            'ParentCastId': [''] * len(self.texts),  # All top-level posts
            'engagement_score': np.random.rand(len(self.texts)) * 10  # Random engagement scores
        })
        
        # Create a mock DuckDB connection
        self.mock_conn = MagicMock()
        self.mock_conn.execute.return_value = MagicMock()
        self.mock_conn.execute.return_value.df.return_value = pd.DataFrame()
        
    @patch('src.data_preprocessing.generate_embeddings')
    def test_filter_with_embeddings(self, mock_generate_embeddings):
        """Test the filter_with_embeddings function with mock embeddings"""
        # Create mock embeddings (2D array with shape [n_samples, n_features])
        # We'll make similar texts have similar embeddings
        
        # Simple embeddings for testing - only 3-dimensional for simplicity
        mock_embeddings = np.array([
            # First group (near duplicates)
            [0.9, 0.1, 0.0],  # Text 0
            [0.88, 0.12, 0.0],  # Text 1 - similar to Text 0
            
            # Second group (crypto)
            [0.1, 0.9, 0.0],  # Text 2
            [0.15, 0.85, 0.0],  # Text 3 - similar to Text 2
            
            # Unique posts
            [0.0, 0.1, 0.9],  # Text 4
            
            # More from second group
            [0.2, 0.8, 0.0],  # Text 5 - similar to Text 2-3
            
            # Similar to unique
            [0.1, 0.2, 0.7],  # Text 6 - somewhat similar to Text 4
            
            # Weather group
            [0.5, 0.0, 0.5],  # Text 7
            [0.45, 0.05, 0.5],  # Text 8 - similar to Text 7
            
            # More crypto
            [0.1, 0.85, 0.05]  # Text 9 - similar to Text 2-3
        ] * 3)  # Repeat to match length
        
        # Test with high similarity threshold (0.95)
        filtered_df = filter_with_embeddings(
            self.df,
            mock_embeddings,
            sample_size=20,
            similarity_threshold=0.95
        )
        
        # With high threshold, we should keep more posts
        # Only exact duplicates should be filtered out
        self.assertGreater(len(filtered_df), 15)
        
        # Test with lower similarity threshold (0.85)
        filtered_df_strict = filter_with_embeddings(
            self.df,
            mock_embeddings,
            sample_size=20,
            similarity_threshold=0.85
        )
        
        # With lower threshold, we should filter more aggressively
        self.assertLess(len(filtered_df_strict), len(filtered_df))
    
    @patch('src.data_preprocessing.generate_embeddings')
    def test_funnel_filter_posts(self, mock_generate_embeddings):
        """Test the funnel_filter_posts function"""
        # Create mock embeddings
        mock_embeddings = np.random.rand(len(self.df), 384)  # Typical embedding size
        mock_model = MagicMock()
        
        # Configure the mock to return our mock embeddings
        mock_generate_embeddings.return_value = (mock_embeddings, mock_model)
        
        # Run the function with a target size smaller than our data
        target_size = 10
        filtered_df = funnel_filter_posts(
            self.mock_conn,
            self.df,
            target_sample_size=target_size
        )
        
        # It should return approximately the target size (or slightly less)
        self.assertLessEqual(len(filtered_df), target_size)
        
        # Make sure generate_embeddings was called
        mock_generate_embeddings.assert_called_once()
        
if __name__ == '__main__':
    unittest.main()