"""
Unit tests for utils module.
"""
import pytest
import pandas as pd
import numpy as np
from utils import (
    calculate_cosine_similarity,
    validate_dataframe,
    get_top_keyword_pairs
)


class TestCalculateCosineSimilarity:
    """Tests for calculate_cosine_similarity function."""

    def test_basic_similarity_calculation(self):
        """Test basic similarity calculation with valid data."""
        data = pd.DataFrame({
            'keywords': [
                'machine learning',
                'deep learning',
                'artificial intelligence',
                'data science'
            ]
        })
        result = calculate_cosine_similarity(data, 'keywords')

        # Check result shape
        assert result.shape == (4, 4)

        # Check diagonal is all 1s (self-similarity)
        assert np.allclose(np.diag(result.values), 1.0)

        # Check symmetry
        assert np.allclose(result.values, result.values.T)

        # Check values are between 0 and 1
        assert result.values.min() >= 0
        assert result.values.max() <= 1

    def test_similar_keywords_have_high_similarity(self):
        """Test that similar keywords have high similarity scores."""
        data = pd.DataFrame({
            'keywords': [
                'machine learning',
                'machine learning algorithms',
                'basketball'
            ]
        })
        result = calculate_cosine_similarity(data, 'keywords')

        # Similarity between 'machine learning' and 'machine learning algorithms'
        # should be higher than with 'basketball'
        ml_similarity = result.iloc[0, 1]
        basketball_similarity = result.iloc[0, 2]

        assert ml_similarity > basketball_similarity

    def test_empty_dataframe_raises_error(self):
        """Test that empty DataFrame raises ValueError."""
        data = pd.DataFrame()

        with pytest.raises(ValueError, match="Data cannot be None or empty"):
            calculate_cosine_similarity(data, 'keywords')

    def test_missing_column_raises_error(self):
        """Test that missing column raises ValueError."""
        data = pd.DataFrame({'keywords': ['test']})

        with pytest.raises(ValueError, match="Column 'missing' not found"):
            calculate_cosine_similarity(data, 'missing')

    def test_null_values_are_handled(self):
        """Test that null values are removed automatically."""
        data = pd.DataFrame({
            'keywords': ['machine learning', None, 'deep learning', 'AI']
        })
        result = calculate_cosine_similarity(data, 'keywords')

        # Should have 3x3 matrix (null removed)
        assert result.shape == (3, 3)

    def test_duplicates_are_handled(self):
        """Test that duplicate keywords are handled."""
        data = pd.DataFrame({
            'keywords': [
                'machine learning',
                'machine learning',
                'deep learning'
            ]
        })
        result = calculate_cosine_similarity(data, 'keywords')

        # Should have 2x2 matrix (duplicate removed)
        assert result.shape == (2, 2)

    def test_minimum_keywords_requirement(self):
        """Test that at least 2 unique keywords are required."""
        data = pd.DataFrame({'keywords': ['single keyword']})

        with pytest.raises(ValueError, match="Need at least 2 unique keywords"):
            calculate_cosine_similarity(data, 'keywords')


class TestValidateDataframe:
    """Tests for validate_dataframe function."""

    def test_valid_dataframe(self):
        """Test validation of valid DataFrame."""
        data = pd.DataFrame({
            'keywords': ['keyword1', 'keyword2', 'keyword3']
        })
        is_valid, error_msg = validate_dataframe(data, 'keywords')

        assert is_valid is True
        assert error_msg is None

    def test_empty_dataframe(self):
        """Test validation of empty DataFrame."""
        data = pd.DataFrame()
        is_valid, error_msg = validate_dataframe(data, 'keywords')

        assert is_valid is False
        assert "empty or invalid" in error_msg

    def test_missing_column(self):
        """Test validation with missing column."""
        data = pd.DataFrame({'keywords': ['test']})
        is_valid, error_msg = validate_dataframe(data, 'missing')

        assert is_valid is False
        assert "not found" in error_msg

    def test_all_null_values(self):
        """Test validation with all null values."""
        data = pd.DataFrame({'keywords': [None, None, None]})
        is_valid, error_msg = validate_dataframe(data, 'keywords')

        assert is_valid is False
        assert "only null values" in error_msg

    def test_insufficient_non_null_keywords(self):
        """Test validation with insufficient non-null keywords."""
        data = pd.DataFrame({'keywords': ['keyword1', None, None]})
        is_valid, error_msg = validate_dataframe(data, 'keywords')

        assert is_valid is False
        assert "Need at least 2 non-null keywords" in error_msg

    def test_insufficient_unique_keywords(self):
        """Test validation with insufficient unique keywords."""
        data = pd.DataFrame({'keywords': ['same', 'same', 'same']})
        is_valid, error_msg = validate_dataframe(data, 'keywords')

        assert is_valid is False
        assert "Need at least 2 unique keywords" in error_msg


class TestGetTopKeywordPairs:
    """Tests for get_top_keyword_pairs function."""

    def test_basic_top_pairs(self):
        """Test basic extraction of top keyword pairs."""
        # Create a simple similarity matrix
        data = pd.DataFrame(
            [[1.0, 0.9, 0.3],
             [0.9, 1.0, 0.2],
             [0.3, 0.2, 1.0]],
            index=['keyword1', 'keyword2', 'keyword3'],
            columns=['keyword1', 'keyword2', 'keyword3']
        )

        result = get_top_keyword_pairs(data, top_n=3)

        # Should have 3 pairs (excluding diagonal)
        assert len(result) == 3

        # Check columns
        assert 'keyword1' in result.columns
        assert 'keyword2' in result.columns
        assert 'similarity' in result.columns

        # Top pair should be keyword1-keyword2 with similarity 0.9
        assert result.iloc[0]['similarity'] == 0.9

    def test_top_n_limit(self):
        """Test that top_n parameter limits results."""
        data = pd.DataFrame(
            [[1.0, 0.9, 0.8, 0.7],
             [0.9, 1.0, 0.6, 0.5],
             [0.8, 0.6, 1.0, 0.4],
             [0.7, 0.5, 0.4, 1.0]],
            index=['k1', 'k2', 'k3', 'k4'],
            columns=['k1', 'k2', 'k3', 'k4']
        )

        result = get_top_keyword_pairs(data, top_n=2)

        # Should only have 2 pairs
        assert len(result) == 2

    def test_sorted_by_similarity(self):
        """Test that results are sorted by similarity in descending order."""
        data = pd.DataFrame(
            [[1.0, 0.5, 0.9],
             [0.5, 1.0, 0.3],
             [0.9, 0.3, 1.0]],
            index=['k1', 'k2', 'k3'],
            columns=['k1', 'k2', 'k3']
        )

        result = get_top_keyword_pairs(data, top_n=3)

        # Check that similarities are in descending order
        similarities = result['similarity'].values
        assert all(similarities[i] >= similarities[i+1] for i in range(len(similarities)-1))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
