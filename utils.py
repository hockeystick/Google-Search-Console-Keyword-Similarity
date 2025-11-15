"""
Shared utility functions for keyword similarity analysis.
"""
import logging
from typing import Optional
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


def calculate_cosine_similarity(
    data: pd.DataFrame,
    column_name: str,
    min_df: int = 1,
    max_df: float = 1.0,
    ngram_range: tuple = (1, 2)
) -> pd.DataFrame:
    """
    Compute TF-IDF vectors and cosine similarity.

    Args:
        data: DataFrame containing keyword data
        column_name: Name of the column with keywords
        min_df: Minimum document frequency
        max_df: Maximum document frequency (proportion)
        ngram_range: Range of n-grams to consider (default: unigrams and bigrams)

    Returns:
        DataFrame with cosine similarity matrix

    Raises:
        ValueError: If data is invalid or column doesn't exist
    """
    # Validate input
    if data is None or data.empty:
        raise ValueError("Data cannot be None or empty")

    if column_name not in data.columns:
        raise ValueError(f"Column '{column_name}' not found in data. Available columns: {list(data.columns)}")

    # Check for null values
    if data[column_name].isnull().any():
        logger.warning(f"Null values found in column '{column_name}'. Removing them.")
        data = data.dropna(subset=[column_name])

    # Check for duplicates
    if data[column_name].duplicated().any():
        logger.warning(f"Duplicate keywords found. Keeping first occurrence.")
        data = data.drop_duplicates(subset=[column_name], keep='first')

    # Check minimum data requirement
    unique_count = len(data[column_name].unique())
    if unique_count < 2:
        raise ValueError(f"Need at least 2 unique keywords for analysis. Found: {unique_count}")

    logger.info(f"Processing {len(data)} unique keywords for similarity analysis")

    try:
        # TF-IDF Vectorization
        tfidf_vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            stop_words='english'
        )
        tfidf_matrix = tfidf_vectorizer.fit_transform(data[column_name])

        # Cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)

        # Convert to DataFrame
        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=data[column_name],
            columns=data[column_name]
        )

        logger.info(f"Successfully computed similarity matrix: {similarity_df.shape}")
        return similarity_df

    except Exception as e:
        logger.error(f"Error computing similarity: {str(e)}")
        raise


def validate_dataframe(data: pd.DataFrame, column_name: str) -> tuple[bool, Optional[str]]:
    """
    Validate DataFrame for keyword analysis.

    Args:
        data: DataFrame to validate
        column_name: Column name to check

    Returns:
        Tuple of (is_valid, error_message)
    """
    if data is None or data.empty:
        return False, "The uploaded file is empty or invalid."

    if column_name not in data.columns:
        return False, f"Column '{column_name}' not found. Available columns: {', '.join(data.columns)}"

    if data[column_name].isnull().all():
        return False, f"Column '{column_name}' contains only null values."

    non_null_count = data[column_name].notna().sum()
    if non_null_count < 2:
        return False, f"Need at least 2 non-null keywords. Found: {non_null_count}"

    unique_count = data[column_name].nunique()
    if unique_count < 2:
        return False, f"Need at least 2 unique keywords. Found: {unique_count}"

    return True, None


def get_top_keyword_pairs(similarity_df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """
    Extract top keyword pairs by similarity score.

    Args:
        similarity_df: Similarity matrix DataFrame
        top_n: Number of top pairs to return

    Returns:
        DataFrame with keyword pairs and their similarity scores
    """
    pairs = []
    for i in range(len(similarity_df)):
        for j in range(i + 1, len(similarity_df)):
            pairs.append({
                'keyword1': similarity_df.index[i],
                'keyword2': similarity_df.columns[j],
                'similarity': similarity_df.iloc[i, j]
            })

    pairs_df = pd.DataFrame(pairs).sort_values('similarity', ascending=False)
    return pairs_df.head(top_n)
