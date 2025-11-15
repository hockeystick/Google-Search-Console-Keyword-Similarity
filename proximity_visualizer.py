"""
Proximity Visualizer and Insights Maker
A Streamlit application for visualizing keyword similarity matrices and generating insights.
"""
import logging
from typing import List, Tuple
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

from config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_similarity_matrix(
    data: pd.DataFrame,
    query_column: str = 'Top queries'
) -> Tuple[bool, str, pd.DataFrame, np.ndarray]:
    """
    Validate and extract similarity matrix from uploaded data.

    Args:
        data: DataFrame containing similarity matrix
        query_column: Name of the column containing query names

    Returns:
        Tuple of (is_valid, error_message, queries_series, matrix_array)
    """
    try:
        # Check if query column exists
        if query_column not in data.columns:
            available_cols = ", ".join(data.columns)
            return (
                False,
                f"Column '{query_column}' not found. Available columns: {available_cols}",
                pd.Series(),
                np.array([])
            )

        # Extract queries and matrix
        queries = data[query_column]
        matrix = data.iloc[:, 1:].values

        # Validate matrix is square
        if matrix.shape[0] != matrix.shape[1]:
            return (
                False,
                f"Matrix is not square. Expected {matrix.shape[0]}x{matrix.shape[0]}, got {matrix.shape}",
                pd.Series(),
                np.array([])
            )

        # Check for valid similarity values (0 to 1)
        if matrix.min() < 0 or matrix.max() > 1:
            logger.warning(f"Similarity values outside [0,1] range: min={matrix.min()}, max={matrix.max()}")

        return True, "", queries, matrix

    except Exception as e:
        logger.error(f"Error validating matrix: {str(e)}")
        return False, f"Error validating matrix: {str(e)}", pd.Series(), np.array([])


def get_top_matches(
    query: str,
    queries: pd.Series,
    matrix: np.ndarray,
    top_n: int = 5
) -> List[Tuple[str, float]]:
    """
    Get top matching queries for a given query.

    Args:
        query: Query to find matches for
        queries: Series of all queries
        matrix: Similarity matrix
        top_n: Number of top matches to return

    Returns:
        List of (query, similarity_score) tuples
    """
    query_index = queries[queries == query].index[0]
    similarities = matrix[query_index]

    # Create list of (query, score) tuples and sort
    matches = list(zip(queries, similarities))
    matches_sorted = sorted(matches, key=lambda x: x[1], reverse=True)

    # Exclude the query itself (first match)
    return matches_sorted[1:top_n + 1]


def main():
    """Main application function."""
    st.set_page_config(
        page_title="Proximity Visualizer",
        page_icon="📊",
        layout=config.LAYOUT
    )
    st.title("📊 Proximity Visualizer and Insights Maker")
    st.write("Upload your cosine similarity data and visualize relationships between keywords.")

    # File Upload
    uploaded_file = st.file_uploader(
        "Upload CSV File (Cosine Similarity Matrix)",
        type=["csv"],
        help="Upload a CSV file with a similarity matrix (first column: query names, rest: similarity values)"
    )

    if uploaded_file:
        try:
            # Read the uploaded file
            data = pd.read_csv(uploaded_file)

            if data.empty:
                st.error("The uploaded file is empty.")
                return

            st.write("### Data Preview:")
            st.dataframe(data.head())
            st.caption(f"Total rows: {len(data)}")

            # Validate and extract matrix
            is_valid, error_msg, queries, matrix = validate_similarity_matrix(data)

            if not is_valid:
                st.error(error_msg)
                return

            logger.info(f"Loaded similarity matrix: {matrix.shape}")

            # Generate Heatmap
            st.write("### Heatmap of Similarities")
            st.write("A heatmap showing the proximity between the top queries.")

            # Allow the user to select the number of queries to visualize
            max_queries = min(len(queries), config.MAX_HEATMAP_SIZE)
            top_n = st.slider(
                "Select the number of queries to visualize:",
                min_value=config.MIN_HEATMAP_SIZE,
                max_value=max_queries,
                value=min(config.DEFAULT_HEATMAP_SIZE, max_queries)
            )

            selected_queries = queries[:top_n]
            selected_matrix = matrix[:top_n, :top_n]

            # Plot heatmap
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(
                selected_matrix,
                annot=False,
                cmap="coolwarm",
                xticklabels=selected_queries,
                yticklabels=selected_queries,
                cbar_kws={'label': 'Similarity Score'}
            )
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig)

            # Insights Section
            st.write("### Insights Maker")
            st.write("Identify top similarities and potential keyword clusters.")

            # User input: select a query to analyze its closest matches
            query_to_analyze = st.selectbox(
                "Select a query to analyze:",
                queries,
                help="Choose a query to see its most similar keywords"
            )

            if query_to_analyze:
                col1, col2 = st.columns(2)

                with col1:
                    # Number of top matches to show
                    num_matches = st.slider(
                        "Number of top matches to show:",
                        min_value=3,
                        max_value=20,
                        value=5
                    )

                    top_matches = get_top_matches(
                        query_to_analyze,
                        queries,
                        matrix,
                        top_n=num_matches
                    )

                    st.write(f"#### Top Matches for '{query_to_analyze}':")
                    for match, score in top_matches:
                        st.write(f"- **{match}**: {score:.4f}")

                with col2:
                    # Statistical summary
                    query_index = queries[queries == query_to_analyze].index[0]
                    similarities = matrix[query_index]

                    st.write("#### Statistics:")
                    st.metric("Average Similarity", f"{similarities.mean():.4f}")
                    st.metric("Max Similarity", f"{similarities.max():.4f}")
                    st.metric("Min Similarity", f"{similarities.min():.4f}")

            # Find clusters
            st.write("### Query Clustering")
            st.write("Group similar queries based on a similarity threshold.")

            threshold = st.slider(
                "Set similarity threshold for clustering:",
                min_value=0.0,
                max_value=1.0,
                value=config.DEFAULT_THRESHOLD,
                step=0.05,
                help="Queries with similarity above this threshold will be grouped together"
            )

            # Generate clusters
            clusters = {}
            for i, query in enumerate(queries):
                cluster = [
                    queries[j]
                    for j in range(len(queries))
                    if matrix[i][j] >= threshold
                ]
                if len(cluster) > 1:
                    clusters[query] = cluster

            if clusters:
                st.info(f"Found {len(clusters)} clusters with threshold {threshold}")

                # Show cluster size distribution
                cluster_sizes = [len(c) for c in clusters.values()]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Cluster Size", f"{np.mean(cluster_sizes):.1f}")
                with col2:
                    st.metric("Largest Cluster", max(cluster_sizes))
                with col3:
                    st.metric("Smallest Cluster", min(cluster_sizes))

                # Display clusters
                st.write("#### Cluster Details:")
                for key, cluster in clusters.items():
                    with st.expander(f"📌 {key} ({len(cluster)} keywords)"):
                        st.write(", ".join(cluster))
            else:
                st.warning("No clusters found. Try lowering the threshold.")

        except pd.errors.EmptyDataError:
            st.error("The uploaded file is empty or invalid.")
        except pd.errors.ParserError:
            st.error("Error parsing the CSV file. Please ensure it's a valid CSV format.")
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            st.error(f"Error processing the file: {str(e)}")


if __name__ == "__main__":
    main()
