"""
Google Search Console Keyword Similarity Analyzer
A simple Streamlit application for analyzing keyword relationships using TF-IDF and cosine similarity.
"""
import logging
import pandas as pd
import streamlit as st

from utils import calculate_cosine_similarity, validate_dataframe, get_top_keyword_pairs
from config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@st.cache_data
def compute_similarity_cached(data: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Cached version of similarity computation.

    Args:
        data: DataFrame containing keyword data
        column_name: Name of the column with keywords

    Returns:
        DataFrame with similarity matrix
    """
    return calculate_cosine_similarity(
        data,
        column_name,
        min_df=config.MIN_DF,
        max_df=config.MAX_DF,
        ngram_range=config.NGRAM_RANGE
    )


def main():
    """Main application function."""
    st.set_page_config(
        page_title="Google Search Console Keyword Similarity",
        page_icon="🔍",
        layout=config.LAYOUT
    )
    st.title("🔍 Google Search Console Keyword Similarity")
    st.write("Upload your Google Search Console keyword data in CSV format to analyze keyword relationships.")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"],
        help="Upload a CSV file containing your keyword data from Google Search Console"
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

            # Column selection
            columns = list(data.columns)
            column_name = st.selectbox(
                "Select the column with keywords",
                columns,
                help="Choose the column that contains your keywords or search queries"
            )

            # Calculate cosine similarity
            if st.button("Calculate Similarity", type="primary"):
                # Validate data
                is_valid, error_message = validate_dataframe(data, column_name)
                if not is_valid:
                    st.error(error_message)
                    return

                # Show warnings for data quality issues
                null_count = data[column_name].isnull().sum()
                if null_count > 0:
                    st.warning(f"⚠ Removing {null_count} rows with null values")

                dup_count = data[column_name].duplicated().sum()
                if dup_count > 0:
                    st.warning(f"⚠ Removing {dup_count} duplicate keywords")

                with st.spinner("Computing similarity matrix..."):
                    try:
                        similarity_df = compute_similarity_cached(data, column_name)

                        logger.info(f"Computed similarity matrix: {similarity_df.shape}")

                        st.success("✓ Similarity matrix computed successfully!")

                        # Display results
                        st.write("### Cosine Similarity Matrix:")
                        st.dataframe(similarity_df, use_container_width=True)

                        # Download options
                        st.write("### Download Results")

                        col1, col2 = st.columns(2)

                        with col1:
                            # Option to download the similarity matrix
                            csv = similarity_df.to_csv(index=True)
                            st.download_button(
                                label="📥 Download Similarity Matrix",
                                data=csv,
                                file_name="cosine_similarity.csv",
                                mime="text/csv",
                                use_container_width=True
                            )

                        with col2:
                            # Export top keyword pairs
                            top_pairs_df = get_top_keyword_pairs(
                                similarity_df,
                                top_n=config.DEFAULT_TOP_PAIRS
                            )
                            st.download_button(
                                label="📥 Download Top Keyword Pairs",
                                data=top_pairs_df.to_csv(index=False),
                                file_name="top_keyword_pairs.csv",
                                mime="text/csv",
                                use_container_width=True
                            )

                        # Show top pairs preview
                        with st.expander("Preview Top Keyword Pairs"):
                            st.dataframe(
                                top_pairs_df.head(10),
                                use_container_width=True
                            )

                    except ValueError as e:
                        logger.error(f"Validation error: {str(e)}")
                        st.error(str(e))
                    except Exception as e:
                        logger.error(f"Error computing similarity: {str(e)}")
                        st.error(f"An error occurred during computation: {str(e)}")

        except pd.errors.EmptyDataError:
            st.error("The uploaded file is empty or invalid.")
        except pd.errors.ParserError:
            st.error("Error parsing the CSV file. Please ensure it's a valid CSV format.")
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            st.error(f"Error processing the file: {str(e)}")


if __name__ == "__main__":
    main()
