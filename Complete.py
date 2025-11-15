"""
News Keyword Proximity & Content Recommendations
A Streamlit application for analyzing keyword relationships and generating AI-powered content recommendations.
"""
import os
import json
import logging
from typing import Dict, List, Optional
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

from utils import calculate_cosine_similarity, validate_dataframe, get_top_keyword_pairs
from config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


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


def get_ai_recommendations(
    client: OpenAI,
    clusters: Dict[str, List[str]],
    similarities: pd.DataFrame
) -> Optional[Dict]:
    """
    Get AI recommendations based on clusters and similarities.

    Args:
        client: OpenAI client instance
        clusters: Dictionary of keyword clusters
        similarities: Similarity matrix DataFrame

    Returns:
        Dictionary with AI recommendations or None if error
    """
    try:
        # Prepare data for OpenAI
        cluster_data = []
        for key, cluster in clusters.items():
            # Get similarity scores for cluster members
            cluster_similarities = {
                k: float(similarities.loc[key, k])
                for k in cluster if k in similarities.columns
            }

            cluster_data.append({
                "main_keyword": key,
                "related_keywords": cluster,
                "similarity_scores": cluster_similarities
            })

        prompt = f"""
        As a journalism and SEO expert, analyze these keyword clusters and their relationships:

        Clusters and Similarities:
        {json.dumps(cluster_data, indent=2)}

        Provide strategic recommendations for a news organization:
        1. Story angles and content opportunities for each cluster
        2. How to cover these related topics effectively
        3. SEO optimization suggestions for news articles
        4. Follow-up story ideas based on keyword relationships
        5. Internal linking strategy between related articles

        Format response as JSON:
        {{
            "clusters": [
                {{
                    "main_topic": "cluster theme",
                    "content_angles": ["angle 1", "angle 2"],
                    "coverage_tips": ["tip 1", "tip 2"],
                    "seo_strategy": ["strategy 1", "strategy 2"],
                    "follow_up_ideas": ["idea 1", "idea 2"]
                }}
            ],
            "overall_strategy": "general strategy for all content",
            "linking_recommendations": "how to link between clusters"
        }}
        """

        logger.info(f"Requesting AI recommendations for {len(clusters)} clusters")

        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert journalism SEO consultant specializing in news content optimization."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=config.OPENAI_TEMPERATURE,
            max_tokens=config.OPENAI_MAX_TOKENS
        )

        response_text = response.choices[0].message.content

        # Try to extract JSON if it's wrapped in markdown
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        result = json.loads(response_text)
        logger.info("Successfully received AI recommendations")
        return result

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {e}")
        st.error("Received invalid response from AI. Please try again.")
        with st.expander("Raw Response"):
            st.code(response_text)
        return None
    except Exception as e:
        logger.error(f"Error getting AI recommendations: {str(e)}")
        st.error(f"Error getting AI recommendations: {str(e)}")
        return None


def validate_api_key(api_key: str) -> bool:
    """
    Validate OpenAI API key format.

    Args:
        api_key: API key to validate

    Returns:
        True if valid format, False otherwise
    """
    if not api_key:
        return False
    if not api_key.startswith('sk-'):
        st.error("Invalid API key format. OpenAI keys start with 'sk-'")
        return False
    return True


def main():
    """Main application function."""
    st.set_page_config(
        page_title=config.PAGE_TITLE,
        page_icon=config.PAGE_ICON,
        layout=config.LAYOUT
    )
    st.title("📰 News Keyword Proximity & Content Recommendations")
    st.write("Upload your keyword data, analyze relationships, and get AI-powered content recommendations.")

    # Sidebar for OpenAI API key
    with st.sidebar:
        st.header("Settings")

        # Try to load from environment first
        default_api_key = os.getenv("OPENAI_API_KEY", "")

        api_key = st.text_input(
            "OpenAI API Key",
            value=default_api_key,
            type="password",
            help="Enter your OpenAI API key or set OPENAI_API_KEY environment variable"
        )

        client = None
        if api_key:
            if validate_api_key(api_key):
                try:
                    client = OpenAI(api_key=api_key)
                    st.success("✓ API Key loaded!")
                except Exception as e:
                    logger.error(f"Error initializing OpenAI client: {str(e)}")
                    st.error(f"Error with API key: {str(e)}")
        else:
            st.warning("⚠ Please enter your OpenAI API key")
            st.info("Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV File (Keyword Data)",
        type=["csv"],
        help="Upload a CSV file containing keywords for analysis"
    )

    if uploaded_file:
        try:
            # Load and preview data
            raw_data = pd.read_csv(uploaded_file)

            if raw_data.empty:
                st.error("The uploaded file is empty.")
                return

            st.write("### Data Preview:")
            st.dataframe(raw_data.head())

            # Display basic statistics
            st.caption(f"Total rows: {len(raw_data)}")

            # Select keyword column
            columns = list(raw_data.columns)
            selected_column = st.selectbox(
                "Select the column containing keywords:",
                columns,
                help="Choose the column that contains your keywords or search queries"
            )

            if st.button("Analyze Keywords", type="primary"):
                # Validate data
                is_valid, error_message = validate_dataframe(raw_data, selected_column)
                if not is_valid:
                    st.error(error_message)
                    return

                # Show warnings for data quality issues
                null_count = raw_data[selected_column].isnull().sum()
                if null_count > 0:
                    st.warning(f"⚠ Removing {null_count} rows with null values")

                dup_count = raw_data[selected_column].duplicated().sum()
                if dup_count > 0:
                    st.warning(f"⚠ Removing {dup_count} duplicate keywords")

                with st.spinner("Computing proximity matrix..."):
                    try:
                        # Compute similarity matrix
                        similarity_df = compute_similarity_cached(raw_data, selected_column)

                        logger.info(f"Computed similarity matrix: {similarity_df.shape}")

                        # Create tabs for different analyses
                        tab1, tab2, tab3 = st.tabs([
                            "📊 Proximity Analysis",
                            "🔗 Clusters",
                            "🤖 AI Recommendations"
                        ])

                        with tab1:
                            st.subheader("Keyword Proximity Matrix")
                            st.dataframe(similarity_df, use_container_width=True)

                            # Download buttons
                            col1, col2 = st.columns(2)

                            with col1:
                                csv = similarity_df.to_csv().encode('utf-8')
                                st.download_button(
                                    "📥 Download Proximity Matrix",
                                    csv,
                                    "proximity_matrix.csv",
                                    "text/csv",
                                    use_container_width=True
                                )

                            with col2:
                                # Export top keyword pairs
                                top_pairs_df = get_top_keyword_pairs(
                                    similarity_df,
                                    top_n=config.DEFAULT_TOP_PAIRS
                                )
                                st.download_button(
                                    "📥 Download Top Keyword Pairs",
                                    top_pairs_df.to_csv(index=False).encode('utf-8'),
                                    "top_keyword_pairs.csv",
                                    "text/csv",
                                    use_container_width=True
                                )

                            # Heatmap visualization
                            st.subheader("Proximity Heatmap")

                            max_keywords = min(len(similarity_df), config.MAX_HEATMAP_SIZE)
                            top_n = st.slider(
                                "Number of keywords to visualize:",
                                min_value=config.MIN_HEATMAP_SIZE,
                                max_value=max_keywords,
                                value=min(config.DEFAULT_HEATMAP_SIZE, max_keywords)
                            )

                            heatmap_subset = similarity_df.iloc[:top_n, :top_n]

                            fig, ax = plt.subplots(figsize=(12, 10))
                            sns.heatmap(
                                heatmap_subset,
                                annot=False,
                                cmap="coolwarm",
                                xticklabels=heatmap_subset.columns,
                                yticklabels=heatmap_subset.index,
                                cbar_kws={'label': 'Similarity Score'}
                            )
                            plt.xticks(rotation=45, ha="right")
                            plt.yticks(rotation=0)
                            plt.tight_layout()
                            st.pyplot(fig)

                        with tab2:
                            st.subheader("Keyword Clusters")

                            threshold = st.slider(
                                "Similarity threshold for clustering:",
                                min_value=0.0,
                                max_value=1.0,
                                value=config.DEFAULT_THRESHOLD,
                                step=0.05,
                                help="Keywords with similarity above this threshold will be clustered together"
                            )

                            # Form clusters
                            clusters = {}
                            for keyword in similarity_df.index:
                                cluster = similarity_df.loc[keyword][
                                    similarity_df.loc[keyword] >= threshold
                                ].index.tolist()
                                if len(cluster) > 1:
                                    clusters[keyword] = cluster

                            if clusters:
                                st.info(f"Found {len(clusters)} clusters with threshold {threshold}")

                                # Display clusters
                                for key, cluster in clusters.items():
                                    with st.expander(f"📌 Cluster: {key} ({len(cluster)} keywords)"):
                                        st.write(", ".join(cluster))
                            else:
                                st.warning("No significant clusters found. Try lowering the threshold.")

                        with tab3:
                            st.subheader("AI-Powered Content Recommendations")

                            if not client:
                                st.warning("⚠ Please enter your OpenAI API key in the sidebar to get recommendations")
                                st.info("Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)")
                            elif not clusters:
                                st.warning("⚠ No significant clusters found. Try adjusting the threshold in the Clusters tab.")
                            else:
                                if st.button("Generate AI Recommendations", type="primary"):
                                    with st.spinner("Generating AI recommendations..."):
                                        recommendations = get_ai_recommendations(
                                            client,
                                            clusters,
                                            similarity_df
                                        )

                                        if recommendations:
                                            # Display cluster-specific recommendations
                                            if 'clusters' in recommendations:
                                                for cluster in recommendations['clusters']:
                                                    with st.expander(
                                                        f"📌 {cluster.get('main_topic', 'Topic')}",
                                                        expanded=True
                                                    ):
                                                        if 'content_angles' in cluster:
                                                            st.markdown("**Content Angles:**")
                                                            for angle in cluster['content_angles']:
                                                                st.write(f"• {angle}")

                                                        if 'coverage_tips' in cluster:
                                                            st.markdown("**Coverage Tips:**")
                                                            for tip in cluster['coverage_tips']:
                                                                st.write(f"• {tip}")

                                                        if 'seo_strategy' in cluster:
                                                            st.markdown("**SEO Strategy:**")
                                                            for strategy in cluster['seo_strategy']:
                                                                st.write(f"• {strategy}")

                                                        if 'follow_up_ideas' in cluster:
                                                            st.markdown("**Follow-up Ideas:**")
                                                            for idea in cluster['follow_up_ideas']:
                                                                st.write(f"• {idea}")

                                            # Overall strategy
                                            if 'overall_strategy' in recommendations:
                                                st.markdown("### Overall Content Strategy")
                                                st.write(recommendations['overall_strategy'])

                                            if 'linking_recommendations' in recommendations:
                                                st.markdown("### Internal Linking Strategy")
                                                st.write(recommendations['linking_recommendations'])

                    except ValueError as e:
                        logger.error(f"Validation error: {str(e)}")
                        st.error(str(e))
                    except Exception as e:
                        logger.error(f"Error during analysis: {str(e)}")
                        st.error(f"An error occurred during analysis: {str(e)}")

        except pd.errors.EmptyDataError:
            st.error("The uploaded file is empty or invalid.")
        except pd.errors.ParserError:
            st.error("Error parsing the CSV file. Please ensure it's a valid CSV format.")
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            st.error(f"Error processing the file: {str(e)}")


if __name__ == "__main__":
    main()
