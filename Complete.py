import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from openai import OpenAI
import json

def calculate_cosine_similarity(data, column_name):
    """Compute TF-IDF vectors and cosine similarity"""
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(data[column_name])
    
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    similarity_df = pd.DataFrame(
        similarity_matrix, 
        index=data[column_name], 
        columns=data[column_name]
    )
    return similarity_df

def get_ai_recommendations(client, clusters, similarities):
    """Get AI recommendations based on clusters and similarities"""
    try:
        # Prepare data for OpenAI
        cluster_data = []
        for key, cluster in clusters.items():
            # Get similarity scores for cluster members
            cluster_similarities = {k: float(similarities.loc[key, k]) for k in cluster}
            
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

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert journalism SEO consultant specializing in news content optimization."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )

        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Error getting AI recommendations: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="News Keyword Analyzer", layout="wide")
    st.title("📰 News Keyword Proximity & Content Recommendations")
    st.write("Upload your keyword data, analyze relationships, and get AI-powered content recommendations.")

    # Sidebar for OpenAI API key
    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input("OpenAI API Key", type="password")
        
        if api_key:
            try:
                client = OpenAI(api_key=api_key)
                st.success("API Key loaded!")
            except Exception as e:
                st.error(f"Error with API key: {str(e)}")
                client = None
        else:
            client = None
            st.warning("Please enter your OpenAI API key")

    # File upload
    uploaded_file = st.file_uploader("Upload CSV File (Keyword Data)", type=["csv"])
    
    if uploaded_file:
        # Load and preview data
        raw_data = pd.read_csv(uploaded_file)
        st.write("### Data Preview:")
        st.dataframe(raw_data.head())

        # Select keyword column
        columns = list(raw_data.columns)
        selected_column = st.selectbox("Select the column containing keywords:", columns)
        
        if st.button("Analyze Keywords"):
            with st.spinner("Computing proximity matrix..."):
                # Compute similarity matrix
                similarity_df = calculate_cosine_similarity(raw_data, selected_column)
                
                # Create tabs for different analyses
                tab1, tab2, tab3 = st.tabs(["Proximity Analysis", "Clusters", "AI Recommendations"])
                
                with tab1:
                    st.subheader("Keyword Proximity Matrix")
                    st.dataframe(similarity_df)
                    
                    # Download button
                    csv = similarity_df.to_csv().encode('utf-8')
                    st.download_button(
                        "📥 Download Proximity Matrix",
                        csv,
                        "proximity_matrix.csv",
                        "text/csv"
                    )
                    
                    # Heatmap visualization
                    st.subheader("Proximity Heatmap")
                    top_n = st.slider("Number of keywords to visualize:", 5, len(similarity_df), 20)
                    heatmap_subset = similarity_df.iloc[:top_n, :top_n]
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(
                        heatmap_subset,
                        annot=False,
                        cmap="coolwarm",
                        xticklabels=heatmap_subset.columns,
                        yticklabels=heatmap_subset.index
                    )
                    plt.xticks(rotation=45, ha="right")
                    st.pyplot(fig)
                
                with tab2:
                    st.subheader("Keyword Clusters")
                    threshold = st.slider("Similarity threshold for clustering:", 0.0, 1.0, 0.5)
                    
                    # Form clusters
                    clusters = {}
                    for keyword in similarity_df.index:
                        cluster = similarity_df.loc[keyword][similarity_df.loc[keyword] >= threshold].index.tolist()
                        if len(cluster) > 1:
                            clusters[keyword] = cluster
                    
                    # Display clusters
                    for key, cluster in clusters.items():
                        with st.expander(f"Cluster: {key}"):
                            st.write(", ".join(cluster))
                
                with tab3:
                    st.subheader("AI-Powered Content Recommendations")
                    
                    if not client:
                        st.warning("Please enter your OpenAI API key to get recommendations")
                    elif not clusters:
                        st.warning("No significant clusters found. Try adjusting the threshold.")
                    else:
                        with st.spinner("Generating AI recommendations..."):
                            recommendations = get_ai_recommendations(client, clusters, similarity_df)
                            
                            if recommendations:
                                # Display cluster-specific recommendations
                                for cluster in recommendations['clusters']:
                                    with st.expander(f"📌 {cluster['main_topic']}", expanded=True):
                                        st.markdown("**Content Angles:**")
                                        for angle in cluster['content_angles']:
                                            st.write(f"• {angle}")
                                        
                                        st.markdown("**Coverage Tips:**")
                                        for tip in cluster['coverage_tips']:
                                            st.write(f"• {tip}")
                                        
                                        st.markdown("**SEO Strategy:**")
                                        for strategy in cluster['seo_strategy']:
                                            st.write(f"• {strategy}")
                                        
                                        st.markdown("**Follow-up Ideas:**")
                                        for idea in cluster['follow_up_ideas']:
                                            st.write(f"• {idea}")
                                
                                # Overall strategy
                                st.markdown("### Overall Content Strategy")
                                st.write(recommendations['overall_strategy'])
                                
                                st.markdown("### Internal Linking Strategy")
                                st.write(recommendations['linking_recommendations'])

if __name__ == "__main__":
    main()
