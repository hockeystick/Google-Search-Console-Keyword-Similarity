# proximity_service_full.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def calculate_cosine_similarity(data, column_name):
    # Compute TF-IDF vectors
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(data[column_name])
    
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Convert to DataFrame for easier handling
    similarity_df = pd.DataFrame(
        similarity_matrix, 
        index=data[column_name], 
        columns=data[column_name]
    )
    return similarity_df

def main():
    st.title("Keyword Proximity Analyzer and Insights Tool")
    st.write("Upload your keyword data, compute proximity relationships, and visualize insights.")

    # Step 1: Upload Raw Keyword Data
    uploaded_file = st.file_uploader("Upload a CSV File (Keyword Data)", type=["csv"])
    
    if uploaded_file:
        # Load the data
        raw_data = pd.read_csv(uploaded_file)
        st.write("### Data Preview:")
        st.dataframe(raw_data.head())

        # Step 2: Select the column containing keywords
        columns = list(raw_data.columns)
        selected_column = st.selectbox("Select the column containing keywords:", columns)
        
        # Step 3: Compute Cosine Similarity
        if st.button("Compute Proximity Matrix"):
            st.write("### Proximity Matrix")
            similarity_df = calculate_cosine_similarity(raw_data, selected_column)
            st.dataframe(similarity_df)

            # Step 4: Download Proximity Matrix
            csv = similarity_df.to_csv().encode('utf-8')
            st.download_button(
                label="Download Proximity Matrix as CSV",
                data=csv,
                file_name='proximity_matrix.csv',
                mime='text/csv'
            )

            # Step 5: Heatmap Visualization
            st.write("### Heatmap of Keyword Proximity")
            top_n = st.slider("Select the number of keywords to visualize:", 5, len(similarity_df), 20)
            heatmap_subset = similarity_df.iloc[:top_n, :top_n]

            # Plot heatmap
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

            # Step 6: Insights Maker
            st.write("### Insights Maker")
            query_to_analyze = st.selectbox("Select a keyword to analyze:", similarity_df.index)
            if query_to_analyze:
                similarities = similarity_df[query_to_analyze]
                top_matches = similarities.sort_values(ascending=False)[1:6]
                st.write(f"Top Matches for '{query_to_analyze}':")
                for match, score in top_matches.items():
                    st.write(f"- **{match}**: {score:.2f}")

            # Step 7: Clustering
            st.write("### Keyword Clustering")
            threshold = st.slider("Set similarity threshold for clustering:", 0.0, 1.0, 0.5)
            clusters = {}
            for keyword in similarity_df.index:
                cluster = similarity_df.loc[keyword][similarity_df.loc[keyword] >= threshold].index.tolist()
                if len(cluster) > 1:
                    clusters[keyword] = cluster
            
            st.write("Clusters formed:")
            for key, cluster in clusters.items():
                st.write(f"- **{key}**: {', '.join(cluster)}")

if __name__ == "__main__":
    main()
