# proximity_visualizer.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Streamlit app
def main():
    st.title("Proximity Visualizer and Insights Maker")
    st.write("Upload your cosine similarity data and visualize relationships between keywords.")

    # File Upload
    uploaded_file = st.file_uploader("Upload CSV File (Cosine Similarity Matrix)", type=["csv"])
    
    if uploaded_file:
        # Read the uploaded file
        data = pd.read_csv(uploaded_file)
        st.write("### Data Preview:")
        st.dataframe(data.head())
        
        # Extract query names and matrix
        queries = data['Top queries']
        matrix = data.iloc[:, 1:].values  # Exclude the first column (query names)

        # Generate Heatmap
        st.write("### Heatmap of Similarities")
        st.write("A heatmap showing the proximity between the top queries.")
        
        # Allow the user to select the number of queries to visualize
        top_n = st.slider("Select the number of queries to visualize:", 5, len(queries), 20)
        selected_queries = queries[:top_n]
        selected_matrix = matrix[:top_n, :top_n]
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            selected_matrix, 
            annot=False, 
            cmap="coolwarm", 
            xticklabels=selected_queries, 
            yticklabels=selected_queries
        )
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)

        # Insights Section
        st.write("### Insights Maker")
        st.write("Identify top similarities and potential keyword clusters.")

        # User input: select a query to analyze its closest matches
        query_to_analyze = st.selectbox("Select a query to analyze:", queries)
        if query_to_analyze:
            query_index = queries[queries == query_to_analyze].index[0]
            similarities = matrix[query_index]
            
            # Extract top matches
            top_matches = sorted(
                zip(queries, similarities), 
                key=lambda x: x[1], 
                reverse=True
            )[1:6]  # Exclude the query itself
            
            st.write(f"### Top Matches for '{query_to_analyze}':")
            for match, score in top_matches:
                st.write(f"- **{match}**: {score:.2f}")
            
            # Find clusters
            st.write("### Query Clustering")
            threshold = st.slider("Set similarity threshold for clustering:", 0.0, 1.0, 0.5)
            clusters = {}
            for i, query in enumerate(queries):
                cluster = [queries[j] for j in range(len(queries)) if matrix[i][j] >= threshold]
                clusters[query] = cluster

            st.write(f"Clusters formed with a threshold of {threshold}:")
            for key, cluster in clusters.items():
                if len(cluster) > 1:
                    st.write(f"- **{key}**: {', '.join(cluster)}")

# Run the Streamlit app
if __name__ == "__main__":
    main()
    