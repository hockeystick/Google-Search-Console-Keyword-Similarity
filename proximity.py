# app.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Function to calculate cosine similarity
def calculate_cosine_similarity(data, column_name):
    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(data[column_name])
    
    # Cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Convert to DataFrame
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=data[column_name],
        columns=data[column_name]
    )
    return similarity_df

# Streamlit app
def main():
    st.title("Google Search Console Keyword Similarity")
    st.write("Upload your Google Search Console keyword data in CSV format.")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    
    if uploaded_file:
        # Read the uploaded file
        try:
            data = pd.read_csv(uploaded_file)
            st.write("Data Preview:")
            st.write(data.head())
            
            # Column selection
            columns = list(data.columns)
            column_name = st.selectbox("Select the column with keywords", columns)
            
            # Calculate cosine similarity
            if st.button("Calculate Similarity"):
                similarity_df = calculate_cosine_similarity(data, column_name)
                st.write("Cosine Similarity Matrix:")
                st.write(similarity_df)
                
                # Option to download the similarity matrix
                csv = similarity_df.to_csv(index=True)
                st.download_button(
                    label="Download Similarity Matrix as CSV",
                    data=csv,
                    file_name="cosine_similarity.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error processing the file: {e}")

if __name__ == "__main__":
    main()
