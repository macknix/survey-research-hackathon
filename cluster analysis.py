import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    # Load the dataset
    file_path = 'Hotel_Reviews_1000.csv'
    if not os.path.exists(file_path):
        # Try looking in the same directory as the script
        file_path = os.path.join(os.path.dirname(__file__), 'Hotel_Reviews_1000.csv')
    
    if not os.path.exists(file_path):
        print(f"Error: Could not find {file_path}")
        return

    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)

    # Preprocessing
    print("Preprocessing data...")
    # Handle "No Negative" and "No Positive" which seem to be placeholders
    df['Negative_Review'] = df['Negative_Review'].replace('No Negative', '')
    df['Positive_Review'] = df['Positive_Review'].replace('No Positive', '')

    # Combine reviews for clustering
    df['Review_Text'] = df['Negative_Review'].fillna('') + " " + df['Positive_Review'].fillna('')

    # Vectorization
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(df['Review_Text'])

    # Clustering
    k = 5  # Number of clusters
    print(f"Performing K-Means clustering with k={k}...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X)


    # Analyze clusters
    print("\nAnalyzing clusters...")
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    
    # Calculate distances to cluster centers to find representative reviews
    distances = kmeans.transform(X)
    
    with open('cluster_topics.txt', 'w', encoding='utf-8') as f:
        for i in range(k):
            # Get cluster size
            cluster_size = len(df[df['Cluster'] == i])
            
            # Get top terms (increased to 20)
            top_terms = [terms[ind] for ind in order_centroids[i, :20]]
            terms_str = ", ".join(top_terms)
            
            # Get representative reviews (closest to centroid)
            # Ensure we use integer indexing for the numpy array 'distances'
            cluster_indices = df.index[df['Cluster'] == i].tolist()
            cluster_distances = distances[cluster_indices, i]
            sorted_local_indices = cluster_distances.argsort()[:3]
            representative_indices = [cluster_indices[idx] for idx in sorted_local_indices]
            representative_reviews = df.loc[representative_indices, 'Review_Text'].values
            
            # Prepare output
            header = f"Cluster {i} (Size: {cluster_size} reviews)"
            print(f"\n{header}")
            print(f"Top Terms: {terms_str}")
            
            f.write(f"{header}\n")
            f.write(f"Top Terms: {terms_str}\n")
            f.write("Representative Reviews:\n")
            
            for idx, review in enumerate(representative_reviews, 1):
                # Clean up newlines in review for cleaner output
                clean_review = str(review).replace('\n', ' ').strip()
                # Truncate if too long
                display_review = (clean_review[:300] + '...') if len(clean_review) > 300 else clean_review
                f.write(f"  {idx}. {display_review}\n")
            
            f.write("-" * 50 + "\n\n")
            
    print("Cluster topics and details saved to 'cluster_topics.txt'")

    # Visualization (Word Clouds)
    print("\nGenerating word clouds...")
    
    # Determine grid size
    cols = 2
    rows = (k + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for i in range(k):
        print(f"Generating word cloud for Cluster {i}...")
        cluster_text = " ".join(df[df['Cluster'] == i]['Review_Text'])
        
        if not cluster_text.strip():
            print(f"Cluster {i} has no text, skipping word cloud.")
            axes[i].text(0.5, 0.5, "No Data", ha='center', va='center')
            axes[i].axis('off')
            continue

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cluster_text)
        
        axes[i].imshow(wordcloud, interpolation='bilinear')
        axes[i].axis('off')
        axes[i].set_title(f'Cluster {i}')
    
    # Hide empty subplots
    for i in range(k, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    output_plot = 'clusters_wordclouds_comparison.png'
    plt.savefig(output_plot)
    plt.close()
    print(f"Saved {output_plot}")

    # Save results
    output_csv = 'clustered_reviews.csv'
    df[['Hotel_Name', 'Negative_Review', 'Positive_Review', 'Cluster']].to_csv(output_csv, index=False)
    print(f"Clustered data saved to '{output_csv}'")

if __name__ == "__main__":
    main()

