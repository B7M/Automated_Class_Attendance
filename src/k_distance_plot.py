from imports import np, os, NearestNeighbors, plt, argparse,KMeans


def load_embeddings(embeddings_path):
    """
    Loads all .npy embedding files from the specified directory.
    
    Args:
        embeddings_path (str): Path to the directory containing .npy embedding files.
    
    Returns:
        np.ndarray: Numpy array containing all embeddings.
        list: List of filenames corresponding to the embeddings.
    """
    embeddings = []
    filenames = []

    for file in os.listdir(embeddings_path):
        if file.endswith('.npy'):
            filepath = os.path.join(embeddings_path, file)
            embedding = np.load(filepath)
            embeddings.append(embedding)
            filenames.append(file)

    embeddings = np.array(embeddings)
    return embeddings, filenames


def plot_k_distance(X, k=4):
    """
    Plots the K-distance graph to help choose the min_samples value for DBSCAN.
    
    Args:
        X (np.ndarray): The dataset of embeddings.
        k (int): The k-th nearest neighbor distance to consider for the plot.
    """
    # Step 1: Fit Nearest Neighbors to the data
    nearest_neighbors = NearestNeighbors(n_neighbors=k)
    nearest_neighbors.fit(X)
    
    # Step 2: Compute distances to the k-th nearest neighbor
    distances, indices = nearest_neighbors.kneighbors(X)
    k_distances = distances[:, k-1]  # Get the distance to the k-th nearest neighbor
    
    # Step 3: Sort the distances
    k_distances = np.sort(k_distances)
    
    # Step 4: Plot the k-distances
    plt.figure(figsize=(10, 6))
    plt.plot(k_distances, label=f'{k}-Distance')
    plt.xlabel('Points sorted by distance to {}-th nearest neighbor'.format(k))
    plt.ylabel('Distance to {}-th nearest neighbor'.format(k))
    plt.title(f'K-Distance Plot for k={k}')
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_elbow_method(X, max_k=10):
    """
    Plots the elbow method to determine the optimal number of clusters for KMeans.
    
    Args:
        X (np.ndarray): The dataset of embeddings.
        max_k (int): The maximum number of clusters to consider for KMeans.
    """
    inertias = []
    K_range = range(1, max_k + 1)  # Test K from 1 to max_k

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    # Step 3: Plot the elbow plot
    plt.figure(figsize=(8, 6))
    plt.plot(K_range, inertias, marker='o', linestyle='--')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia (Sum of Squared Distances)')
    plt.title('Elbow Plot to Determine Optimal K')
    plt.xticks(K_range)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def parse_arguments():
    """
    Parse command-line arguments for the script.

    Returns:
        Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Analyze embeddings, plot K-distance and elbow plots.")
    parser.add_argument('--embeddings_path', type=str, required=True, 
                        help="Path to the directory containing .npy embedding files.")
    parser.add_argument('--k', type=int, default=4, 
                        help="Value of k for K-distance plot.")
    parser.add_argument('--max_k', type=int, default=10, 
                        help="Maximum number of clusters for the elbow plot.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Parse the command-line arguments
    args = parse_arguments()
    
    embeddings_path = args.embeddings_path
    k = args.k
    max_k = args.max_k

    # Load embeddings from the provided path
    embeddings, filenames = load_embeddings(embeddings_path)
    
    # Plot the K-Distance graph for DBSCAN
    print(f"Plotting K-Distance plot for k={k}...")
    plot_k_distance(embeddings, k)
    
    # Plot the elbow plot for KMeans
    print(f"Plotting Elbow plot to determine the optimal number of clusters (up to {max_k})...")
    plot_elbow_method(embeddings, max_k)