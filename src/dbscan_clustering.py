import os
import shutil
import json
import numpy as np
import argparse
from sklearn.cluster import DBSCAN

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


def cluster_embeddings(embeddings, filenames, eps=0.55, min_samples=5):
    """
    Clusters embeddings using DBSCAN and organizes the images into cluster folders.
    
    Args:
        embeddings (np.ndarray): The dataset of embeddings.
        filenames (list): List of filenames corresponding to the embeddings.
        eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
    """
    # Clear previous clusters if they exist
    if os.path.exists('./frames/clusters'):
        shutil.rmtree('./frames/clusters')

    # Cluster the embeddings using DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = dbscan.fit_predict(embeddings)

    # Create folders for each cluster and move images
    cluster_path = './frames/clusters'
    os.makedirs(cluster_path, exist_ok=True)
    cluster_mapping = {}

    for label, filename in zip(labels, filenames):
        if label == -1:
            continue  # -1 means outlier
        cluster_folder = os.path.join(cluster_path, f'student_{label}')
        os.makedirs(cluster_folder, exist_ok=True)
        
        # Replace .npy with .jpg and use the face crop folder for images
        image_file = filename.replace('.npy', '.jpg').replace('embeddings', 'face_crops')
        image_path = os.path.join('./frames/face_crops', image_file)
        
        if os.path.exists(image_path):
            shutil.copy(image_path, cluster_folder)
        
        cluster_mapping[filename] = label

    # Save the cluster mapping as a JSON file
    with open('./frames/cluster_mapping.json', 'w') as f:
        json.dump(cluster_mapping, f, indent=4, default=str)
    
    print(f"Clustering complete. Clusters saved in {cluster_path}")


def parse_arguments():
    """
    Parse command-line arguments for the script.

    Returns:
        Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Cluster embeddings using DBSCAN and organize images by clusters.")
    parser.add_argument('--embeddings_path', type=str, required=True, 
                        help="Path to the directory containing .npy embedding files.")
    parser.add_argument('--eps', type=float, default=0.55, 
                        help="DBSCAN eps parameter (default: 0.55).")
    parser.add_argument('--min_samples', type=int, default=5, 
                        help="DBSCAN min_samples parameter (default: 5).")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()
    
    embeddings_path = args.embeddings_path
    eps = args.eps
    min_samples = args.min_samples

    # Load embeddings from the provided path
    embeddings, filenames = load_embeddings(embeddings_path)
    
    # Cluster the embeddings and organize images into folders
    cluster_embeddings(embeddings, filenames, eps=eps, min_samples=min_samples)