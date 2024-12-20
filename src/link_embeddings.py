import os
import json
import numpy as np
import argparse


def load_cluster_to_student_mapping(mapping_path):
    """
    Loads the cluster-to-student mapping from a JSON file.
    
    Args:
        mapping_path (str): Path to the JSON file containing the cluster-to-student mapping.
    
    Returns:
        dict: A dictionary where keys are cluster IDs and values are student identifiers.
    """
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")
    
    with open(mapping_path, 'r') as f:
        cluster_to_student = json.load(f)
    
    print(f"Loaded cluster-to-student mapping from {mapping_path}")
    return cluster_to_student


def load_existing_embeddings(embeddings_folder):
    """
    Loads existing embeddings and corresponding labels from the given folder.
    
    Args:
        embeddings_folder (str): Path to the folder containing .npy embedding files.
    
    Returns:
        np.ndarray: Numpy array containing all embeddings.
        list: List of cluster labels corresponding to each embedding.
    """
    if not os.path.exists(embeddings_folder):
        raise FileNotFoundError(f"Embeddings folder not found: {embeddings_folder}")
    
    existing_embeddings = []
    existing_labels = []

    for filename in os.listdir(embeddings_folder):
        if filename.endswith('.npy'):
            embedding_path = os.path.join(embeddings_folder, filename)
            embedding = np.load(embedding_path)
            
            try:
                cluster_id = int(filename.split('_')[1])  # Assuming the file name contains the cluster id
            except ValueError:
                raise ValueError(f"Filename {filename} does not follow the expected format. Expected format: 'frame_<cluster_id>_face_*.npy'")
            
            existing_embeddings.append(embedding)
            existing_labels.append(cluster_id)
    
    existing_embeddings = np.array(existing_embeddings)
    print(f"Loaded {len(existing_embeddings)} embeddings from {embeddings_folder}")
    return existing_embeddings, existing_labels


def parse_arguments():
    """
    Parse command-line arguments for the script.

    Returns:
        Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Load cluster-to-student mapping and existing embeddings.")
    parser.add_argument('--mapping_path', type=str, required=True, 
                        help="Path to the JSON file containing the cluster-to-student mapping.")
    parser.add_argument('--embeddings_folder', type=str, required=True, 
                        help="Path to the folder containing .npy embedding files.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()
    
    # Load the cluster-to-student mapping
    cluster_to_student = load_cluster_to_student_mapping(args.mapping_path)
    
    # Load existing embeddings and labels
    existing_embeddings, existing_labels = load_existing_embeddings(args.embeddings_folder)