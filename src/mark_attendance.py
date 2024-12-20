import os
import json
import shutil
import cv2
import dlib
import torch
import numpy as np
import argparse
from sklearn.neighbors import NearestNeighbors
from facenet_pytorch import InceptionResnetV1


def load_cluster_to_student_map(map_path):
    """Load the cluster-to-student mapping from a JSON file."""
    with open(map_path, "r") as f:
        cluster_to_student = json.load(f)
    print(f"Loaded cluster-to-student mapping from {map_path}")
    return cluster_to_student


def extract_faces_from_image(image_path, face_output_folder="new_session"):
    """Extract faces from an image and save them as separate files."""
    face_paths = []
    if os.path.exists(face_output_folder):
        shutil.rmtree(face_output_folder)
    
    cnn_face_detector = dlib.cnn_face_detection_model_v1('./models/mmod_human_face_detector.dat')
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detected_faces = cnn_face_detector(gray, 1)
    
    for i, face in enumerate(detected_faces):
        x1, y1, x2, y2 = face.rect.left(), face.rect.top(), face.rect.right(), face.rect.bottom()
        margin = 20  # Add margin to avoid cutting the face
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(image.shape[1], x2 + margin)
        y2 = min(image.shape[0], y2 + margin)

        face_crop = image[y1:y2, x1:x2]
        os.makedirs(face_output_folder, exist_ok=True)
        face_path = os.path.join(face_output_folder, f"face_{i}.jpg")
        cv2.imwrite(face_path, face_crop)
        face_paths.append(face_path)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv2.imwrite(os.path.join(face_output_folder, "face_detected.png"), image)
    print(f"Extracted {len(face_paths)} faces from {image_path}")
    return face_paths


def extract_embeddings_from_faces(face_paths):
    """Extract embeddings for each face image."""
    facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
    embeddings = []
    for face_path in face_paths:
        try:
            face_crop = cv2.imread(face_path)
            face_crop_resized = cv2.resize(face_crop, (160, 160))
            face_crop_tensor = torch.tensor(face_crop_resized).permute(2, 0, 1).float().div(255).unsqueeze(0)
            
            if torch.cuda.is_available():
                facenet_model = facenet_model.cuda()
                face_crop_tensor = face_crop_tensor.cuda()
            
            with torch.no_grad():
                embedding = facenet_model(face_crop_tensor).cpu().numpy().flatten()
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error extracting embedding for {face_path}: {e}")
    print(f"Extracted embeddings for {len(embeddings)} faces")
    return np.array(embeddings)


def load_existing_embeddings_and_labels(embeddings_folder, cluster_mapping_path):
    """Load the previously saved embeddings and cluster labels."""
    embeddings = []
    labels = []
    filenames = []
    with open(cluster_mapping_path, 'r') as f:
        cluster_mapping = json.load(f)
        
    for filename, cluster_label in cluster_mapping.items():
        embedding_path = os.path.join(embeddings_folder, filename)
        if os.path.exists(embedding_path):
            embedding = np.load(embedding_path)
            embeddings.append(embedding)
            labels.append(cluster_label)
            filenames.append(filename)
    
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    print(f"Loaded {len(embeddings)} existing embeddings")
    return embeddings, labels, filenames


def match_faces_to_clusters(new_embeddings, existing_embeddings, cluster_labels, cluster_to_student):
    """Match new embeddings to the existing clusters using KNN."""
    knn = NearestNeighbors(n_neighbors=4, metric='euclidean')
    knn.fit(existing_embeddings)
    
    student_names = []
    for i, new_embedding in enumerate(new_embeddings):
        distances, indices = knn.kneighbors(new_embedding.reshape(1, -1))
        closest_index = indices[0][0]
        cluster_id = cluster_labels[closest_index]
        student_name = cluster_to_student.get(str(cluster_id), "Unknown")
        student_names.append(student_name)
    
    print(f"Matched faces to the following students: {student_names}")
    return student_names


def mark_attendance(present_students):
    """Mark students as present."""
    present_students_set = set(present_students)
    print(f"Students present: {present_students_set}")
    return list(present_students_set)


def parse_arguments():
    """Parse command-line arguments for the script."""
    parser = argparse.ArgumentParser(description="Mark student attendance using face recognition.")
    parser.add_argument('--new_image_path', type=str, required=True, 
                        help="Path to the new image for attendance marking.")
    parser.add_argument('--mapping_path', type=str, default="./map/cluster_to_student.json", 
                        help="Path to the cluster-to-student mapping JSON file.")
    parser.add_argument('--embeddings_folder', type=str, default="./frames/embeddings", 
                        help="Path to the folder containing .npy embedding files.")
    parser.add_argument('--cluster_mapping_path', type=str, default="./frames/cluster_mapping.json", 
                        help="Path to the JSON file containing cluster-to-student mapping.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    
    # 1. Load the cluster-to-student mapping
    cluster_to_student = load_cluster_to_student_map(args.mapping_path)
    
    # 2. Extract faces from the new picture
    face_paths = extract_faces_from_image(args.new_image_path, face_output_folder="new_session")
    
    # 3. Extract embeddings for the new faces
    new_embeddings = extract_embeddings_from_faces(face_paths)
    
    # 4. Load existing embeddings and cluster labels
    existing_embeddings, cluster_labels, filenames = load_existing_embeddings_and_labels(
        embeddings_folder=args.embeddings_folder, 
        cluster_mapping_path=args.cluster_mapping_path
    )
    
    # 5. Match the new embeddings with the existing cluster embeddings
    present_students = match_faces_to_clusters(new_embeddings, existing_embeddings, cluster_labels, cluster_to_student)
    
    # 6. Mark the attendance of students
    mark_attendance(present_students)