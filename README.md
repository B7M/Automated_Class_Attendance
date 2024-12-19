Face Recognition Attendance System

Table of Contents
	•	Introduction
	•	Features
	•	Project Structure
	•	Installation
	•	Usage
	•	Configuration
	•	Dependencies
	•	File Descriptions
	•	Potential Improvements
	•	Contributing
	•	License

Introduction

The Face Recognition Attendance System is a Python-based application that uses facial recognition to automatically detect, recognize, and mark the attendance of students. The system extracts faces from an input image, generates embeddings, and matches them against existing embeddings stored in a local database. It assigns student names to detected faces and records attendance. This system can be used in classrooms, meetings, and other events where attendance tracking is required.

Features
	•	📷 Face Detection: Detects multiple faces in an image using Dlib’s CNN face detector.
	•	🧑‍🤝‍🧑 Face Recognition: Recognizes students by matching detected faces with a pre-saved set of embeddings using K-Nearest Neighbors (KNN).
	•	💾 Data Persistence: Stores existing embeddings, cluster labels, and face-to-student mappings.
	•	📑 Attendance Marking: Automatically identifies present students and prints their names.
	•	🔄 Easy-to-Update Database: Add or update the face database by adding new images and embedding files.
	•	📂 File Organization: The project follows a clean and modular file structure for better maintainability and readability.

Project Structure

├── map/
│   └── cluster_to_student.json   # Mapping of cluster labels to student names
├── models/
│   └── mmod_human_face_detector.dat  # Dlib CNN face detection model
├── frames/
│   ├── embeddings/                # Folder containing embeddings of saved faces
│   └── cluster_mapping.json      # Cluster mapping of filenames to cluster labels
├── new_faces/
│   └── face_x.jpg                # New face images extracted from an input photo
├── app.py                        # Main script to run the app
└── README.md                     # This readme file

Installation
	1.	Clone the Repository

git clone https://github.com/your-username/face-recognition-attendance.git
cd face-recognition-attendance


	2.	Set Up a Python Virtual Environment (Optional)

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`


	3.	Install Dependencies

pip install -r requirements.txt


	4.	Download Pre-trained Models
	•	Place the mmod_human_face_detector.dat model file in the models/ directory. You can download it from Dlib’s model repository.

Usage

Follow these steps to run the face recognition and attendance marking system.
	1.	Update Cluster-to-Student Mapping
	•	Update the map/cluster_to_student.json file with the mapping of cluster labels to student names.
	2.	Run the Application

python app.py


	3.	How It Works
	•	The app will:
	1.	Load the cluster_to_student.json file.
	2.	Detect faces in the image 00.png using Dlib’s CNN detector.
	3.	Extract and save individual face images in the new_faces/ folder.
	4.	Extract embeddings for these faces and compare them to pre-existing embeddings.
	5.	Use K-Nearest Neighbors (KNN) to identify the closest matching face in the database.
	6.	Print a list of students present.
	4.	Results
	•	The present students are printed to the console.
	•	Detected faces are saved in the new_faces/ directory.

Configuration

File Paths
	•	Input Image: The default input image is 00.png, but you can update the file path in app.py:

new_image_path = "path/to/your/image.jpg"


	•	Cluster-to-Student Mapping: The file map/cluster_to_student.json should be formatted as:

{
    "1": "John Doe",
    "2": "Jane Smith",
    "3": "Alice Johnson"
}


	•	Embeddings Folder: This folder contains the pre-saved embeddings for students. Each embedding is a NumPy array stored in .npy format. You can add new embeddings here.

KNN Configuration
	•	Nearest Neighbors: The app uses n_neighbors=1 in the KNN model, but this can be changed to allow for a more robust matching system.
	•	Face Margin: The margin around the detected face is set to 20px. This value can be increased or decreased in the following line:

margin = 20

Dependencies

This project requires the following libraries, which can be installed using pip install -r requirements.txt.
	•	OpenCV: For image processing.
	•	Dlib: For CNN-based face detection.
	•	DeepFace: For facial recognition (can be replaced with other libraries).
	•	NumPy: For matrix operations and embeddings.
	•	Scikit-learn: For the K-Nearest Neighbors (KNN) model.

File Descriptions

1. app.py

The main file for the entire pipeline. It performs the following tasks:
	1.	Loads the cluster_to_student.json mapping file.
	2.	Detects faces from an input image.
	3.	Extracts face embeddings.
	4.	Loads existing embeddings and cluster labels.
	5.	Matches new face embeddings with existing embeddings.
	6.	Marks and prints student attendance.

Potential Improvements

Here are some ideas to improve the system:
	•	Live Camera Feed: Add support for real-time recognition using a webcam.
	•	Web App Interface: Create a user-friendly interface using Flask or Streamlit.
	•	Batch Processing: Allow for batch processing of multiple images.
	•	More Robust Matching: Use a distance threshold in KNN to filter false positives.
	•	Model Upgrades: Use FaceNet or ResNet-based models for more accurate facial recognition.
	•	Embeddings Management: Add a module to update, delete, or manage the embeddings directly from the app.
	•	Error Handling: Add better exception handling and logging.
	•	Environment File: Use a .env file to store paths and other configuration values.

Contributing

Contributions are welcome! If you’d like to suggest a feature, report a bug, or contribute to the project, follow these steps:
	1.	Fork the repository.
	2.	Create a feature branch.
	3.	Commit changes.
	4.	Submit a pull request.

For any questions, please create an issue on the GitHub repository.

License

This project is licensed under the MIT License. You are free to use, modify, and distribute it as long as proper credit is given.

By following this README, you should be able to set up, run, and understand the Face Recognition Attendance System. Let me know if you’d like any updates or additions to this documentation.