from imports import *

def augment_image(image):
    """
    Applies random augmentations to the image.
    Augmentations: Rotation, Brightness, Gaussian Blur, Horizontal Flip.
    """
    angle = np.random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((image.shape[1]//2, image.shape[0]//2), angle, 1)
    rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    brightness_factor = np.random.uniform(0.9, 1.1)
    bright_image = np.clip(rotated_image * brightness_factor, 0, 255).astype(np.uint8)

    if np.random.rand() > 0.5:
        blurred_image = cv2.GaussianBlur(bright_image, (3, 3), 0)
    else:
        blurred_image = bright_image

    if np.random.rand() > 0.5:
        flipped_image = cv2.flip(blurred_image, 1)
    else:
        flipped_image = blurred_image

    return flipped_image


def process_frames(frame_numbers):
    """
    Process each frame to detect faces, augment them, extract embeddings, and save the results.
    
    Args:
        frame_numbers (list of int): List of frame numbers to process.
    """
    # Load the FaceNet model (pre-trained on VGGFace2)
    facenet_model = InceptionResnetV1(pretrained='vggface2').eval()

    # Load the Dlib CNN face detector 
    cnn_face_detector = dlib.cnn_face_detection_model_v1('./models/mmod_human_face_detector.dat')

    # Clear existing directories
    if os.path.exists('./frames/face_boxes'):
        shutil.rmtree('./frames/face_boxes')
    if os.path.exists('./frames/face_crops'):
        shutil.rmtree('./frames/face_crops')
    if os.path.exists('./frames/embeddings'):
        shutil.rmtree('./frames/embeddings')
    
    # Create directories to store output images and embeddings
    os.makedirs('./frames/face_boxes', exist_ok=True)
    os.makedirs('./frames/face_crops', exist_ok=True)
    os.makedirs('./frames/embeddings', exist_ok=True)

    variance_of_embedding = []
    for frame_number in frame_numbers:
        # Load the image
        image_path = os.path.join('./frames', f'frame_{frame_number}.jpg')
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            continue

        # Convert the image to grayscale (dlib works better with grayscale)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces using the CNN detector
        faces = cnn_face_detector(gray, 1)  # The second argument is the upsampling factor

        # Loop through detected faces
        for i, face in enumerate(faces):
            x1, y1, x2, y2 = face.rect.left(), face.rect.top(), face.rect.right(), face.rect.bottom()
            margin = 20  # Add margin to avoid cutting the face
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(image.shape[1], x2 + margin)
            y2 = min(image.shape[0], y2 + margin)

            # Extract the face region from the image
            face_crop = image[y1:y2, x1:x2]

            # Apply data augmentation to the face
            augmented_faces = [face_crop]  # Original face
            for _ in range(3):  # Create 3 augmented versions of the face
                augmented_faces.append(augment_image(face_crop))

            for j, face in enumerate(augmented_faces):
                face_crop_resized = cv2.resize(face, (160, 160))
                face_crop_tensor = torch.tensor(face_crop_resized).permute(2, 0, 1).float().div(255).unsqueeze(0)
                
                if torch.cuda.is_available():
                    facenet_model = facenet_model.cuda()
                    face_crop_tensor = face_crop_tensor.cuda()
                
                with torch.no_grad():
                    face_embedding = facenet_model(face_crop_tensor).cpu().numpy().flatten()
                
                variance_of_embedding.append(np.var(face_embedding))
                if np.var(face_embedding) < 0.00194:
                    print(f"Low-quality embedding for frame {frame_number}, face {i+1}")
                    continue
                
                embedding_filename = f'frame_{frame_number}_face_{i+1}_aug_{j}.npy'
                embedding_path = os.path.join('./frames/embeddings', embedding_filename)
                np.save(embedding_path, face_embedding)
                
                face_filename = f'frame_{frame_number}_face_{i+1}_aug_{j}.jpg'
                face_crop_path = os.path.join('./frames/face_crops', face_filename)
                cv2.imwrite(face_crop_path, face)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        output_path = os.path.join('./frames/face_boxes', f'frame_{frame_number}_face_boxes.jpg')
        cv2.imwrite(output_path, image)

    print("Face detection, augmentation, and embedding extraction completed for the frames")


def parse_arguments():
    """
    Parse command-line arguments for the script.

    Returns:
        Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Extract, augment, and embed faces from specific frames.")
    parser.add_argument('--frames', type=str, required=True, 
                        help="Comma-separated list of frame numbers to process (e.g., 100,200,300,400,500)")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    frame_numbers = [int(num) for num in args.frames.split(',')]
    process_frames(frame_numbers)