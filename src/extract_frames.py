from imports import cv2, os

def extract_frame(video_path, frame_numbers=[100]):
    """
    Extract specific frames from a video and save them as image files.

    Args:
        video_path (str): The path to the video file.
        frame_numbers (list): List of frame numbers to extract and save.
    """
    os.makedirs('frames', exist_ok=True)  # Create 'frames' directory if it doesn't exist
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video was successfully opened
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return
    
    for frame_number in frame_numbers:
        # Set the video position to the frame number
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        # Read the frame from the video
        ret, frame = cap.read()
        
        if ret:
            # Save the frame as an image file
            frame_filename = f'frames/frame_{frame_number}.jpg'
            cv2.imwrite(frame_filename, frame)
            print(f"Frame {frame_number} saved as {frame_filename}")
        else:
            print(f"Error: Could not read the frame {frame_number} from the video")
        
    # Release the video capture object
    cap.release()

if __name__ == "__main__":
    # Replace this with the path to your video file
    video_path = 'classroom.mp4'
    
    # List of frame numbers to extract
    frame_numbers = [100, 200, 300, 400, 500]
    
    # Call the function to extract and save the frames
    extract_frame(video_path, frame_numbers)