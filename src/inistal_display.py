from imports import cv2, sys

def main(video_path):

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was successfully opened
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        sys.exit()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Display the frame
        key = cv2.waitKey(30)
        if key == ord('q'):  # Quit when 'q' is pressed
            print("Exit key pressed")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <path_to_video>")
        sys.exit()
    
    video_path = sys.argv[1]
    main(video_path)