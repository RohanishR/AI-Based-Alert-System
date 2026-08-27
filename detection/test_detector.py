import sys
import os
import cv2
import argparse

# Allow importing from the root project folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camera_feed import CameraFeed
from detector import VehicleDetector

def main():
    parser = argparse.ArgumentParser(description="Test Module 2: Vehicle Detector")
    parser.add_argument("--source", default="0", help="Video source (0 for webcam, or path to video file)")
    parser.add_argument("--skip", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--config", default="detection/config.yaml", help="Path to detector config file")
    args = parser.parse_args()
    
    source = args.source
    if source.isdigit():
        source = int(source)
        
    print("1. Initializing CameraFeed (Module 1)...")
    feed = CameraFeed(source=source, frame_skip=args.skip)
    
    print("2. Initializing VehicleDetector (Module 2)...")
    try:
        detector = VehicleDetector(config_path=args.config)
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config}.")
        print("Please ensure you are running this script from the project root directory:")
        print("Example: python detection/test_detector.py")
        sys.exit(1)
        
    try:
        feed.open()
        print("\nPipeline active. Press 'Q' to stop the video display.")
        
        while feed.is_opened():
            # Step 1: Get frame from CameraFeed
            frame_data = feed.read()
            
            if frame_data is None:
                print("End of video stream.")
                break
                
            frame = frame_data["frame"]
            
            # Step 2: Pass frame to VehicleDetector
            detections, inference_time = detector.detect(frame_data)
            
            # Step 3: Draw detections for demo purposes
            annotated_frame = detector.draw_detections(frame, detections, inference_time)
            
            # Draw standard Module 1 metadata overlay
            overlay_text = f"Frame: {frame_data['frame_index']} | Time: {frame_data['timestamp']:.2f}s"
            cv2.rectangle(annotated_frame, (5, 5), (400, 40), (0, 0, 0), -1)
            cv2.putText(annotated_frame, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display results
            cv2.imshow("Module 1 + Module 2 Pipeline Demo", annotated_frame)
            
            # Wait 1ms and check if user pressed 'q'
            if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
                print("User requested quit.")
                break
                
    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")
    finally:
        feed.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        print("Pipeline shut down safely.")

if __name__ == "__main__":
    main()
