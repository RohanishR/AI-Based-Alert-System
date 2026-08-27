import cv2
import argparse
import sys
from camera_feed import CameraFeed

def main():
    parser = argparse.ArgumentParser(description="Test CameraFeed module")
    parser.add_argument("--source", default="0", help="Video source (0 for webcam, or path/RTSP to video file)")
    parser.add_argument("--skip", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--resize", type=str, default="", help="Resize dimensions as WxH, e.g. 640x480")
    args = parser.parse_args()
    
    # Process source argument
    source = args.source
    if source.isdigit():
        source = int(source)
        
    # Process resize argument
    resize_dim = None
    if args.resize:
        try:
            w, h = map(int, args.resize.split('x'))
            resize_dim = (w, h)
        except ValueError:
            print("Error: Invalid resize dimension format. Use WxH, e.g. 640x480")
            sys.exit(1)
            
    print(f"Initializing CameraFeed with:")
    print(f" - Source: {source}")
    print(f" - Frame Skip: {args.skip}")
    print(f" - Resize: {resize_dim}")
    
    feed = CameraFeed(source=source, frame_skip=args.skip, resize_dim=resize_dim)
    
    try:
        feed.open()
        print(f"\nSource opened successfully.")
        print(f"Original Resolution: {feed.width}x{feed.height}")
        print(f"FPS: {feed.fps}")
        if feed.total_frames > 0:
            print(f"Total Frames: {feed.total_frames}")
            
        print("\nPress 'Q' to quit the display window.")
        
        while feed.is_opened():
            data = feed.read()
            if data is None:
                print("End of stream reached or frame could not be read.")
                break
                
            frame = data["frame"]
            frame_index = data["frame_index"]
            timestamp = data["timestamp"]
            
            # Display info overlay on the frame
            overlay_text = f"Frame: {frame_index} | Time: {timestamp:.2f}s"
            
            # Draw black background rectangle for text readability
            cv2.rectangle(frame, (5, 5), (400, 40), (0, 0, 0), -1)
            cv2.putText(frame, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show the frame
            cv2.imshow("Camera Feed Test (Press Q to quit)", frame)
            
            # Wait for 1 ms and check for 'q' key to exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("Quit requested by user.")
                break
                
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        feed.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        print("Resources released gracefully.")

if __name__ == "__main__":
    main()
