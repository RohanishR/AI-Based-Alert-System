import cv2
import time

class CameraFeed:
    def __init__(self, source, frame_skip=1, resize_dim=None):
        """
        Initialize the CameraFeed.
        :param source: int (for webcam) or str (for video file/RTSP)
        :param frame_skip: int, process every Nth frame (default 1 means process all frames)
        :param resize_dim: tuple (width, height) to resize frames, or None
        """
        self.source = source
        self.frame_skip = max(1, frame_skip)
        self.resize_dim = resize_dim
        
        self.cap = None
        self.frame_index = 0
        self.internal_frame_counter = 0
        self.start_time = None
        
        # Metadata
        self.fps = 0
        self.total_frames = 0
        self.width = 0
        self.height = 0

    def open(self):
        """
        Open the video source and initialize metadata.
        Raises ValueError if the source cannot be opened.
        """
        self.cap = cv2.VideoCapture(self.source)
        if not self.is_opened():
            raise ValueError(f"Failed to open video source: {self.source}")
        
        # Read metadata
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.start_time = time.time()
        
    def is_opened(self):
        """
        Check if the video source is opened successfully.
        :return: bool
        """
        return self.cap is not None and self.cap.isOpened()
        
    def read(self):
        """
        Read the next frame from the video source, applying frame skipping and resizing.
        :return: dict with frame, frame_index, timestamp, or None if end of stream/error
        """
        if not self.is_opened():
            return None
            
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                return None  # End of stream or error reading frame
                
            self.internal_frame_counter += 1
            
            # Frame skipping logic
            if self.internal_frame_counter % self.frame_skip != 0:
                continue
                
            # Resize if needed
            if self.resize_dim:
                frame = cv2.resize(frame, self.resize_dim)
                
            self.frame_index += 1
            
            # Calculate timestamp
            # Use the video's current position in milliseconds if available,
            # otherwise fallback to real time elapsed since start (useful for webcams/streams)
            timestamp_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            if timestamp_ms >= 0:
                timestamp = timestamp_ms / 1000.0
            else:
                timestamp = time.time() - self.start_time
                
            return {
                "frame": frame,
                "frame_index": self.frame_index,
                "timestamp": timestamp
            }
            
    def release(self):
        """
        Release the video source and clean up.
        """
        if self.cap:
            self.cap.release()
            self.cap = None
