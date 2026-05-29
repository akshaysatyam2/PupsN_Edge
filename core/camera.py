import cv2
import threading
import time
import logging

logger = logging.getLogger(__name__)

class VideoStream:
    def __init__(self, source, reconnect_interval=10):
        self.source = source
        self.reconnect_interval = reconnect_interval
        self.stream = None
        self.stopped = False
        self.latest_frame = None
        # print(cv2.getBuildInformation())
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self._connect()
        self.thread.start()

    def _connect(self):
        """Initialize or re-initialize the video capture."""
        if self.stream:
            self.stream.release()
            self.stream = None
        logger.info(f"Trying to connect to video source: {self.source}")
        self.stream = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        if not self.stream.isOpened():
            logger.warning(f"Failed to open video source: {self.source}")

    def update(self):
        """Continuously grab the latest frame, reconnect if needed."""
        while not self.stopped:
            if self.stream is None or not self.stream.isOpened():
                logger.info(f"Video source not opened, retrying in {self.reconnect_interval} seconds...")
                time.sleep(self.reconnect_interval)
                self._connect()
                continue

            ret, frame = self.stream.read()
            if ret:
                with self.lock:
                    self.latest_frame = frame
            else:
                logger.warning("Failed to read frame from video source, reconnecting...")
                self.stream.release()
                self.stream = None
                time.sleep(self.reconnect_interval)
                self._connect()

    def read(self):
        """Return the latest frame"""
        with self.lock:
            return self.latest_frame

    def stop(self):
        """Stop the video stream"""
        self.stopped = True
        if threading.current_thread() != self.thread:
            self.thread.join(timeout=1.0)
        if self.stream:
            self.stream.release()
            self.stream = None
