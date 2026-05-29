import cv2
import threading
import time

class RealTimeVideoStream:
    """
    Robust Multi-Protocol Stream Intake State Machine.
    Dedicated background thread strictly for polling `cap.read()` to clear hardware
    queues and prevent frame backlog. Incorporates exponential backoff.
    """
    def __init__(self, src):
        self.src = src
        self.stream = None
        self.ret = False
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        self.retry_count = 0
        
        self._init_stream()

    def _init_stream(self):
        # Hardware Bus Target vs Network Protocol
        target = int(self.src) if str(self.src).isdigit() else self.src
        self.stream = cv2.VideoCapture(target)
        if self.stream.isOpened():
            self.ret, frame = self.stream.read()
            if self.ret:
                self.frame = frame

    def start(self):
        t = threading.Thread(target=self.update, daemon=True)
        t.start()
        return self

    def update(self):
        while not self.stopped:
            if self.stream is None or not self.stream.isOpened():
                self._handle_reconnect()
                continue
                
            ret, frame = self.stream.read()
            if not ret:
                self._handle_reconnect()
                continue
                
            # Success, clear retry
            self.retry_count = 0
            with self.lock:
                self.ret = ret
                self.frame = frame
                
    def _handle_reconnect(self):
        self.ret = False
        self.retry_count += 1
        
        if self.stream:
            self.stream.release()
            
        # Exponential Backoff Reconnection Machine
        if self.retry_count == 1:
            sleep_dur = 5
        elif self.retry_count == 2:
            sleep_dur = 10
        elif self.retry_count == 3:
            sleep_dur = 15
        else:
            sleep_dur = 30
            
        print(f"Camera feed lost. Attempt {self.retry_count}, retrying in {sleep_dur}s...")
        time.sleep(sleep_dur)
        self._init_stream()

    def read(self):
        with self.lock:
            if self.ret and self.frame is not None:
                return self.ret, self.frame.copy()
            return self.ret, None

    def stop(self):
        self.stopped = True
        if self.stream:
            self.stream.release()
