from threading import Thread
import cv2 as cv
import time


class VideoCaptureThread:

	def __init__(self, frame_rate):
		self.stream = cv.VideoCapture(0)
		_, self.frame = self.stream.read()
		self.stopped = False
		wCam, hCam = 640, 320
		self.stream.set(3, wCam)
		self.stream.set(4, hCam)	
		self.prev = 0
		self.frame_rate = frame_rate


	def start(self):
		Thread(target=self.update, args=()).start()
		return self
	

	def update(self):
		prev = time.time()
		while True:
			if self.stopped:
				return
			time_elapsed = time.time() - prev
			_, frame = self.stream.read()
			if time_elapsed > 1 / self.frame_rate:
				prev = time.time()
				self.frame = frame
	

	def read(self):
		return self.frame
	

	def stop(self):
		self.stopped = True