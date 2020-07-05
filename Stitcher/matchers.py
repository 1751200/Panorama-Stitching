import cv2
import numpy as np
import copy
# from imagedt.decorator import time_cost


class matchers:
	def __init__(self):
		self.sift = cv2.xfeatures2d.SIFT_create()
		self.surf = cv2.xfeatures2d.SURF_create()
		self.orb = cv2.ORB_create()
		self.akaze = cv2.AKAZE_create()
		self.brisk = cv2.BRISK_create()
		self.fast = cv2.FastFeatureDetector_create()
		self.kaze = cv2.KAZE_create()
		self.detect = {
			"SIFT": self.getSIFTFeatures,
			"SURF": self.getSURFFeatures,
			"ORB": self.getORBFeatures,
			"AKAZE": self.getAKAZEFeatures,
			"BRISK": self.getBRISKFeatures,
			"FAST": self.getFASTFeatures,
			"KAZE": self.getKAZEFeatures
		}
		FLANN_INDEX_KDTREE = 0
		index_params = dict(algorithm=0, trees=5)
		search_params = dict(checks=50)
		self.flann = cv2.FlannBasedMatcher(index_params, search_params)
		self.bf = cv2.BFMatcher()

	def match(self, i1, i2, detect_method, direction=None):
		imageSet1 = self.detect[detect_method](i1)
		imageSet2 = self.detect[detect_method](i2)
		print("Direction : ", direction)
		matches = self.bf.knnMatch(imageSet2['des'], imageSet1['des'], k=2)
		good = []
		matchesMask = [[0, 0] for i in range(len(matches))]
		for i, (m, n) in enumerate(matches):
			if m.distance < 0.8 * n.distance:
				good.append((m.trainIdx, m.queryIdx))
				matchesMask[i] = [1, 0]

		if len(good) > 4:
			pointsCurrent = imageSet2['kp']
			pointsPrevious = imageSet1['kp']

			matchedPointsCurrent = np.float32(
				[pointsCurrent[i].pt for (__, i) in good]
			)
			matchedPointsPrev = np.float32(
				[pointsPrevious[i].pt for (i, __) in good]
				)

			draw_params = dict(matchesMask=matchesMask)

			img = cv2.drawMatchesKnn(i2, pointsCurrent, i1, pointsPrevious, matches, None, **draw_params)

			H, s = cv2.findHomography(matchedPointsCurrent, matchedPointsPrev, cv2.RANSAC, 5)
			return H, img
		return None

	def getSURFFeatures(self, im):
		print("SURF")
		gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		kp, des = self.surf.detectAndCompute(gray, None)
		return {'kp': kp, 'des': des}

	def getSIFTFeatures(self, im):
		print("SIFT")
		gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		kp, des = self.sift.detectAndCompute(gray, None)
		return {'kp': kp, 'des': des}

	def getORBFeatures(self, im):
		print("ORB")
		gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		kp, des = self.orb.detectAndCompute(gray, None)
		return {'kp': kp, 'des': des}

	def getAKAZEFeatures(self, im):
		print("AKAZE")
		gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		kp, des = self.akaze.detectAndCompute(gray, None)
		return {'kp': kp, 'des': des}

	def getBRISKFeatures(self, im):
		print("BRISK")
		gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		kp, des = self.brisk.detectAndCompute(gray, None)
		return {'kp': kp, 'des': des}

	def getFASTFeatures(self, im):
		print("FAST")
		gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		kp, des = self.fast.detectAndCompute(gray, None)
		return {'kp': kp, 'des': des}

	def getKAZEFeatures(self, im):
		print("KAZE")
		gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		kp, des = self.kaze.detectAndCompute(gray, None)
		return {'kp': kp, 'des': des}


