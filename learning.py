'''
Python
SVM with implemented Bag of words

Raymond Tse
Jonathan Beekman



'''


import numpy as np
import cv2
import sys
import os
import pickle
from scipy.spatial import cKDTree as sp
import time
from sklearn import svm


def getFeatures(image,desc):
	# Use sift to extract features
	kp, des = desc.detectAndCompute(image,None)
	return des


def createWords(images):
	k = 1000
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
	flag = cv2.KMEANS_RANDOM_CENTERS

	desc = cv2.SIFT()
	obs = []

	features = []
	for image in images:
		feats = getFeatures(image,desc)
		for f in feats:
			features.append(f)
	#print len(features)
	f_array = np.array(features,dtype=np.float32)
	#print f_array.shape

	dev, labels, centers = cv2.kmeans(f_array, k, criteria, 1, flag)

	return (labels, centers)

# Function to turn an image keypoint features into a histogram based off given centers
def ItoHist(features,centers):
	#print features
	indices = findClosestCenter(features,centers)
	b = [j for j in range(len(centers))]
	#print("made hist")
	h,_ = np.histogram(indices,bins=b) 
	#print h
	return h
	#return count

def calcDist(a,b):
	diff = np.subtract(a,b)
	return np.sum(np.square(diff))

def findClosestCenter(f,centers):
	# Brute force naive
	#print ("in findClosestCenter")
	#print f
	'''
	minIndex = 0
	minDist = calcDist(f,centers[0])
	for p in range(len(centers)):
		d = calcDist(f,centers[p])
		if d < minDist:
			minDist = d
			minIndex = p
	return minIndex
	'''
	# Use scipy cKD-trees to maximize speed
	tree = sp(centers)
	_,i = tree.query(f)
	return i

def genHists(dirpath,pfoldername,centers):
	# Function to train a svm with one vs all linear svm
	# Generate sets of data for training
	histsdata = []
	classification = []
	s = cv2.SIFT()
	for d in os.listdir(dirpath):
		path = dirpath+"/"+d
		#print d
		if d == "Nothing":
			continue
		if os.path.isdir(path):
			#print("isdir")
			f = dirToHists(path,s,centers)
			if d == pfoldername:
				c = 1
			else:
				c = -1
			for n in f:
				histsdata.append(n.flatten())
				classification.append(c)
	#pickle histograms to a seperate file
	data = {"hist": histsdata, "classes":classification}
	#print(data)
	return data
			
def dirToHists(d,s,centers):
	# given directory d, return list of histograms of images in d
	hists = []
	for i in os.listdir(d):
		if i == ".DS_Store":
			continue
		else:
			# create hist based on type and appends to hists
			img = cv2.imread((d+"/"+i))
			# get sift features
			_,f = s.detectAndCompute(img,None)
			print f
			# h is 1d array as "histogram"
			h = ItoHist(f,centers)
			hists.append(h)
	# hists has an array/histogram for every image
	#print(hists)
	return hists


def squarePossibilities(square, squares_x, squares_y, square_length):

	magic_value = squares_x * squares_y
	for i in range(magic_value):
		max_x = square_length + 1
		max_y = square_length + 1
		max_x += (i % squares_x) * square_length
		max_y += ((i / squares_x) % squares_y) * square_length

		checkForTile(square[0 : max_x, 0 : max_y, :], square_length)

def slidingWindow(image, square_length):
	# Let the window be a 4x4 of squares of square_length
	# Avoid the edge cases
	end_x = image.shape[0] - 3 * square_length
	end_y = image.shape[1] - 3 * square_length
	for i in range(0, end_x, square_length):
		for j in range(0, end_y, square_length):
			edge_x = i + 4 * square_length + 1
			edge_y = j + 4 * square_length + 1
			# Do stuff here, using these parameters
			squarePossibilities(image[i : edge_x, j : edge_y, :], square_length)

def scanBase(base,svms,categories,centers,fast=False):
	# Given an image of a base, scan for buildings
	img = cv2.imread(base)
	newx,newy = img.shape[1]/2,img.shape[0]/2 #new size (w,h)
 	img = cv2.resize(img,(newx,newy))
	windowsize= (np.ceil([.15 * img.shape[0]]), np.ceil([.1 * img.shape[1]]))
	clones = []
	sift = cv2.SIFT()
	# Slow method with non maximal supression algorithm with code that doesn't belong to me
	if not fast:
		for c in range(len(categories)):
			if categories[c] == "nothing":
				continue
			else:
				boundingrects = []
				clone = img.copy()
				s = svms[categories[c]]
				for (x,y,window) in slidingwindow2(img,20,windowsize):
					#Perform classifier on slice
					snip = img[y:y+windowsize[1],x:x+windowsize[0]]
					_,f = sift.detectAndCompute(snip,None)
					if f == None:
						continue
					# h is 1d array as "histogram"
					h = ItoHist(f,centers)
					prediction = s.predict(h)
					if prediction[0] == 1:
						# It classified the object as c
						#cv2.rectangle(clone, (x,y), (x+windowsize[0],y+windowsize[1]), (0,255,0),2)
						boundingrects.append((x,y,x+windowsize[0],y+windowsize[1]))
					#print 6
				boundingrects = np.asarray(boundingrects)
				#print boundingrects
				best_bounds = nms.non_max_suppression_slow(boundingrects, 0.3)
				for (startX, startY, endX, endY) in best_bounds:
					cv2.rectangle(clone, (int(startX),int(startY)), (int(endX),int(endY)), (0,255,0),2)
				clones.append(clone)
	else:
		# Fast algorithm with no non-maximal suppression
		for c in range(len(categories)-1):
			clones.append(img.copy())
		for (x,y,window) in slidingwindow2(img,20,windowsize):
					#Perform classifier on slice
					snip = img[y:y+windowsize[1],x:x+windowsize[0]]
					_,f = sift.detectAndCompute(snip,None)
					if f == None:
						continue
						# h is 1d array as "histogram"
					h = ItoHist(f,centers)
					for c in range(len(categories)-1):
						if categories[c] == "nothing":
							c = c-1
							continue
						s = svms[categories[c]]
						prediction = s.predict(h)
						if prediction[0] == 1:
							# It classified the object as c
							cv2.rectangle(clones[c], (x,y), (x+windowsize[0],y+windowsize[1]), (0,255,0),2)
							#boundingrects.append((x,y,x+windowsize[0],y+windowsize[1]))

	for pic in clones:
		cv2.imshow("img",pic)
		cv2.waitKey(0)

def slidingwindow2(image,stepsize,windowsize):
	for y in xrange(0,image.shape[0], stepsize):
		for x in xrange(0, image.shape[1], stepsize):
			yield(x,y,image[y:y + windowsize[1],x:x+windowsize[0]])


if __name__ == "__main__":
	path = "./images/"
	categories = ["air_defence","air_sweeper","aqueen_alter","archer_tower",
						"army_camp","barracks","bking_alter","builder_hut","cannon","cc","dark_barracks",
						"de_collector","de_storage","elixer_collector","elixer_storage","gold_collector",
						"gold_storage","inferno_tower","laboratory","mortar","spell_factory","town_hall",
						"wiz_tower","xbow"]

	if sys.argv[1] == "build":
		imglist = []
		# Scan through training folder
		basepath = path +"train/"
		# Build image list
		for d in os.listdir(basepath):
			path = basepath+d
			#print path
			if d == "nothing":
				continue
			if os.path.isdir(path):
				for f in os.listdir(path):
					if f == ".DS_Store":
						continue
					else:
						im = cv2.imread(path+"/"+f)
						imglist.append(im)
		#print(len(imglist))
		labs, centers = createWords(imglist)
		# pickle centers
		pickle.dump(centers, open("centers.p","wb"))
		print("Pickling centers complete!")

	elif sys.argv[1] == "train":
		# perform training based on centers from pickled data
		# Reads pickled data
		centers = pickle.load(open("centers.p", "r"))
		# Pickle image data to create svms with later
		
		starttime = time.time()

		for c in categories:
			if c == "nothing":
				continue
			# pickle training feature data to be fit to each svm
			training = genHists(path+"train/",c,centers)
			filename = c+"_training"
			pickle.dump(training, open(filename+".p", "wb"))
			print("Pickled training features")

		endtime = time.time()
		print "Ran for "+ str(endtime - starttime)
	# Pickles individual testing image buildings
	elif sys.argv[1] == "train2":
		centers = pickle.load(open("centers.p", "r"))
		starttime = time.time()
		#pickle histograms of test images
		testing = []
		s = cv2.SIFT()
		for d in os.listdir(path+"test/"):
			newpath = path+"test/"+d
			#print d
			if os.path.isdir(newpath):
				f = dirToHists(newpath,s,centers)
				testing.append(f)
		filename = "testdata"
		pickle.dump(testing, open(filename+".p", "wb"))
		print("Pickled test features")

		endtime = time.time()
		print "Ran for "+ str(endtime - starttime)

	elif sys.argv[1] == "test":
		#confusion matrix
		cmat = np.zeros((len(categories),len(categories)),dtype=np.int)
		svms = {}
		# list of lists(directories) of hists(images)
		testdata = np.asarray(pickle.load(open("testdata.p","r")))
		#print testdata.shape
		for c in categories:
			# Read in pickled data
			traindata = pickle.load(open(c+"_training.p","r"))
			hists = np.asarray(traindata["hist"])
			classes = np.asarray(traindata["classes"])
			clf = svm.LinearSVC()
			clf.fit(traindata["hist"],traindata["classes"])
			svms[c] = clf
		# For each image, use all svms on it
		for i in range(len(testdata)):
			print i
			for img in range(len(testdata[i])) :
				print img
				# Apply each svm to it
				votes = np.zeros(len(categories),dtype=np.int)
				sheeple = []
				for c in range(len(categories)):
					print c
					s = svms[categories[c]]
					prediction = s.predict(testdata[i][img])
					if prediction[0] == 1:
						# It classified the object as c
						votes[c] += 1
						sheeple.append(c)
					print(votes)
				if len(sheeple) > 1:
					# conflict, more than one svm thinks it identifies the image
					# Use larger decision function to decide
					maxIndex = 0
					maxDecision = 0
					for index in sheeple:
						d = svms[categories[index]].decision_function(testdata[i][img])
						print(d)
						if np.fabs(d) > maxDecision:
							maxDecision = np.fabs(d)
							maxIndex = index
					cmat[i][maxIndex]
				elif len(sheeple) == 0:
					# No svm identifies it
					pass
				else:
					# One svm identifies it
					index = np.argmax(votes)
					cmat[i][index] += 1

		print cmat
	
	elif sys.argv[1] == "apply":
		if sys.argv[2] == None:
			pass
		else:
			centers = pickle.load(open("centers.p", "r"))
			svms = {}
			# list of lists(directories) of hists(images)
			testdata = np.asarray(pickle.load(open("testdata.p","r")))
			#print testdata.shape
			for c in categories:
				# Read in pickled data
				if c == "nothing":
					continue
				traindata = pickle.load(open(c+"_training.p","r"))
				hists = np.asarray(traindata["hist"])
				classes = np.asarray(traindata["classes"])
				clf = svm.LinearSVC()
				clf.fit(traindata["hist"],traindata["classes"])
				svms[c] = clf
			#scanBase(sys.argv[2],svms,categories,centers)
			scanBase(sys.argv[2],svms,categories,centers,fast=True)




