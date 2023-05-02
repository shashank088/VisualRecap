import cv2
import os
import shutil
import math
import random
from os.path import isfile, join
import numpy as np
from functools import lru_cache
from numpy.linalg import norm
from sklearn.metrics import f1_score
import pandas as pd

dir_video = "./tvsum50_ver_1_1/ydata-tvsum50-v1_1/ydata-tvsum50-video/video/"
dir_object = "./tvsum50_ver_1_1/ydata-tvsum50-v1_1/ydata-tvsum50-thumbnail/thumbnail/"

#function to count no of files inside a directory which will be used to count the no of frames in a video
def count_files(dir_path):
	count = 1
	for path in os.listdir(dir_path):
    # check if current path is a file
		if os.path.isfile(os.path.join(dir_path, path)):
			count += 1
	return count
def create_frames(video_path,frame_path):
	# Read the video from specified path
	cam = cv2.VideoCapture(video_path)
	print(frame_path)
	try:
		
		# creating a folder named data
		if os.path.exists(frame_path):
			shutil.rmtree(frame_path)
		os.makedirs(frame_path)


	# if not created then raise error
	except OSError:
		print ("Error: Creating directory of data")

	currentframe = 1

	while(True):
		
		# reading from frame
		ret,frame = cam.read()

		if ret:
			# if video is still left continue creating images
			name = frame_path+"/"+"frame" + str(currentframe) + ".jpg"
			# print ("Creating..." + name)

			# writing the extracted images
			cv2.imwrite(name, frame)

			# increasing counter so that it will show how many frames are created
			currentframe += 1
		else:
			break

	# Release all space and windows once done
	cam.release()
	cv2.destroyAllWindows()



#function to compute mean square error between two images
def mse(img1, img2):
	img1 = cv2.resize(img1, (224, 224)).astype(np.float32)
	img2 = cv2.resize(img2, (224, 224)).astype(np.float32)
	h, w = img1.shape
	diff = cv2.subtract(img1, img2)
	err = np.sum(diff**2)
	mse = err/(float(h*w))
	return mse,diff


#fucntion to read images and get error or difference between them
@lru_cache
def get_error(path1,path2):
	img1 = cv2.imread(path1)
	img2 = cv2.imread(path2)
	img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

	error,diff = mse(img1, img2)
	# print("Image matching Error between the two images:", error)
	# cv2.imshow("Difference",diff)
	return error


#function to generate m random numbers uniformly out of numbers from 1 to n
def get_rand_indices(n,m):
	list = random.sample(range(1,n),m)
	return list


#function to get best indices out of that firefly binary vector
def get_best_indices(firefly,m):
	ind = np.argpartition(firefly,-m)[-m:]
	ind = np.sort(ind)
	return tuple(np.array(ind))

#function to compute similarity of object of interest with summary frames
@lru_cache
def similarity_factor(ind,filename):
	error = 0
	for i  in ind:
		error += get_error(dir_object+filename+".jpg","./data/frame"+str(i)+".jpg")
	return 1/error


#function to get other factor of intensity which accounts for dissmilarity between adjacent frames so that there are no redundant frames
@lru_cache
def adj_dissimilarity_factor(ind,filename):
	error = 0
	for i in range(0,len(ind)-1):
		error += get_error("./data/frame"+str(ind[i])+".jpg","./data/frame"+str(ind[i+1])+".jpg")
	
	return error


#function to compute intensity of a firefly vector
def intensity(firefly,m,filename):
	ind = get_best_indices(firefly,m)
	alpha = 0.1
	beta = 0.2
	sf = similarity_factor(ind,filename)
	adf = adj_dissimilarity_factor(ind,filename)
	# return sf
	return (sf*alpha + adf*beta)//(alpha+beta)


def s(r):
	f = 0.5
	l = 1.5
	return f*math.exp(-r/l) - math.exp(-r)


def firefly(frame_path,filename):
	# create 50 random binary vectors of size equal to no of frames in the video and having certain no of set bits (indicating that the particular frame is there in the summary) 

	n = count_files(frame_path) -1
	count = 30
	m = n//7
	alpha = 0.1
	firefly = []
	for i in range(0,count):
		indices = get_rand_indices(n,m)
		list = [0 for element in range(n)]
		vector = np.array(list)
		for i in indices:
			vector[i] = 1.0
		firefly.append(vector)

	
	# now we run a loop till max_iter no of times and in each iteration we will be updating our fireflies

	max_iter = 10
	max_intensity = -1000000
	max_intensity_firefly = [0 for element in range(n)]

	for itr in range(0,max_iter):
		print("Iteration : "+str(itr))
		for i in range(0,count):
			for j in range(0,count):
				if intensity(firefly[j],m,filename) > intensity(firefly[i],m,filename):
					# rand_vector = get_rand_vector(n)
					firefly[i] = firefly[i] + (math.exp(-math.pow(dis(firefly[i],firefly[j]),2)))*(firefly[j]-firefly[i]) 

		#finding best firefly after each iteration
		for k in range(0,count):
			if intensity(firefly[k],m,filename) > max_intensity:
				max_intensity = intensity(firefly[k],m,filename)
				max_intensity_firefly = firefly[k]
	
	#instead of the best firefly we are interested in its best indices
	return get_best_indices(max_intensity_firefly,m)



#function to get distance between two fireflies
def dis(point1,point2):
	
	dist = np.linalg.norm(point1 - point2)
	return dist


#function to convert frames to videos
def convert_frames_to_video(pathIn,frames,pathOut,fps):
    frame_array = []
    # files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    #for sorting the file names properly
    # files.sort(key = lambda x: int(x[5:-4]))
    for i in range(len(frames)):
        filename=pathIn + "frame" + str(frames[i])+".jpg"
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        # print(filename)
        #inserting the frames into an image array
        frame_array.append(img)
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


def generate_bin_vector(filename):
	create_frames(dir_video+filename+".mp4","./data")
	n = count_files('./data')
	print("No Of Frames is : "+str(n))
	best_frames_generated = firefly("./data",filename)
	B = [0 for element in range(n-1)]
	for i in best_frames_generated:
		B[i-1] = 1

	# convert_frames_to_video("./data/",best_frames_generated,"generated.mp4",30)
	return B

def format_vector(vector,n):
    # Determine the number of rows required for the matrix
    num_rows = int(np.ceil(len(vector) / n))
    
    # Pad the vector with zeros to ensure its length is a multiple of 5
    padded_vector = np.pad(vector, (0, num_rows * n - len(vector)), mode='constant')
    
    # Reshape the padded vector into a matrix with n columns
    matrix = padded_vector.reshape(num_rows, n)
    
    return matrix

def main():
	file1 = open("./reference_bin_vectors.txt",'r')
	lines = file1.readlines()
	reference_bin_vectors_mat = []
	for line in lines:
		a = line.split(",")
		reference_bin_vectors_mat.append([int(x) for x in a])


	# read the excel sheet into a pandas dataframe
	df = pd.read_excel('tvsum50_ver_1_1/ydata-tvsum50-v1_1/ydata-tvsum50-data/data/ydata-tvsum50-info.xls')

	# iterate through the values of a particular column
	filenames = []
	for index, row in df.iterrows():
		value = row['video_id'] 
		filenames.append(value)


	generated_bin_vectors_mat = []

	for i in range(0,len(filenames)):
		if i==1:
			break

		B = generate_bin_vector(filenames[i])
		generated_bin_vectors_mat.append(B)
	
	with open('generated_bin_vectors.txt', 'w') as f:
		for row in generated_bin_vectors_mat:
			f.write(','.join(str(elem) for elem in row))
			f.write('\n')

	result_vector = []
	start = 0
	for generated_bin_vector in generated_bin_vectors_mat:
		max_score = -1
		for i in range(start,start+20):
			max_score = max(max_score,f1_score(reference_bin_vectors_mat[i],generated_bin_vector))
		result_vector.append(max_score)
		start = start+20


	category_vector = format_vector(result_vector,1)

	print(category_vector)



if __name__ == "__main__":
    main()


