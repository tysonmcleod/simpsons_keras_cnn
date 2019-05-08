import numpy as np # array operations
import matplotlib.pyplot as plt  #to show image
import os   # to iterate through directories and join paths
import cv2 # image operations
import random # shuffle the training data
import pickle # to just load the dataset??

DATADIR = "dataset"
CATEGORIES = ["bart_simpson", "charles_montgomery_burns", "homer_simpson", "krusty_the_clown", "lisa_simpson", "marge_simpson", "milhouse_van_houten", "moe_szyslak", "ned_flanders", "principal_skinner"]

# create training data

training_data = []

# why??
IMG_SIZE_WIDTH = 120
IMG_SIZE_HEIGHT = 120

def create_training_data():
    # iterate through all the images in the different folders
    for category in CATEGORIES:
        # path to bart, burns, homers ... directories
        path = os.path.join(DATADIR, category)
        # class number --> category index
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                # rgb 3 times grayscale
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                # resize
                new_array = cv2.resize(img_array, (IMG_SIZE_WIDTH, IMG_SIZE_HEIGHT))
                training_data.append([new_array, class_num])
            # if some images are buggy
            except Exception as e:
                pass

create_training_data()
# training data consists of same amount of pics, no class weights, balanced data-set
# print(len(training_data)) -> 10794

# shuffle the data
random.shuffle(training_data)

# create feature and label sets
X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

# features can't be a list in Keras
X = np.array(X).reshape(-1,IMG_SIZE_WIDTH,IMG_SIZE_HEIGHT,1) # -1 how many features do we have (catch all), shape of data, 1 --> grayscale

# save data to not do it every time when tweaking the model
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
