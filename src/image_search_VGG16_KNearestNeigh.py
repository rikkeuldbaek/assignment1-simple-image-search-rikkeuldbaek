### Visual Assignment 1
## Cultural Data Science - Visual Analytics 
# Author: Rikke Uldbæk (202007501)
# Date: 2nd of May 2023

#--------------------------------------------------------#
### IMAGE SEARCH USING VGG16 AND K-NEAREST NEIGHBORUS ###
#--------------------------------------------------------#

# (please note that some of this code has been adapted from class sessions)


#loading packages
# base tools
import os, sys 

# cv2
import cv2

# data analysis 
import numpy as np
import pandas as pd
from numpy.linalg import norm
from tqdm import notebook

 
# tensorflow
import tensorflow_hub as hub
from tensorflow.keras.applications.vgg16 import (VGG16, 
                                                 preprocess_input)

# K-Nearest Neighbours
from sklearn.neighbors import NearestNeighbors

#utils (helper functions)
sys.path.append(os.path.join("utils"))
import imutils as hf

# Scripting
import argparse


############### PARSER FUNCTION ###############

def input_parse():
    #initialise the parser
    parser = argparse.ArgumentParser()
    #add arguments
    parser.add_argument("--target_flower_image", type= str, default= "image_0021.jpg", help= "Specify filename of target image.")
    # parse the arguments from the command line 
    args = parser.parse_args()
    #define a return value
    return(args)


################## LOAD MODEL ##################

def model_load():
    model = VGG16(weights='imagenet', 
                include_top=False,
                pooling='avg',
                input_shape=(224, 224, 3))
    return model


############### EXTRACT FEATURES ################

def feature_extraction(model):  
    # path to flowers
    path = os.path.join(os.getcwd(), "data", "flowers") #fix path
    # full path of all flowers
    filenames = [path + "/"+name for name in sorted(os.listdir(path))]


    # extract features from every image in the "flowers" folder
    feature_list = []
    for i in notebook.tqdm(range(len(filenames)), position =0, leave = True): #iterate over filenames
        feature_list.append(hf.extract_features(filenames[i], model))

    return(filenames, feature_list)


######## INITIALIZING K-NEAREST NEIGHBORS #########

def initializing_KNN(feature_list):
    neighbors = NearestNeighbors(n_neighbors=10,  #find 10 nearest neighbors
                                algorithm='brute',
                                metric='cosine').fit(feature_list) #fitting the knearest clustering model to the features list
    return(neighbors)


########### FIND INDEX OF TARGET IMAGE #############
def find_index(filenames, target_flower_image):
    index_path = os.path.join(os.getcwd(), "data", "flowers/")
    target_index =filenames.index(index_path+target_flower_image)
    return(target_index)



######### APPPLYING K-NEAREST NEIGHBORS  ##########

def applying_KNN(target_index, neighbors, feature_list, filenames):
    distances, indices = neighbors.kneighbors([feature_list[target_index]]) #target image 20

    idxs = []
    files =[]
    dist = []

    # Loop through indices but not the first one (i.e, the target image)
    for i in range(1,6): 
        idxs.append(indices[0][i]) #fetch index
        dist.append(distances[0][i]) #fetch cosine distance

    # Get image names
    for j in idxs:
        files.append(filenames[j])
        characters_to_remove =os.path.join(os.getcwd(), "data", "flowers/")
        files = [k.replace(characters_to_remove, "") for k in files]
        

    Iseries = pd.Series(files, name= 'images')
    Dseries = pd.Series(dist, name='distance')
    df =pd.concat([Iseries , Dseries], axis= 1)

    return(df, filenames, idxs, target_index)




############## SAVE OUTPUT ##############

def save_function(df, filenames, idxs, target_index):
    for i in df["images"]:
        out_paths = os.path.join(os.getcwd(), "out", "VGG16_KNearestNeigh_out", i)
        read_paths = os.path.join(os.getcwd(), "data", "flowers", i)
        top_images = cv2.imread(read_paths)
        cv2.imwrite(out_paths, top_images)


    # save target image
    target_outpath = os.path.join(os.getcwd(), "out", "VGG16_KNearestNeigh_out")
    target_img = cv2.imread(filenames[target_index])
    cv2.imwrite(target_outpath+"/"+"target_image.jpg",target_img)

    # save top 5 as .csv
    df = pd.DataFrame(df)
    outpath_csv = os.path.join(os.getcwd(), "out", "VGG16_KNearestNeigh_out", "top5_VGG16_KNear.csv")
    df.to_csv(outpath_csv) 

    return()




############# MAIN FUNCTION #############
def main():
    # input parse
    args = input_parse()
    # pass arguments to script functions
    model =model_load()
    filenames, feature_list= feature_extraction(model)
    neighbors = initializing_KNN(feature_list)
    target_index =find_index(filenames, args.target_flower_image)
    df, filenames, idxs, target_index = applying_KNN(target_index, neighbors, feature_list, filenames)
    save_function(df, filenames, idxs, target_index)

main()
