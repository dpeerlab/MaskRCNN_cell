import os
import sys
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import imageio
import imageio as si
import scipy.ndimage
import keras
import random
import tensorflow as tf
import tensorflow
import mrcnn.model as modellib
import imgaug as ia
import imgaug.augmenters as iaa
from mrcnn.CellConfig import CellConfig as CellConfig
from mrcnn.CellDataset import CellDataset as CellDataset
from mrcnn.model import log
from mrcnn import visualize
from mrcnn import  utils 
from pycocotools import mask
from skimage.filters import roberts, sobel, sobel_h, sobel_v, scharr, scharr_h, scharr_v, prewitt, prewitt_v, prewitt_h
import pycocotools.mask as mask
from skimage.measure import find_contours
from skimage import measure 
import numpy as np
from itertools import groupby
import json
import pycocotools

#Matplotlib function
def get_ax(rows=1, cols=1, size=30):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows),dpi=150)
    return ax

#Visualizes the mask and overlaying the mask onto the original image
#Input: Reference to model, image shape: (X,Y,3)
def visualize_results(model, img):
    results = model.detect([img], verbose=0)
    r = results[0]
    print("Number of mask instances: " + str(r['masks'].shape[2]))
    
    z = np.zeros((img.shape[0],img.shape[1]))
    for i in range((r['masks'].shape[2])):
        z += r['masks'][:,:,i]
    z = np.where(z > 0.01, 1, 0)
    plt.figure()
    plt.imshow(z)
    edges = np.where(sobel(z) > 0, 255, 0)
    plt.figure()
    original_image = img.astype(np.uint8)
    original_image[:,:,0] = edges
    plt.figure(figsize=(20,20))
    plt.imshow(original_image.astype(np.uint8))

#Need to implement stitching
def showRandomPrediction(model, imgPath):
    img = imageio.imread(imgPath[random.randint(0, len(imgPath)-1)])
    
    if(img.shape[0] >= 512 and img.shape[1] >= 512):
        img = img[:512, :512]
    elif(img.shape[0] >= 256 and img.shape[1] >= 256):
        img = img[:256, :256]
    else:
        img = img[:128, :128]

    print(img.shape)
    visualize_results(model, img)

def createPaths(output_path):
    maskPath = output_path + "/Masks/"
    overlayPath = output_path + "/Overlay/"
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(maskPath):
        os.makedirs(maskPath)
    if not os.path.exists(overlayPath):
        os.makedirs(overlayPath)
        
    return maskPath, overlayPath
        
    

def saveToJSON(maskArray, imagePath):
    SIZE_X = maskArray.shape[0]
    SIZE_Y = maskArray.shape[1]
    image = {
        'name':'',
        'num_annotations': -1,
        'annotations': [],
        'area':[],
        'bounding-box':[]
    }
    
    image['name'] = "Image Output"
    image['num_annotations'] = len(maskArray)
    
    dim = maskArray.shape[2]
    for i in range(dim):
        fortran_ground_truth_binary_mask = np.asfortranarray(maskArray[:,:,i].reshape(SIZE_X,SIZE_Y))
        encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
        ground_truth_area = mask.area(encoded_ground_truth)
        ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
        contours = measure.find_contours(maskArray[:,:,i].reshape(SIZE_X,SIZE_Y), 0.5)
        image['annotations'].append(binary_mask_to_rle(fortran_ground_truth_binary_mask))
        image['area'].append(str(ground_truth_area))
        image['bounding-box'].append(str(ground_truth_bounding_box))
        
    with open(imagePath + '.json', 'w') as json_file:
        json.dump(image, json_file)

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    #print(rle)
    return rle

#decode to mask
def decode(dataName):
    with open(dataName) as json_file:
        data = json.load(json_file)
        rle = data['annotations'][0]
        compressed_rle = mask.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
        x = mask.decode(compressed_rle)
        for i in range(1, len(data['annotations'])):
            rle = data['annotations'][i]
            compressed_rle = mask.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
            x += mask.decode(compressed_rle)
        x = np.where(x > 0.01, 1, 0)
    return x;