#TODO: Need to consider the case where a subimage has NO predictions for the region

import warnings
warnings.filterwarnings("ignore")
from mrcnn.main import *
#from NumpyEncoder import *
import glob
import matplotlib.pyplot as plt
import imageio as si
import math
import numpy as np
import copy
import cv2
import pickle
import skimage

import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)


def init(MODEL_PATH,POST_NMS_ROIS_INFERENCE_in=5000, DETECTION_MAX_INSTANCES_in = 1000, DETECTION_MIN_CONFIDENCE_in=0.7,DETECTION_NMS_THRESHOLD_in=0.2, GPU_COUNT_in=1):   
    print("DETECTION_MAX_INSTANCES_in: " + str(DETECTION_MAX_INSTANCES_in))
    ROOT_DIR = os.path.abspath("")
    
    
    '''
    Function: Initialization
    Inputs:
        MODEL_PATH: Mask-RCNN Model Path
        Else: Mask-RCNN Parameters
    Outputs:
        Returns Mask-RCNN Model Variable
    '''
    
    
    class InferenceConfig(CellConfig):
        GPU_COUNT = GPU_COUNT_in
        IMAGES_PER_GPU = 1
        IMAGE_RESIZE_MODE = 'none' #'pad64'
        DETECTION_MAX_INSTANCES = DETECTION_MAX_INSTANCES_in
        DETECTION_MIN_CONFIDENCE = DETECTION_MIN_CONFIDENCE_in
        DETECTION_NMS_THRESHOLD = DETECTION_NMS_THRESHOLD_in
        ROI_POSITIVE_RATIO = 0.8
        RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
        POST_NMS_ROIS_INFERENCE=POST_NMS_ROIS_INFERENCE_in
    
    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    #Load trained model path here. This most recent model path can be found in Train.ipynb.
    model_path = MODEL_PATH
    #Directory of prediction dataset
    #test_directory = 'Test Images/*'

    config = CellConfig()
    inference_config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference", 
                              config=inference_config,
                              model_dir=MODEL_DIR)
    model.load_weights(model_path, by_name=True)

    return model

def reset():
    '''
    Utility Function: Removes all temporary .json files
    Used to ensure that glob.glob() feature works properly
    Inputs:
        None
    Outputs:
        None
    '''
    delFiles = sorted(glob.glob('.*.json'))
    for file in delFiles:
        os.remove(file)
        
def mask_to_rle_nonoverlap(image_id, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    #print(order)
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    plt.figure()
    plt.imshow(mask)
    # Loop over instance masks
    lines = []
    print(order)
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            print("K")
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)

def decode_masks(json):
    x = np.asarray(rle_decode(json['enc_masks'][0],json['shape']))
    for i in range(1,len(json['enc_masks'])):
        x = np.dstack((x, np.asarray(rle_decode(json['enc_masks'][i],json['shape']))))
    return x

def checkOverlap(json):
    a = decode_masks(json).astype(np.uint8)
    a_0 = a[:,:,0]
    for i in range(a.shape[2]):
        a_0 += a[:,:,i]
    if(np.max(a_0) > 1):
        return False
    return True

def remove_overlap(out):
    if(checkOverlap(out)):
        x = decode_masks(out)
        #print(str(len(out['rois']))+ " " + str(len(out['enc_masks'])))
        t = mask_to_rle_nonoverlap('',x,out['scores']).split(',')
        for i in range(len(t)-1):
            t[i] = t[i].replace('\r', '').replace('\n', '')
        t=np.asarray(t[1:])
        out['enc_masks'] = t
        return out
    else:
        return out

def setName(currData, numI=0, numJ=0):
    '''
    Function: Name json files for temporary storage 
    Naming format: .numI_numJ_xStart_xEnd_yStart_yEnd_ShapeX.json
    Format:
    Inputs:
        currData - JSON Dictionary
        numI - INT (DEFAULT = 0) - iterator storage
        numJ - INT (DEFAULT = 0) - iterator storage
    Outputs:
        Updated JSON file
    '''
    #name = '.' + str(numI) + '_' + str(numJ) + '_' + str(currData['xStart'])  + '_' + str(currData['xEnd']) + '_' + str(currData['yStart']) + '_' + str(currData['yEnd']) + '_' + str(currData['shape'][0]) + '.json'
    name = '.' + '_' + str(numI) + '_' + str(numJ) + '_' + str(currData['xStart'])  + '_' + str(currData['xEnd']) + '_' + str(currData['yStart']) + '_' + str(currData['yEnd']) + '_' + str(currData['shape'][0]) + '.json'
    currData['fileName'] = name
    return currData


def removeEntry(currData):
    '''
    Function: Remove entries that are overlapped or too small
    Inputs:
        currData - f(PadPredict(img)) output
    Outputs:
        Copy-constructed modified JSON file
    '''
    #currData = copy.deepcopy(currData)
    A = sorted(list(set(currData['removeList'])),reverse=True)
    #N = len(currData['removeList'])
    N = len(A)
    for i in range(N):
        currData['rois'] = np.delete(currData['rois'],A[i],0)
        currData['scores'] = np.delete(currData['scores'],A[i],0)
        currData['enc_masks'] = np.delete(currData['enc_masks'],A[i],0)
    currData['removeList'] = np.array([])
    return currData

def between(x,a,b):
    '''
    Utility Function: Checks if a value x is between a and b, inclusive
    Inputs:
        x - Comparable data type
        a - Comparable data type
        b - Comparable data type
    Outputs:
        True or False boolean value
    '''
    if(x <= b+1 and x >= a):
        return True
    return False

def overlay(output):
    '''
    Utility Function: Overlay all binary masks from segmentation into 1-dim
    
    Inputs: 
        output - dictionary from f(output)
    Outputs:
        m x n binary array 
    '''
    if(len(output['enc_masks']) != 0 or output['enc_masks'].shape != ()):
        out = rle_decode(output['enc_masks'][0], output['shape'])
        for i in range(len(output['enc_masks'])):
            out += rle_decode(output['enc_masks'][i], output['shape'])
        return out
    else:
        return np.zeros(output['shape'])


def area(rle_mask, shape):
    '''
    Function: Measure pixel-based area of each segmented cell
    Inputs:
        rle_mask - one instance segmentation in RLE form
        shape.- image shape (x,y)
    Outputs:
        integer value 
    '''
    mask = rle_decode(rle_mask, shape)
    return np.sum(mask)

def rle_encode(mask):
    '''
    Function: Encodes a mask in run length encoding
    Inputs:
        mask - binary mask (m x n)
    Outputs:
        RLE string 
    '''
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return str(" ".join(map(str, rle.flatten())))

def rle_decode(rle, shape):
    '''
    Function: Decodes an RLE encoded list of space separated
    numbers and returns a binary mask
    Inputs:
        rle: RLE encoded string
        shape: (x,y) tuple representing m x n shape of binary mask
    Outputs:
        m x n binary array
    '''
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask

def PadPredict(img, model, OVERLAP=64): 
    '''
    Function: Run prediction on an image or sub-image regardless of irregular boundaries
    Inputs:
        image (x,y,3) or (x,y)
        model Segmentation ML model
        OVERLAP (default=64) - unit to use for padding, recommend multiple of 64 for most ML models
    Outputs: 
        dictionary: 
            rois (n,1) array of [y1, x1, y2, x2]
            scores (n,1) array
            mask (n,x,y) array
    '''
    
    img_width = img.shape[1]
    img_length = img.shape[0]
    
    add_x = OVERLAP - (img_width % OVERLAP)
    add_y = OVERLAP - (img_length % OVERLAP)
    
    image = cv2.copyMakeBorder(img, 0, add_y, 0, add_x, cv2.BORDER_CONSTANT)
    if(image.ndim==2):
        image = skimage.color.gray2rgb(image).astype(np.uint16)
    results = model.detect([image], verbose=0)
    mask = []
    for i in range(len(results[0]['masks'][0][0])):
        mask.append(np.array(results[0]['masks'][:img_length,:img_width, i]))
    results[0]['masks'] = np.array(mask)
    del results[0]['class_ids']
    results[0]['shape'] = img.shape[:2]
    return results[0]

def f(data):
    '''
    Function: Convert all prediction output into this format to be used in stitching. Returns copy of original
    Inputs:
        data - output of PadPredict
    Outputs:
        dictionary:
            rois - rois (n,1) array of [y1, x1, y2, x2]
            scores- scores (n,1) array
            removeList - Placeholder for LIST of index values 
            xStart - Placeholder for INT value
            xEnd - Placeholder for INT value
            yStart - Placeholder for INT value
            yEnd - Placeholder for INT value
            shape - (x,y) - Tuple for shape of EACH binary mask
            enc_masks - (n,1) array of n binary masks in RLE encoding
            fileName - Placeholder for STR value
    '''
    currData = data
    #currData = copy.deepcopy(data)
    currData['removeList'] = []
    currData['xStart'] = None
    currData['xEnd'] = None
    currData['yStart'] = None
    currData['yEnd'] = None
    currData['enc_masks'] = []
    currData['fileName'] = None
    currData['rois'] = np.asarray(currData['rois'])

    if(currData['masks'].ndim == 3):
        for i in range(len(currData['masks'])):
            currData['enc_masks'].append(str(rle_encode(currData['masks'][i,:,:])))
        
    else:
        currData['enc_masks'] = None
    currData['enc_masks'] = np.asarray(currData['enc_masks'])
        
    
    del currData['masks']
    return currData

#[xStart,xEnd,yStart,yEnd]
def read_json(r1):
    '''
    Function: Reads a saved JSON file into Python environment. Note that for this workflow, the coordinate
    dict values are not modified until after they are loaded into the environment, which is why 
    lines 15-18 are present. 
    Inputs:
        r1 - string of JSON file name
    Outputs:
        Returns the JSON file associated with the saved JSON file in dict format
    '''
    a = r1.split('_')[1:]
    with open(r1, 'r') as fp1:
        contents1 = json.loads(fp1.read())
    return contents1

def stitch_vert(json_t,json_b,OVERLAP=64, MIN_AREA=15, PADDING=3, CHECK_AREA=False):
    '''
    Function: Vertical stitching between two images. Top image is "dominant" (top image 
    preserved in overlapping area, except bottom padding area). Also removes small segmentation to be ignored
    Inputs:
        json_t - f(Input) of TOP image
        json_b - f(Input) of BOTTOM image
        OVERLAP - CONSTANT PARAMETER (default=)
        MIN_AREA - CONSTANT PARAMETER (default=15)
        PADDING - CONSTANT PARAMETER (default=10)
        CHECK_AREA - CONSTANT BOOLEAN (default=True); Skip this to bypass checking segmentation size
    Outputs:
        None
        Saves top and bottom image json files individually as temporary files
    '''
    #json_t = copy.deepcopy(json_t)
    #json_b = copy.deepcopy(json_b)
    #TOP is dominant
    for i in range(len(json_t['rois'])):
        ROI = json_t['rois'][i]
        if(between(ROI[2],json_t['shape'][0]-PADDING, json_t['shape'][0])):
            json_t['removeList'] = np.append(json_t['removeList'],int(i))
        if(CHECK_AREA and area(json_t['enc_masks'][i], json_t['shape']) < MIN_AREA):
            json_t['removeList'] = np.append(json_t['removeList'],int(i))
    for i in range(len(json_b['rois'])):
        ROI = json_b['rois'][i]
        #if(between(ROI[2], 0, OVERLAP-PADDING)):
        if((ROI[2]>=0) & (ROI[2]<(OVERLAP-PADDING-1))): 
            json_b['removeList'] = np.append(json_b['removeList'],int(i))
        if(CHECK_AREA and area(json_b['enc_masks'][i], json_b['shape']) < MIN_AREA):
            json_b['removeList'] = np.append(json_b['removeList'],int(i))
    json_t = removeEntry(json_t)
    json_b = removeEntry(json_b)
    
    newShape = (json_b['yEnd'] - json_t['yStart'],json_b['xEnd'] - json_t['xStart'])

    for i in range(len(json_t['enc_masks'])):
        enc_arr = rle_decode(json_t['enc_masks'][i], json_t['shape'])
        tempArr = np.zeros(newShape)
        #print(tempArr.shape)
        #print(tempArr[:json_t['shape'][0],:json_t['shape'][1]].shape)
        #print(enc_arr.shape)
        #print('tempArr, tempArr_part, enc_arr')
        tempArr[:json_t['shape'][0],:json_t['shape'][1]] += enc_arr
        json_t['enc_masks'][i] = rle_encode(tempArr)

    for i in range(len(json_b['rois'])):
        NEW_ROI = json_b['rois'][i]
        NEW_ROI[0] += (json_b['yStart'] - json_t['yStart'] )
        NEW_ROI[2] += (json_b['yStart'] - json_t['yStart'] )

        json_t['rois'] = np.append(json_t['rois'], [NEW_ROI], axis=0)
        json_t['scores'] = np.append(json_t['scores'], json_b['scores'][i])
        enc_arr = rle_decode(json_b['enc_masks'][i], json_b['shape'])
        tempArr = np.zeros(newShape)
        tempArr[(json_b['yStart']-json_t['yStart']):,:] += enc_arr
        json_t['enc_masks'] = np.append(json_t['enc_masks'],rle_encode(tempArr))

    json_t['shape'] = newShape
    json_t['yEnd'] = json_b['yEnd']
    json_f2 = setName(json_t)
    
    
    with open(json_f2['fileName'], 'w') as fp:
        json.dump(json_f2, fp, sort_keys=True, indent=4, cls=NumpyEncoder)  
    #print("Wrote " + str(json_f2['fileName']))
        
    #print("Vertical Stitching Iteration Complete")
    return json_f2
    
def stitch_horiz(json_l,json_r,OVERLAP=64, MIN_AREA=15, PADDING=3, CHECK_AREA=True):
    #print("Performing Horizontal Stitch on " + str(json_l['fileName']) + " " + str(json_r['fileName']))
    #[TODO] 
    #
    #(1) WRITE JSON WRITER
    #(2) TEST 
    #(3) FORMER AND LATTER PARTS CAN BE SIMPLIFIED
    '''
    Function: Horizontal stitching between two images. Left image is "dominant" (left image 
    preserved in overlapping area, except right-most padding area). Also removes small segmentation to be ignored
    Inputs:
        json_l - f(Input) of LEFT image
        json_r - f(Input) of RIGHT image
        OVERLAP - CONSTANT PARAMETER (default=64)
        MIN_AREA - CONSTANT PARAMETER (default=15)
        PADDING - CONSTANT PARAMETER (default=10)
        CHECK_AREA - CONSTANT BOOLEAN (default=True); Skip this to bypass checking segmentation size
    Outputs:
        None
        Saves combined json file into current directories
    '''

    #LEFT is dominant
    for i in range(len(json_l['rois'])):
        ROI = json_l['rois'][i]
        if(between(ROI[3],json_l['shape'][1]-PADDING, json_l['shape'][1])):  
            json_l['removeList'] = np.append(json_l['removeList'],int(i))
        if(CHECK_AREA and area(json_l['enc_masks'][i], json_l['shape']) < MIN_AREA): 
            json_l['removeList'] = np.append(json_l['removeList'],int(i))
    for i in range(len(json_r['rois'])):
        ROI = json_r['rois'][i]
        #if(between(ROI[3], 0, OVERLAP-PADDING)): 
        if((ROI[3]>=0) & (ROI[3]<(OVERLAP-PADDING-1))): 
            json_r['removeList'] = np.append(json_r['removeList'],int(i))
        if(CHECK_AREA and area(json_r['enc_masks'][i], json_r['shape']) < MIN_AREA):
            json_r['removeList'] = np.append(json_r['removeList'],int(i))
    json_l = removeEntry(json_l)
    json_r = removeEntry(json_r)
    
    ### FORMER AND LATTER PART BOUNDARY ###
    
    newShape = (json_r['yEnd'] - json_l['yStart'],json_r['xEnd'] - json_l['xStart'])

    for i in range(len(json_l['enc_masks'])):
        enc_arr = rle_decode(json_l['enc_masks'][i], json_l['shape'])
        tempArr = np.zeros(newShape)
        tempArr[:json_l['shape'][0],:json_l['shape'][1]] += enc_arr
        json_l['enc_masks'][i] = rle_encode(tempArr)

    for i in range(len(json_r['rois'])):
        NEW_ROI = json_r['rois'][i]
        NEW_ROI[1] += (json_r['xStart'] - json_l['xStart'] )
        NEW_ROI[3] += (json_r['xStart'] - json_l['xStart'] )
        json_l['rois'] = np.append(json_l['rois'], [NEW_ROI],axis=0)
        json_l['scores'] = np.append(json_l['scores'], json_r['scores'][i])
        enc_arr = rle_decode(json_r['enc_masks'][i], json_r['shape'])
        tempArr = np.zeros(newShape)
        tempArr[:json_r['shape'][0],(json_r['xStart']-json_l['xStart']):] += enc_arr
        json_l['enc_masks'] = np.append(json_l['enc_masks'],rle_encode(tempArr))
    json_l['shape'] = newShape
    json_l['xEnd'] = json_r['xEnd']
    json_f1 = setName(json_l)

    with open(json_f1['fileName'], 'w') as fp:
        json.dump(json_f1, fp, sort_keys=True, indent=4, cls=NumpyEncoder)  
    #print("Wrote " + str(json_f1['fileName']))
        
    #print("Horizontal Stitching Iteration Complete")
    return json_f1

def stitch_predict(img, name="output.json", model=False,modelVar=None,check_overlap=False,WINDOW=576, OVERLAP=64, MIN_AREA=15, PADDING=10, POST_NMS_ROIS_INFERENCE=5000,MODEL_PATH=''):
    if(model==False):
        model = init(MODEL_PATH)
    else:
        model = modelVar
    reset()
    if(img.shape[0] < WINDOW + OVERLAP or img.shape[1] < WINDOW + OVERLAP):
        x = f(PadPredict(img, model))
        with open(name, 'w') as fp:
            json.dump(x, fp, sort_keys=True, indent=4, cls=NumpyEncoder)
        return x
    IMG_LENGTH = img.shape[0]
    IMG_WIDTH = img.shape[1]
    
    NUM_WIDTH_ITER = int(math.ceil(IMG_WIDTH/(WINDOW+OVERLAP))) #dj - GLOBAL USE
    NUM_LENGTH_ITER = int(math.ceil(IMG_LENGTH/(WINDOW+OVERLAP))) #di - GLOBAL USE
    #NUM_WIDTH_ITER = int(math.ceil(IMG_WIDTH/((WINDOW-OVERLAP))))
    for i in range(NUM_LENGTH_ITER):
        for j in range(NUM_WIDTH_ITER):
            if(i == 0):
                if(j == 0):
                    subImg = img[0: WINDOW+OVERLAP, 0:WINDOW + OVERLAP]
                    xStart = 0
                    xEnd = WINDOW + OVERLAP
                    yStart = 0
                    yEnd = WINDOW + OVERLAP
                elif(j == NUM_WIDTH_ITER - 1):
                    subImg = img[0: WINDOW+OVERLAP, j*WINDOW: img.shape[1]]
                    xStart = 0
                    xEnd = WINDOW + OVERLAP
                    yStart = j*WINDOW
                    yEnd = img.shape[1]
                else:
                    subImg = img[0: WINDOW+OVERLAP, j*WINDOW: (j+1)*WINDOW+OVERLAP]
                    xStart = 0
                    xEnd = WINDOW + OVERLAP
                    yStart = j*WINDOW
                    yEnd = (j+1)*WINDOW+OVERLAP
            elif(j == 0):
                if(i == NUM_LENGTH_ITER-1):
                    subImg = img[i*WINDOW:img.shape[0], 0:WINDOW+OVERLAP]
                    xStart = i*WINDOW
                    xEnd = img.shape[0]
                    yStart = 0
                    yEnd = WINDOW+OVERLAP
                else:
                    subImg = img[i*WINDOW: (i+1)*WINDOW+OVERLAP,0:WINDOW + OVERLAP]
                    xStart = i*WINDOW
                    xEnd = (i+1)*WINDOW+OVERLAP
                    yStart = 0
                    yEnd = WINDOW+OVERLAP
            elif(i == NUM_LENGTH_ITER - 1):
                if(j == NUM_WIDTH_ITER - 1):
                    subImg = img[i*WINDOW: img.shape[0], j*WINDOW: img.shape[1]]
                    xStart = i*WINDOW
                    xEnd = img.shape[0]
                    yStart = j*WINDOW
                    yEnd = img.shape[1]
                else: 
                    subImg = img[i*WINDOW: img.shape[0], j*WINDOW: (j+1)*WINDOW+OVERLAP]
                    xStart = i*WINDOW
                    xEnd = img.shape[0]
                    yStart = j*WINDOW
                    yEnd = (j+1)*WINDOW+OVERLAP
            elif(j == NUM_WIDTH_ITER - 1):
                subImg = img[i*WINDOW: (i+1)*WINDOW+OVERLAP, j*WINDOW: img.shape[1]]
                xStart = i*WINDOW
                xEnd = (i+1)*WINDOW+OVERLAP
                yStart = j*WINDOW
                yEnd = img.shape[1]
                
            else:
                subImg = img[i*WINDOW: (i+1)*WINDOW+OVERLAP, j*WINDOW: (j+1)*WINDOW+OVERLAP]
                xStart = i*WINDOW
                xEnd = (i+1)*WINDOW+OVERLAP
                yStart = j*WINDOW
                yEnd = (j+1)*WINDOW+OVERLAP
            
            if(check_overlap):
                output = remove_overlap(f(PadPredict(subImg, model, OVERLAP=OVERLAP)))
            else:
                output = f(PadPredict(subImg, model, OVERLAP=OVERLAP))
            output['yStart'] = xStart
            output['yEnd'] = xEnd
            output['xStart'] = yStart
            output['xEnd'] = yEnd
            
            output = setName(output, i, j)
            #print(output['fileName'])
            #print(output['shape'])
            with open(output['fileName'], 'w') as fp:
                json.dump(output, fp, sort_keys=True, indent=4, cls=NumpyEncoder)
                
            I = i
            J = j

    A = sorted(glob.glob(".*.json"))
    totalStitch = None
    for j in range(J+1):
        for i in range(I):
            #print(i)
            if(i==0):
                #print(j,i)
                tempStitch = stitch_horiz(read_json(A[j*(J+1)+i]),read_json(A[j*(J+1)+i+1]), OVERLAP=OVERLAP, MIN_AREA=MIN_AREA,PADDING=PADDING)
            elif(i > 0):
                #print(j,i)
                tempStitch = stitch_horiz(tempStitch, read_json(A[j*(J+1)+i+1]), OVERLAP=OVERLAP, MIN_AREA=MIN_AREA,PADDING=PADDING)
        if(j == 0):
            totalStitch = tempStitch
        else:
            #return totalStitch, tempStitch
            totalStitch = stitch_vert(totalStitch, tempStitch, OVERLAP=OVERLAP, MIN_AREA=MIN_AREA,PADDING=PADDING)
        
    with open(name, 'w') as fp:
        json.dump(totalStitch, fp, sort_keys=True, indent=4, cls=NumpyEncoder)
        
    if(check_overlap):
        order = np.argsort(totalStitch['scores'])[::-1] + 1 
        masks = decode_masks(totalStitch)
        mask = np.max(masks * np.reshape(order, [1, 1, -1]), -1)
        imgName = name[:-4] + 'tiff'
        si.imsave(imgName, mask)
        
    reset()
    #print("Done!")
    return totalStitch


def normal_stitch(img, name="output.json", model=False,modelVar=None,check_overlap=False,WINDOW=576, OVERLAP=64, MIN_AREA=15, PADDING=10, POST_NMS_ROIS_INFERENCE=5000,MODEL_PATH=''):
    IMG_WIDTH = img.shape[1]
    IMG_LENGTH = img.shape[0]
    NUM_WIDTH_ITER = int(math.ceil(IMG_WIDTH/(WINDOW+OVERLAP))) #dj - GLOBAL USE
    NUM_LENGTH_ITER = int(math.ceil(IMG_LENGTH/(WINDOW+OVERLAP))) #di - GLOBAL USE
    output = np.zeros((IMG_LENGTH, IMG_WIDTH))
    for i in range(NUM_LENGTH_ITER):
        for j in range(NUM_WIDTH_ITER):
            if(i == 0):
                if(j == 0):
                    subImg = img[0: WINDOW+OVERLAP, 0:WINDOW + OVERLAP]
                    xStart = 0
                    xEnd = WINDOW + OVERLAP
                    yStart = 0
                    yEnd = WINDOW + OVERLAP
                elif(j == NUM_WIDTH_ITER - 1):
                    subImg = img[0: WINDOW+OVERLAP, j*WINDOW: img.shape[1]]
                    xStart = 0
                    xEnd = WINDOW + OVERLAP
                    yStart = j*WINDOW
                    yEnd = img.shape[1]
                else:
                    subImg = img[0: WINDOW+OVERLAP, j*WINDOW: (j+1)*WINDOW+OVERLAP]
                    xStart = 0
                    xEnd = WINDOW + OVERLAP
                    yStart = j*WINDOW
                    yEnd = (j+1)*WINDOW+OVERLAP
            elif(j == 0):
                if(i == NUM_LENGTH_ITER-1):
                    subImg = img[i*WINDOW:img.shape[0], 0:WINDOW+OVERLAP]
                    xStart = i*WINDOW
                    xEnd = img.shape[0]
                    yStart = 0
                    yEnd = WINDOW+OVERLAP
                else:
                    subImg = img[i*WINDOW: (i+1)*WINDOW+OVERLAP,0:WINDOW + OVERLAP]
                    xStart = i*WINDOW
                    xEnd = (i+1)*WINDOW+OVERLAP
                    yStart = 0
                    yEnd = WINDOW+OVERLAP
            elif(i == NUM_LENGTH_ITER - 1):
                if(j == NUM_WIDTH_ITER - 1):
                    subImg = img[i*WINDOW: img.shape[0], j*WINDOW: img.shape[1]]
                    xStart = i*WINDOW
                    xEnd = img.shape[0]
                    yStart = j*WINDOW
                    yEnd = img.shape[1]
                else: 
                    subImg = img[i*WINDOW: img.shape[0], j*WINDOW: (j+1)*WINDOW+OVERLAP]
                    xStart = i*WINDOW
                    xEnd = img.shape[0]
                    yStart = j*WINDOW
                    yEnd = (j+1)*WINDOW+OVERLAP
            elif(j == NUM_WIDTH_ITER - 1):
                subImg = img[i*WINDOW: (i+1)*WINDOW+OVERLAP, j*WINDOW: img.shape[1]]
                xStart = i*WINDOW
                xEnd = (i+1)*WINDOW+OVERLAP
                yStart = j*WINDOW
                yEnd = img.shape[1]
                
            else:
                subImg = img[i*WINDOW: (i+1)*WINDOW+OVERLAP, j*WINDOW: (j+1)*WINDOW+OVERLAP]
                xStart = i*WINDOW
                xEnd = (i+1)*WINDOW+OVERLAP
                yStart = j*WINDOW
                yEnd = (j+1)*WINDOW+OVERLAP
                
            # if(check_overlap):
            #     output = remove_overlap(f(PadPredict(subImg, model, OVERLAP=OVERLAP)))
            # else:
            #     output = f(PadPredict(subImg, model, OVERLAP=OVERLAP))
            # output['yStart'] = xStart
            # output['yEnd'] = xEnd
            # output['xStart'] = yStart
            # output['xEnd'] = yEnd
            
            # output = setName(output, i, j)
            # #print(output['fileName'])
            # #print(output['shape'])
            # if 
            # with open(output['fileName'], 'w') as fp:
            #     json.dump(output, fp, sort_keys=True, indent=4, cls=NumpyEncoder)
                
            # I = i
            # J = j

            output[xStart:xEnd, yStart:yEnd] = overlay(f(PadPredict(subImg, model, OVERLAP=OVERLAP)))

    reset()        
    return output
