from imgaug import augmenters as iaa
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn.model import log
from mrcnn.config import Config
import numpy as np

class CellConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "cell"
    BACKBONE = 'resnet50'
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    #RPN_NMS_THRESHOLD = 0.8
    DETECTION_MAX_INSTANCES = 256 #500
    MAX_GT_INSTANCES = 256
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 cell

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128 #128
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256 #1024 #256 #1024 #256
    IMAGE_RESIZE_MODE = 'crop'
    ROI_POSITIVE_RATIO = 0.33

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 320 #1000 #320 #1000 #320 # 80
    #LOSS_WEIGHTS  ={}
    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1000 #100 #1000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 10# 10 #5 #50
    
    MEAN_PIXEL = np.array([0,0,0])

    # ALSO CHANGED CONFIG.PY to change IMAGE SHAPE
    
class InferenceConfig(CellConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_RESIZE_MODE = "pad64" # 'none' #
    DETECTION_MAX_INSTANCES = 1000 #3000
    DETECTION_MIN_CONFIDENCE = 0.7
    DETECTION_NMS_THRESHOLD = 0.2
    ROI_POSITIVE_RATIO = 0.8
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    #MEAN_PIXEL = np.array([40,15,30])
    
    POST_NMS_ROIS_INFERENCE=5000 #15000
    