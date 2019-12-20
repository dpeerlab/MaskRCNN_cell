# Mask R-CNN for single-cell segmentation 

This implementation of Mask R-CNN is designed for single-cell instance segmentation in the context of multiplexed tissue imaging. The model generates bounding boxes and segmentation masks for each instance of an object in the image. The code is based on the implementation of Mask R-CNN by [Matterport](https://github.com/matterport/Mask_RCNN) in Python 3, Keras, and TensorFlow. The purpose of this Mask R-CNN implementation is to wrap its functions for ease of use and to provide a more streamlined pipeline for single-cell instance segmentation using Mask R-CNN.

Features:

* Optimized parameters for segmenting cells in multiplexed tissue imaging  

* Solution for segmenting batch and large images

* Support for training on custom datasets 

  

## Quick Start

### Installation 

* Set up programming environment and necessary dependencies

  * Set up conda for python environment management and cuda for GPU environment

  * Clone this repository 

  * Create a new conda environment

  * ```bash
    conda env create -n maskrcnn_cell -f environment.yml
    ```

  * Install pycocotools

    ```bash
    pip install git+git://github.com/tryolabs/cocoapi.git#subdirectory=PythonAPI
    ```

* Install Mask R-CNN 

  * Run setup from the repository root directory

  * ```bash
    python setup.py install
    ```

### Usage

* Train on custom dataset. Example: [train_custom_data.ipynb]. 

  * Load data

  * Data augmentation

  * Adjust training parameters. Current parameters are optimized for 200X images

  * Set model backend and pretained weights

  * Training. Training progress can be viewed using `tensorboard --logdir logs/`

  * Tip:

    * GPU is highly recommended for training 

    

* Predict segmentation on images using pre-trained model [predict_segmentation.ipynb] 
  * Load target images
  * Adjust prediction parameters
  * Predict segmentation on example image
  * Batch output. The model allows input of a list of images, and it will generate masks in JSON format together with the original image overlaid with the mask
  * Prediction on large image via stitching function. The model's memory usage depends on the maximum target number set in the prediction parameter and on the size of input image. For example, an input image of 1024x1024 pixels with 3500 max prediction could take 8 GB of memory. 
  * Tip: 
    * CPU or GPU? Prediction can be performed in a CPU environment, and since most CPU setups provide more memory than GPUs, it is actually cheaper and more stable to perform prediction in a CPU environment. 



## Model Architecture



![img](./resource/figure/maskrcnn_framework.png)



## Acknowledgement 

This work is supported by the [Parker Institute for Cancer Immunotherapy](https://www.parkerici.org/)



## Reference

He, K., G. Gkioxari, P. Dollár, and R. Girshick. 2017. “Mask R-CNN.” In *2017 IEEE International Conference on Computer Vision (ICCV)*, 2980–88. [link](https://arxiv.org/abs/1703.06870)

https://github.com/matterport/Mask_RCNN 

https://medium.com/@jonathan_hui/image-segmentation-with-mask-r-cnn-ebe6d793272

