# Mask R-CNN for single cell segmentation 

This implementation of Mask R-CNN is designed for single cell instance segmentation in multiplexed tissue imaging context. The model generates bounding boxes and segmentation masks for each instance of an object in the image. The codes are based on implementation of Mask R-CNN by [Matterport](https://github.com/matterport/Mask_RCNN) on Python 3, Keras, and TensorFlow. The purpose of this implementation of Mask_RCNN is to wrap its functions to be easier to use and name it a more streamlined pipeline for those who want to use Mask_RCNN for single cell instance segmentation.

Features:

* Optimized parameters for segmenting cells in multiplexed tissue imaging  

* Solution for segmenting batch and large images

* Support training on custom dataset 

  

## Quick Start

### Installation 

* Set up programming environment and necessary dependencies

  * Set up conda for python environment management and cuda for GPU environment.

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

  * Load data. 

  * Data augmentation. 

  * Adjust training parameter. Current parameters are optimized for 200X images.

  * Backend and pertained model parameters. 

  * Training.  Progress of training can be viewed with `tensorboard --logdir logs/`

  * Tip:

    * GPU is highly recommend for training process. 

    

* Predict segmentation on images with pre-trained model [predict_segmentation.ipynb] 
  * Load target images.
  * Adjust prediction parameters. 
  * Predict on example image. 
  * Batch output. The model allows input of a list of images and it will generate masks in Json format together with the original image overlapped with the mask.
  * Prediction on large image via stitching function. The memory usage of the model depends on the maximum target number set in the prediction parameter and the size of input image. E.g., input image of 1024x1024 pixel with 3500 max prediction could take 8 G memory. 
  * Tip: 
    * CPU or GPU? Prediction can be done under CPU environment. And since for most services and computers, there are much more memory in CPU than GPU. It is actually cheaper and more stable to do prediction in CPU environment. 



## Model Architecture



![img](./resource/figure/maskrcnn_framework.png)



## Acknowledgement 

This work is supported by [Parker Institute for Cancer Immunotherapy](https://www.parkerici.org/)



## Reference

He, K., G. Gkioxari, P. Dollár, and R. Girshick. 2017. “Mask R-CNN.” In *2017 IEEE International Conference on Computer Vision (ICCV)*, 2980–88. [link](https://arxiv.org/abs/1703.06870)

https://github.com/matterport/Mask_RCNN 

https://medium.com/@jonathan_hui/image-segmentation-with-mask-r-cnn-ebe6d793272

