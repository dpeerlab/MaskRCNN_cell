from imgaug import augmenters as iaa
import imageio
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn.model import log
from mrcnn.config import Config
import scipy
import numpy as np


class CellDataset(utils.Dataset):    
    
    #self.images = []
    #self.labels = []
    #self.size = 0
    
    def setImages(self,x):
        self.images = x
        
    def setLabels(self,x):
        self.labels = x
        
    def setSize(self,x):
        self.size = x
        
    def load_cells(self, path):
        # Add classes
        images = self.images
        labels = self.labels
        size = self.size
        self.add_class("cell", 1, "cell")
        img_id = 0
        for image in images:
            self.add_image("cell", image_id=img_id, path=image, x_offset=0, y_offset=0)
            img_id += 1

    def load_image(self, image_id):
        images = self.images
        labels = self.labels
        size = self.size
        info = self.image_info[image_id]
        raw_image = imageio.imread(info['path'])
        x_offset = info['x_offset']
        y_offset = info['y_offset']
        if len(raw_image.shape)==3:
            image = raw_image[x_offset:x_offset+size,y_offset:y_offset+size,:3]
        else:
            image = np.zeros((size,size,3))
            for i in range(3):
                image[:,:,i] = raw_image[x_offset:x_offset+size,y_offset:y_offset+size]
        return image
        
    def image_reference(self, image_id):
        images = self.images
        labels = self.labels
        size = self.size
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "nucleus":
            return info["nucleus"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        images = self.images
        labels = self.labels
        size = self.size
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        mask = (imageio.imread(labels[image_id]))
    
        x_offset = info['x_offset']
        y_offset = info['y_offset']
    
        trimmed_mask = mask[x_offset:x_offset+size,y_offset:y_offset+size]
        
        labeled_mask, num_cells = scipy.ndimage.label(trimmed_mask)
        multi_dim_mask = np.zeros([size,size,num_cells])
    
        count=0
        for i in range(1,num_cells+1):
            mask = (labeled_mask == i).astype(int)
            #print(mask.shape)
            if np.sum(mask)>25:
                multi_dim_mask[:,:,i-1] = (labeled_mask == i).astype(int)
                count+=1
        multi_dim_mask = multi_dim_mask[:,:,:count]
        class_ids = np.ones(count).astype(int)

        return multi_dim_mask, class_ids
