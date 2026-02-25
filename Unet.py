import colorsys
import copy
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from nets.unet import Unet as UnetNetwork
from utils.utils import cvtColor, preprocess_input, resize_image

class UnetInference(object):
    """
    U-Net Inference class for extracting ice layers and bedrock interfaces.
    """
    _defaults = {
        # Path to the trained weights file
        "model_path"    : 'weights/best_epoch_weights.pth',
        # Number of classes: background + target features
        "num_classes"   : 2,
        # Backbone network used
        "backbone"      : "vgg",
        # Input image size
        "input_shape"   : [512, 512],
        # Visualization type: 0: Blend with original; 1: Mask only; 2: Background removed
        "mix_type"      : 1,
        # Use GPU for inference
        "cuda"          : True,
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        
        # Color palette for visualization
        self.colors = [(255, 255, 255), (0, 128, 0), (0, 0, 128)] 
        
        self.load_model()

    def load_model(self):
        """ Initialize the network and load weights. """
        self.net = UnetNetwork(num_classes=self.num_classes, backbone=self.backbone)
        device = torch.device('cuda' if torch.cuda.is_available() and self.cuda else 'cpu')
        
        # Load weights
        state_dict = torch.load(self.model_path, map_location=device)
        self.net.load_state_dict(state_dict)
        self.net = self.net.eval()
        
        if self.cuda:
            self.net = nn.DataParallel(self.net).cuda()
        print(f'Model loaded from {self.model_path}')

    def detect_image(self, image):
        """ Perform inference on a single image. """
        # Ensure image is in RGB format
        image = cvtColor(image)
        old_img = copy.deepcopy(image)
        original_h, original_w = np.array(image).shape[0], np.array(image).shape[1]

        # Resize image without distortion (add gray bars)
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        
        # Preprocessing: convert to tensor and add batch dimension
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
                
            # Forward pass
            pr = self.net(images)[0]
            
            # Get probability map and remove padding (gray bars)
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            
            # Resize back to original dimensions
            pr = cv2.resize(pr, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1)

        # Post-inference visualization based on mix_type
        if self.mix_type == 0:
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [original_h, original_w, -1])
            image = Image.fromarray(np.uint8(seg_img))
            image = Image.blend(old_img, image, 0.3) # Blend predicted mask with raw image
        elif self.mix_type == 1:
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [original_h, original_w, -1])
            image = Image.fromarray(np.uint8(seg_img))
        elif self.mix_type == 2:
            seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')
            image = Image.fromarray(np.uint8(seg_img))
        
        return image

    def get_FPS(self, image, test_interval=100):
        """ Measure inference speed (FPS). """
        image = cvtColor(image)
        image_data, _, _ = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda: images = images.cuda()
            
            t1 = time.time()
            for _ in range(test_interval):
                self.net(images)
            t2 = time.time()
            
        return (t2 - t1) / test_interval
