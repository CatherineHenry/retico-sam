"""
Segment Anything Module
=======================

This module provides ability to segment anything within an image and 
detect all different objects within the image. 
"""
import base64
from collections import deque
from datetime import datetime

import numpy as np
import threading 
import time 
import torch
from transformers import pipeline
import supervision as sv
import retico_core
import gc
import sys
# import cv2

import numpy as np
import matplotlib.pyplot as plt
import gc

#prefix = '../../'
#sys.path.append(prefix+'retico_vision')


from retico_vision.vision import ImageIU, DetectedObjectsIU

class SAMModule(retico_core.AbstractModule):
    @staticmethod
    def name():
        return "HuggingFace SAM Object Detection Module"
    
    @staticmethod
    def description():
        return "An object detection moduel using SAM."
    
    @staticmethod
    def input_ius():
        return [ImageIU]
    
    @staticmethod
    def output_iu():
        return DetectedObjectsIU
  

    def __init__(self, model="facebook/sam-vit-base", show=False, use_bbox=False, use_seg=False, **kwargs):
        """
        Initialize the SAM Object Detection Module
        Args:
            model (str): the name of the SAM model
                will correspond to the model checkpoint
        """
        super().__init__(**kwargs)

      
        self.generator = pipeline("mask-generation", model=model, device="cuda")
        self.queue = deque(maxlen=1)
        self.use_bbox = use_bbox
        self.use_seg = use_seg
        self.show = show

    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
        del mask
        gc.collect()

    def show_masks_on_image(self, raw_image, masks):
        plt.imshow(np.array(raw_image))
        ax = plt.gca()
        ax.set_autoscale_on(False)
        for mask in masks:
            self.show_mask(mask, ax=ax, random_color=True)
        plt.axis("off")
        plt.show()
        plt.savefig(f'sam_masks/sam_mask_{datetime.now().strftime("%m-%d_%H-%M-%S")}.png')
        del mask
        gc.collect()

    def process_update(self, update_message):
        for iu, ut in update_message:
            if ut != retico_core.UpdateType.ADD:
                continue
            else:
                self.queue.append(iu)

    def _detector_thread(self):
         
        while self._detector_thread_active:
            time.sleep(2)
            if len(self.queue) == 0:
                time.sleep(0.5) # original(0.5) ~ change this for more time between segmentation of each image
                continue
            
            input_iu = self.queue.popleft()
            image = input_iu.payload 

            # sam_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
            # cv2.imwrite('test_image.png', sam_image)
            print(f"Generating SAM mask")
            start = time.time()
            outputs = self.generator(image, points_per_batch=64)
            end = time.time()
            print(f"[time elapsed: {end - start}]")
            masks = np.array(outputs["masks"])
            self.show_masks_on_image(image, masks)

            if len(masks) == 0: continue

            output_iu = self.create_iu(input_iu)
            if self.use_bbox:
                bbox = sv.mask_to_xyxy(masks)
                bytes = image.tobytes()
                output_iu.set_detected_objects(base64.b64encode(bytes).decode(), bbox.tolist(), "bb")
            elif self.use_seg:
                bytes = image.tobytes()
                output_iu.set_detected_objects(base64.b64encode(bytes).decode(), masks.tolist(), "seg")
            um = retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
            self.append(um)

    def prepare_run(self):
        self._detector_thread_active = True
        threading.Thread(target=self._detector_thread).start()

    def shutdown(self):
        self._detector_thread_active = False
