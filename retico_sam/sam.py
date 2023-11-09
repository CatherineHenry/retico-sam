"""
Segment Anything Module
=======================

This module provides ability to segment anything within an image and 
detect all different objects within the image. 
"""

from collections import deque
import cv2
import numpy as np
import threading 
import time 
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
#could same path as retico_yolo and use ultralytics 
#from ultralytics import SAM

import retico_core
import sys

prefix = '../../'
sys.path.append(prefix+'retico-vision')

from retico_vision.vision import ImageIU, DetectedObjectsIU

class SAM(retico_core.AbstractModule):
    @staticmethod
    def name():
        return "SAM Object Detection Module"
    
    @staticmethod
    def description():
        return "An object detection moduel using SAM."
    
    @staticmethod
    def input_ius():
        return [ImageIU]
    
    @staticmethod
    def output_iu():
        return DetectedObjectsIU
    
    MODEL_OPTIONS = {
        "vit_h": "../checkpoint/sam_vit_h_4b8939.pth",
        "vit_l": "../checkpoint/sam_vit_l_0b3195.pth",
        "vit_b": "../checkpoint/sam_vit_b_01ec64.pth",
    }

    #if decided that using alytics like YOLO did is better, use this
    # MODEL_OPTIONS = {
    #     "b": "sam_b.pt",
    #     "l": "sam_l.pt",
    # }


    def __init__(self, model_type=None, **kwargs):
        """
        Initialize the SAM Object Detection Module
        Args:
            model (str): the name of the SAM model
                will correspond to the model checkpoint
        """
        super().__init__(**kwargs)

        if model_type not in self.MODEL_OPTIONS.keys():
            print("Unknown model option. Defaulting to VIT-H SAM model.")
            #print("Unknown model option. Defaulting to b (SAM base).")
            print("Other options include 'vit_l' and 'vit_b'.")
            #print("Other options include 'l'.")
            #print("See https://docs.ultralytics.com/models/sam/#key-features-of-the-segment-anything-model-sam for mroe info.")
            model_type = "vit_h"

        #device = "cuda"

        self.model = sam_model_registry[model_type](self.MODEL_OPTIONS.get(model_type))
        #self.model.to(device)
        self.queue = deque(maxlen=1)


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
                time.sleep(0.5)
                continue
            
            input_iu = self.queue.popleft()
            image = input_iu.payload 

            mask_generator = SamAutomaticMaskGenerator(
                model= self.model,
                points_per_side=32,
                pred_iou_thresh=0.9,
                stability_score_thresh=0.92,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=400
            )

            masks_generated = mask_generator.generate(image)

            print(masks_generated[0].keys())

            valid_boxes = []
            for mask in masks_generated:
                valid_boxes.append(masks_generated[mask]['bbox'].cpu().numpy()) #mask bounding box in XYWH format 

            print(valid_boxes)

            if len(valid_boxes) == 0: continue

            output_iu = self.create_iu(input_iu)
            output_iu.set_detected_objects(image, valid_boxes)
            um = retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
            self.append(um)



    def prepare_run(self):
        self._detector_thread_active = True
        threading.Thread(target=self._detector_thread).start()

    def shutdown(self):
        self._detector_thread_active = False