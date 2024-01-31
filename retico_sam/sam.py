"""
Segment Anything Module
=======================

This module provides ability to segment anything within an image and 
detect all different objects within the image. 
"""
import gc
from collections import deque
from datetime import datetime
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import threading 
import time 
import torch
from matplotlib import pyplot as plt
# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator

import retico_core
import sys

#prefix = '../../'
#sys.path.append(prefix+'retico_vision')



from retico_vision.vision import ImageIU, DetectedObjectsIU


class ExtractType(Enum):
    bb = 'bounding box'
    seg = 'segment'


class SAMModule(retico_core.AbstractModule):
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
    
    # MODEL_OPTIONS = {
    #     "h": "vit_h",
    #     "l": "vit_l",
    #     "b": "vit_b",
    # }

    MODEL_OPTIONS = {
        "h": "vit_h",
        "l": "vit_l",
        "b": "vit_b",
        "t": "vit_t",
    }

    def __init__(self, model=None, path_to_chkpnt=None, extract_type: ExtractType=None, **kwargs):
        """
        Initialize the SAM Object Detection Module
        Args:
            model (str): the name of the SAM model
                will correspond to the model checkpoint
        """
        super().__init__(**kwargs)

        if model and model.lower() in self.MODEL_OPTIONS:
            model = self.MODEL_OPTIONS[model.lower()]
            print(f"Using {model}. Make sure you have the corresponding checkpoint being passed in.")
        else: 
            print("Unknown model option. Defaulting to h (VIT-H) SAM model.")
            print("Other options include 'l' for vit_l and 'b' for vit_b.")
            model = "vit_h"

        if (path_to_chkpnt==None):
            print("Path to checkpoint matching model type must be passed in.")
            exit()

        cuda_available = torch.cuda.is_available()
      
        self.model = sam_model_registry[model](checkpoint=path_to_chkpnt) #Only worked if passed in otherwise struggled to find path
        if (cuda_available):
            device = "cuda"
            self.model.to(device=device)
        self.queue = deque(maxlen=1)
        self.extract_type = extract_type

    def process_update(self, update_message):
        for iu, ut in update_message:
            if ut != retico_core.UpdateType.ADD:
                continue
            else:
                self.queue.append(iu)
    #
    # def show_mask(self, mask, ax, random_color=False):
    #     if random_color:
    #         color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    #     else:
    #         color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    #     h, w = mask.shape[-2:]
    #     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    #     ax.imshow(mask_image)
    #     del mask
    #     gc.collect()
    #
    # def show_masks_on_image(self, raw_image, masks):
    #     plt.clf()
    #
    #     fig = plt.figure()
    #     plt.imshow(np.array(raw_image))
    #     ax = fig.gca()
    #     ax.set_autoscale_on(False)
    #     for mask in masks:
    #         self.show_mask(mask, ax=ax, random_color=True)
    #     plt.axis("off")
    #     # plt.show()
    #     plt.savefig(f'sam_masks/sam_mask_{datetime.now().strftime("%m-%d_%H-%M-%S")}.png')
    #     del mask
    #     gc.collect()

    # def show_mask(self, mask, ax, random_color=False):
    #     if random_color:
    #         color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    #     else:
    #         color = np.array([30/255, 144/255, 255/255, 0.6])
    #     h, w = mask.shape[-2:]
    #     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    #     ax.imshow(mask_image)
    #     plt.savefig(f'sam_masks/sam_mask_{datetime.now().strftime("%m-%d_%H-%M-%S")}.png')


    def show_points(self, coords, labels, ax, marker_size=375):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

    def show_box(self, box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))


    def show_anns(self, anns):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:,:,3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        ax.imshow(img)



    def _detector_thread(self):
        while self._detector_thread_active:
            time.sleep(2)
            if len(self.queue) == 0:
                time.sleep(0.5) # original(0.5) ~ change this for more time between segmentation of each image
                continue
            
            input_iu = self.queue.popleft()
            image = input_iu.payload

            sam_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
            # cv2.imwrite('./test_sam_image.jpg', sam_image)

            mask_generator = SamAutomaticMaskGenerator(
                model= self.model,
                points_per_side=32,
                pred_iou_thresh=0.999,
                stability_score_thresh=0.96,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=400
            )
            print(f"Generating SAM mask")
            start = time.time()
            masks_generated = mask_generator.generate(sam_image)
            end = time.time()
            print(f"[time elapsed: {end - start}] {masks_generated[0].keys()}")
            # self.show_masks_on_image(sam_image, masks_generated)
            plt.figure(figsize=(20,20))
            plt.imshow(image)
            self.show_anns(masks_generated)
            plt.axis('off')

            if self.extract_type is ExtractType.bb:
                valid_extractions = []
                for box_num in range(len(masks_generated)):
                    valid_extractions.append(masks_generated[box_num]['bbox']) #mask bounding box in XYWH format


            elif self.extract_type is ExtractType.seg:
                valid_extractions = []
                for seg_num in range(len(masks_generated)):
                    valid_extractions.append(masks_generated[seg_num]['segmentation'])

            if len(valid_extractions) == 0:
                path = Path(f"./no_{self.extract_type.value}/{input_iu.execution_uuid}")
                path.mkdir(parents=True, exist_ok=True)
                file_name = f"{input_iu.flow_uuid}.png" # TODO: png or jpg better?
                imwrite_path = f"{str(path)}/{file_name}"
                plt.savefig(imwrite_path)
                plt.close()
                continue

            path = Path(f"./no_seg/{input_iu.execution_uuid}")
            path.mkdir(parents=True, exist_ok=True)
            file_name = f"{input_iu.flow_uuid}.png" # TODO: png or jpg better?
            imwrite_path = f"{str(path)}/{file_name}"
            plt.savefig(imwrite_path)
            plt.close()

            output_iu = self.create_iu(input_iu)

            output_iu.set_detected_objects(image, valid_extractions, self.extract_type.value)
            output_iu.set_flow_uuid(input_iu.flow_uuid)
            output_iu.set_motor_action(input_iu.motor_action)
            um = retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
            self.append(um)

    def prepare_run(self):
        self._detector_thread_active = True
        threading.Thread(target=self._detector_thread).start()

    def shutdown(self):
        self._detector_thread_active = False
