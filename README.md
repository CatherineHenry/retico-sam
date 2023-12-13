# retico_sam
A ReTico module for SAM. See below for more information on the models.

As suggested in [SAM](https://github.com/facebookresearch/segment-anything) it is highly recommended to use SAM with a GPU, but CPU only is possible, it will just take much longer.

### Installation and requirements
Most requirements are within the 'requirements.txt'.
The most basic ones needed are:
* python >= 3.8
* pytorch >= 1.7
* torchvision >= 0.8

Once you have an environment that matches all these requirements you can run the following command to install SAM 
* pip install git+https://github.com/facebookresearch/segment-anything.git

As well, it is required you install a checkpoint for whichever model version you would like to use. The default model is 'vit_h' which can be found here: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth. This will start a download for that checkpoint. For use you must include the path to the checkpoint when creating the sam module, as demonstrated in the example runner file. 

If you would like to use a different model you must download the corresponding checkpoint and update the path, as well as, running the python command with the matching model name i.e. python runner_sam.py vit_l

You will also need to have retico modules on your local machine, specifically ensure you have retico_vision and retico_core. As seen in the example these are saved in the os.environ.

#### Important Parameters
The SAM module can take the following parameters: model, path_to_chkpnt, use_bbox, use_seg
* model = model type, the user can enter either 'h', 'l', or 'b' NOTE: the user must match the model type with the corresponding checkpoint
* path_to_chkpnt = the path to the model checkpoint that was downloaded by the user; it is suggested to store the checkpoint file in the same directory as the runner file 
* use_bbox = boolean value to use bounding box segmentation; default to false 
* use_seg = boolean value to use segmentation mask; default to false 

The Extract Objects module allows the user to determine how many images to plot to be displayed
* num_obj_to_display = number of extracted segmentation objects to display in the plot that will be saved to a folder named 'extracted_objects' NOTE: if the folder doesn't exist it will be created, otherwise it will just store the new image in the folder

### Example
import sys, os
from retico import *

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'  
os.environ['RETICO'] = 'retico_core'  
os.environ['SAM'] = 'retico_sam'  
os.environ['VISION'] = 'retico_vision'

sys.path.append(os.environ['RETICO'])  
sys.path.append(os.environ['SAM'])  
sys.path.append(os.environ['VISION'])

from retico_core.debug import DebugModule   
from retico_vision.vision import WebcamModule  
from retico_vision.vision import ExtractObjectsModule   
from retico_sam.sam import SAMModule  
  
path_var = *** REPLACE WITH PATH TO YOUR CHECKPOINT ***


webcam = WebcamModule()  
sam = SAMModule(model='h', path_to_chkpnt=path_var, use_bbox=True)  
extractor = ExtractObjectsModule(num_obj_to_display=20)  
debug = DebugModule()  

webcam.subscribe(sam)  
sam.subscribe(extractor)  
extractor.subscribe(debug)    

webcam.run()  
sam.run()  
extractor.run()  
debug.run()  

print("Network is running")  
input()  

webcam.stop()  
sam.stop()  
extractor.stop()   
debug.stop()  
