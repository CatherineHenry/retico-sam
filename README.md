# retico_sam
A ReTico module for SAM. See below for more information on the models.

### Installation and requirements

### Example
```python
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
from retico_sam.sam import SAMModule

path_var = "C:/Users/Drew/CS_SDS/sam_vit_h_4b8939.pth"


webcam = WebcamModule()
sam = SAMModule(path_to_chkpnt=path_var)
debug = DebugModule()

webcam.subscribe(sam)
sam.subscribe(debug)

webcam.setup()

webcam.run()
sam.run()
debug.run()

print("Network is running")
input()

webcam.shutdown()

webcam.stop()
sam.stop()
debug.stop()
```
