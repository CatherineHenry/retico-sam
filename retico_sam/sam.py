"""
Segment Anything Module
=======================

This module provides on-device object detection capabilites by using SAM.
"""

from collections import deque
import cv2
import numpy as np
import threading 
import time 
from ultralytics import SAM