"""LUMEN: an agentic proof-of-life vision system built on OpenCV 5."""
import cv2 as _cv

__version__ = "0.2.0"

# Hide OpenCV's harmless "Targets are not supported by the new graph engine" warning.
_cv.utils.logging.setLogLevel(_cv.utils.logging.LOG_LEVEL_ERROR)
