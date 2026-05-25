import numpy as np
import cpp_detect

class Detect:
    
    stride = [32]  # strides computed during build
    
    def __init__(self, nc=1, anchors=()):
        self.nc = nc  # number of classes
        self.nl = len(anchors)  # number of detection layers
        self.anchors = np.array(anchors, dtype=np.float32).reshape(self.nl, -1, 2)
        
    def __call__(self, x):
        """Forward pass"""
        return self.forward(x)
    
    def forward(self, x):
        """
        forward pass with reduced allocations
        """
        # Delegate to C++ extension
        z = cpp_detect.detect_forward(x, self.anchors.astype(np.float32), self.nc, self.stride)
        return (z, x)