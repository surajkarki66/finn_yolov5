import numpy as np

class Detect:
    """Detection layer for inference only"""
    stride = [32]  # strides computed during build

    def __init__(self, nc=80, anchors=()):  # detection layer
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [np.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [np.zeros(1)] * self.nl  # init anchor grid
        self.anchors = np.array(anchors, dtype=np.float32).reshape(self.nl, -1, 2)  # shape(nl,na,2)

    def __call__(self, x):
        """Forward pass - callable like a function"""
        return self.forward(x)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].reshape(bs, self.na, self.no, ny, nx).transpose(0, 1, 3, 4, 2)

            # Inference mode (always)
            if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

            y = x[i].copy()
            y[..., 0:2] = (y[..., 0:2] + 0.5 + self.grid[i]) * self.stride[i]  # xy
            y[..., 2:4] = (y[..., 2:4] + 1.0) ** 2 * self.anchor_grid[i]  # wh
            y[..., 4:] = y[..., 4:] / 2 + 0.5
            z.append(y.reshape(bs, -1, self.no))

        return (np.concatenate(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        # Create meshgrid using NumPy
        yv, xv = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
        grid = np.stack((xv, yv), 2)
        grid = np.tile(grid[np.newaxis, np.newaxis, :, :, :], (1, self.na, 1, 1, 1)).astype(np.float32)
        
        anchor_grid = (self.anchors[i].copy() * self.stride[i]).reshape(1, self.na, 1, 1, 2)
        anchor_grid = np.tile(anchor_grid, (1, 1, ny, nx, 1)).astype(np.float32)
        
        return grid, anchor_grid
   
