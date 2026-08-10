import numpy as np


class Detect:
    """
    Pure NumPy reimplementation of the YOLO-style Detect head.
    Equivalent to the original PyTorch nn.Module version, but with
    torch, cpp_detect, and training-mode code paths removed since
    this is inference-only (FINN/FPGA accelerator output -> boxes).
    """

    stride = [32]  # strides computed during build

    def __init__(self, nc=1, anchors=()):
        self.nc = nc                                   # number of classes
        self.no = nc + 5                                # number of outputs per anchor
        self.nl = len(anchors)                          # number of detection layers
        self.anchors = np.array(anchors, dtype=np.float32).reshape(self.nl, -1, 2)  # (nl, na, 2)
        self.na = self.anchors.shape[1]                 # number of anchors per layer

        # cached grids, built lazily per layer once (ny, nx) is known
        self.grid = [None] * self.nl
        self.anchor_grid = [None] * self.nl
        self._grid_shape = [None] * self.nl

    def __call__(self, x):
        """Forward pass"""
        return self.forward(x)

    def _make_grid(self, nx, ny, i):
        """
        Build the (1, na, ny, nx, 2) position grid and the
        (1, na, ny, nx, 2) anchor grid for detection layer i,
        matching the original torch.meshgrid(indexing='ij') convention.
        """
        yv, xv = np.meshgrid(
            np.arange(ny, dtype=np.float32),
            np.arange(nx, dtype=np.float32),
            indexing="ij",
        )
        grid = np.stack((xv, yv), axis=-1)                       # (ny, nx, 2)
        grid = np.broadcast_to(grid, (1, self.na, ny, nx, 2)).astype(np.float32)

        anchor_grid = (self.anchors[i] * self.stride[i]).reshape(1, self.na, 1, 1, 2)
        anchor_grid = np.broadcast_to(anchor_grid, (1, self.na, ny, nx, 2)).astype(np.float32)

        return grid, anchor_grid

    def forward(self, x):
        """
        x: list of length nl, each element shape (bs, na*no, ny, nx)
        Returns: (z, x) where z has shape (bs, total_preds, no)
        """
        z = []

        for i in range(self.nl):
            xi = x[i]
            bs, _, ny, nx = xi.shape

            # (bs, na*no, ny, nx) -> (bs, na, no, ny, nx) -> (bs, na, ny, nx, no)
            xi = xi.reshape(bs, self.na, self.no, ny, nx).transpose(0, 1, 3, 4, 2)

            if self._grid_shape[i] != (ny, nx):
                self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                self._grid_shape[i] = (ny, nx)

            y = xi.copy()
            y[..., 0:2] = (y[..., 0:2] + 0.5 + self.grid[i]) * self.stride[i]        # xy
            y[..., 2:4] = (y[..., 2:4] + 1.0) ** 2 * self.anchor_grid[i]              # wh
            y[..., 4:] = y[..., 4:] * 0.5 + 0.5                                       # obj + cls conf

            z.append(y.reshape(bs, -1, self.no))

        return (np.concatenate(z, axis=1), x)