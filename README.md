# kNN Feature aggregation in CUDA
  - [How does it work?](#how-does-it-work)
  - [Usage](#usage)
  - [Installation](#installation)
  - [Acknowledgement](#acknowledgement)

This is a CUDA PyTorch binding of kNN feature aggregation with anisotropic gaussian weighting.

The code supports 2nd-order backpropagation through the 1st-order gradient w.r.t the query points. This enables calculation for [Eikonal constraint](https://proceedings.mlr.press/v119/gropp20a/gropp20a.pdf) that is commonly used for regularizing Neural SDF.
## How does it work?
Given `q` the query points, and `p` the feature point clouds, the function finds the k nearest neighbors of each query in the feature point clouds. The feature of the query is then determined by a weighted sum of the features from the k neighbors.

In particular, we compute
![](example/equation.png)

where
- `q (B, Q, 3)` is the query point coordinate with `B` batches, `Q` points
- `p (B, P, 3)` is the feature point coordinate with `B` batches, `P` points
- `f (B, P, F)` is the feature of the feature point associated with `p`
- `R (B, P, 3, 3)` is the rotation matrix of the gaussian.
- `sigma (B, P, 3)` is the scale of the gaussian for each point.
Note that we split the covariance matrix into rotation and scale.

## Usage
Below is a simple example on how to use the package.
```
import torch
from cuda_knn_aggregate import knn_aggregate_aniso

B = 2
Q = 1000
P = 500
K = 12
F = 32

# initialize query, points to query, feature, gaussian scale, and rotation
q = torch.rand(B, Q, 3).cuda()
p = torch.rand(B, P, 3).cuda()
f = torch.rand(B, P, F).cuda()
sigma = torch.rand(B, P, 3).cuda()
R = torch.rand(B, P, 3, 3).cuda()

# outputs are the aggregated feature, sum of weights, distance, and kNN indices
f_out, w_out, dists, idxs = knn_aggregate_aniso(q, p, f, sigma, R, K)
```

### Speedup
Tested with `B=2`, `Q=12288`, `P=4096`, `F=64` and `K=16` on a single RTX 3080.
The reported time is the time spent per 100 iteration, averaged over 100 trials.
| per 100 iters |         Forward         |         Backward        |  Backward with 2nd order |
|---------------|:-----------------------:|:-----------------------:|:------------------------:|
| CUDA          | 23.609 ms, ± 177.801 µs | 57.124 ms, ± 562.607 µs | 78.553 ms, ± 2708.505 µs |
| Plain Pytorch |  3.018 ms, ± 23.738 µs  | 20.806 ms, ± 355.332 µs | 29.818 ms, ± 463.335 µs  |

Overall, we see a 7x speedup in forward, and 2-3x speed in backward.

## Installation
Preqrequisites:
```
Python >= 3.8
CUDA >= 11.0
PyTorch >= 2.0.0
```

To install, simply do
```
pip install .
```

## Acknowledgement
This repository borrows code ([1](https://github.com/LemonATsu/CUDA-kNN-Aniso-Gaussian-Feature-Aggregation/blob/main/csrc/mink.cuh)) from [PyTorch3D](https://github.com/facebookresearch/pytorch3d).