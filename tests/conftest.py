"""Local: bound CPU test overhead. Global: tiny numerical tests should not launch sixteen-thread kernels."""

import torch  # Test execution uses the same tensor library as the pipeline.

torch.set_num_threads(1)  # Small fixtures run faster and predictably with one CPU worker.
