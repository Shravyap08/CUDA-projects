# FlashAttention-2 Forward Pass in Triton

![Python](https://img.shields.io/badge/Python-3.x-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg) ![Triton](https://img.shields.io/badge/Triton-NVIDIA-green.svg)

This project contains a from-scratch implementation of the **Non-Causal/Causal FlashAttention-2 forward pass** using the Triton language. The goal is to demonstrate the significant performance gains achievable with custom GPU kernels that optimize memory access patterns compared to a standard PyTorch implementation.

The primary notebook (`Causal_flash_attn2.ipynb`) includes:
* A baseline causal attention implementation in pure PyTorch.
* A memory-efficient, fused Triton kernel for the FlashAttention-2 forward pass.
* A correctness check to ensure the Triton kernel's output is numerically equivalent to the PyTorch version.
* A performance benchmark comparing the two implementations across various sequence lengths.
* Instructions for profiling the custom Triton kernel with NVIDIA's **Nsight Systems (nsys)** and **Nsight Compute (ncu)**.

## Performance Results

The Triton implementation shows a significant speedup, especially as the sequence length (`S`) increases. This is because the custom kernel minimizes slow reads/writes to HBM (GPU DRAM) by keeping intermediate results in the much faster on-chip SRAM.

The following results were generated on an NVIDIA A100 GPU:

| Sequence Length (S) | PyTorch Time (ms) | Triton Time (ms) | Speedup (vs. PyTorch) |
|:--------------------|:------------------|:-----------------|:----------------------|
| 64                  | 0.54              | 0.26             | 2.1×                  |
| 128                 | 0.88              | 0.32             | 2.8×                  |
| 256                 | 2.82              | 0.59             | 4.8×                  |
| 512                 | 10.84             | 1.59             | 6.8×                  |
| 1024                | 40.50             | 4.08             | 9.9×                  |
| 2048                | 160.94            | 14.63            | 11.0×                 |

**Execution Time vs. Sequence Length**
*(You can add a screenshot of your plot here)*
`![Performance Plot](./images/performance_plot.png)`

**Speedup Factor**
*(You can add a screenshot of your plot here)*
`![Speedup Plot](./images/speedup_plot.png)`


## How to Run

### Prerequisites
* A CUDA-enabled NVIDIA GPU (tested on A100).
* Python 3.8+
* Required libraries: `torch`, `triton`, `matplotlib`.

The easiest way to run this is in a GPU-enabled Google Colab environment.

### Steps
1.  Open the `Causal_flash_attn2.ipynb` notebook in a compatible Jupyter environment.
2.  Ensure you have a GPU runtime selected.
3.  Run the cells sequentially to set up the models, check for correctness, and run the benchmark and plotting.

## 🛠️ Profiling the Triton Kernel

The notebook contains cells to profile the custom kernel using NVIDIA's command-line tools. This allows for a deep dive into GPU performance metrics.

1.  **Install Nsight Systems:** A cell in the notebook handles the download and installation of the profiler.
2.  **Generate Script:** A `%%writefile` cell creates a standalone Python script (`profile_attention.py`) which is necessary for the profilers to run.
3.  **Execute Profiler:**
    * **Nsight Compute (`ncu`):** For detailed kernel-level metrics. The following command targets the Triton kernel when the sequence length is 1024.
        ```bash
        !ncu --set detailed --nvtx --nvtx-include "Triton S=1024/" -o triton_kernel_report.ncu-rep python profile_attention.py
        ```
    * **Nsight Systems (`nsys`):** For a system-wide timeline view of CPU and GPU work.
        ```bash
        !nsys profile --trace=cuda,nvtx -o flash_attention_report.nsys-rep python flash_attention.py
        ```
4.  **Analyze Report:** Download the generated `.ncu-rep` or `.nsys-rep` file and open it with the corresponding Nsight GUI on your local machine for detailed analysis.
