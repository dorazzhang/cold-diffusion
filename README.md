# Cold Diffusion: Re-implementation & Extension

## Introduction
* **Purpose:** This repository contains a PyTorch re-implementation of the paper *Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise* (Bansal et al., 2022).
* **Contribution:** The original paper challenges the assumption that Gaussian noise is essential to diffusion models, demonstrating that iterative refinement can reverse arbitrary deterministic degradations (like blur and pixelation) using an improved sampling procedure.

## Chosen Result
* **Objective:** We aimed to reproduce the conditional reconstruction results from Tables 1 and 3 of the original paper to verify that Gaussian noise is not necessary for diffusion-style image restoration. 
* **Focus:** We targeted Gaussian blur and pixelation degradations on the MNIST and CIFAR-10 datasets to test whether the method generalizes across mathematically distinct corruptions.

## GitHub Contents
* **`configs/`**: YAML files dictating model sizes, dataset paths, and degradation parameters (e.g., `mnist.yaml`, `cifar10.yaml`).
* **`src/`**: Core modules including the U-Net architecture, dataset loaders, degradation operators, and the sampling algorithms.
* **`utils.py`**: Centralized setup logic for argument parsing, device management, and config routing.
* **`train.py` / `generate.py` / `test_restoration.py` / `test_pipeline.py`**: Standalone executable scripts for training the model, generating sample grids, and evaluating quantitative metrics.
* **`cold_diffusion.ipynb`**: An interactive Google Colab notebook providing a streamlined, GUI-like wrapper to train and generate from the pipeline.

## Re-implementation Details
* **Approach:** We trained a U-Net restoration network to predict clean images from degraded inputs at timestep 1, minimizing the L1 loss. We evaluated both direct reconstruction and iterative sampling strategies using FID, SSIM, and RMSE metrics.
* **Modifications:** Due to compute constraints, we trained for 350 epochs instead of the original 700k gradient steps. 
* **Extension:** We designed a novel projection-based sampling variant (Algorithm 3) for pixelation that enforces a block mean constraint at each step to help decouple low-frequency consistency from high-frequency hallucination.

## Reproduction Steps
* **Environment & Compute:** Install dependencies via `pip install -r requirements.txt`. A CUDA-enabled GPU (e.g., T4/V100 on Colab) is highly recommended. CPU functionality is available but not recommended due to high compute requirements.
* **Datasets:** MNIST and CIFAR-10 are automatically downloaded by `torchvision` during execution. The download location is managed dynamically (depending on local run vs. Colab run).

**Option A: Interactive Colab Notebook (Recommended)**
* Open `cold_diffusion.ipynb`. The notebook automatically detects whether it is running locally or in Google Colab.
* If in Colab, it handles Google Drive mounting and intentionally reroutes dataset downloads to the local VM storage (`/content/data`) to prevent severe Drive I/O bottlenecks.
* Simply execute the cells sequentially to load a configuration, train the model, and display the generated restoration grids directly inline.

**Option B: Command Line (Local/Server)**
* **Training:** Run the training loop by passing a configuration file: 
  ```bash
  python train.py --config configs/mnist_blur.yaml
* **Generation:** Run the reverse sampler to restore images and save visualization grids to the samples/ directory (Supported algorithms: sampler (Alg 2), direct, adafusion, projection):
  ```bash
  python generate.py --config configs/mnist_blur.yaml --algorithm sampler

## Results/Insights
* **Visuals:** We qualitatively reproduced the paper's main finding; our models successfully recovered recognizable images from heavily degraded inputs across both blur and pixelation.
* **Metrics:** Our quantitative metrics (e.g., sampled FID of 114.36 for CIFAR-10 blur) were poorer than the original paper's (80.08) due to the heavily reduced training duration.
* **Surprises:** In several configurations, direct reconstruction unexpectedly outperformed Algorithm 2 quantitatively, highlighting a limitation in using standard metrics to assess visually comprehensible outputs.

## Conclusion
* **Takeaways:** The iterative refinement loop is the core mechanism of diffusion, capable of extending to deterministic degradations. However, the value of iterative sampling depends heavily on the degradation structure and the maturity of the restoration network.
* **Lessons Learned:** Long training runs require lightweight sanity checks (training on tiny subsets) and intermediate sampling visualizations to catch bugs early without wasting hours of compute.

## References
1. Bansal, A., Borgnia, E., Chu, H.-M., Li, J. S., Kazemi, H., Huang, F., Goldblum, M., Geiping, J., & Goldstein, T. (2022). Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise.
2. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. arXiv:2006.11239.
3. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. arXiv:1505.04597.

## Acknowledgements
We would like to acknowledge and thank Professor Weinberger, Professor Ma, and the teaching staff of the Cornell Spring 2026 CS 4782 course. Through their guidance throughout the semester and during the project, we were able to produce meaningful results and expand our knowledge in research and deep learning implementations.