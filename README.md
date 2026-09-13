# SAMF

## Introduction
SAMF (Structure-constrained Attention and Hierarchical Multi-scale Fusion for Progressive Integration of Multi-slice Spatial Transcriptomics) is a comprehensive analytical framework designed for multi-slice spatially resolved transcriptomics (SRT) data. It integrates wavelet transform-based multi-scale feature extraction, causal inference-guided attention mechanisms, and multi-agent adaptive loss balancing to decipher spatial structures and correct batch effects across tissue slices. The method supports spatial domain identification, cross-slice integration, cross-developmental stage comparisons, and cross-platform data harmonization.

---

## Catalogs
- **/SAMF**: Contains the core implementation of the SAMF algorithm.
  - `Models.py`: SAMF model architecture, including the wavelet transform module (`SpaWaveletTransform`), causal inference module (`SpaCausalInference`), Encoder, Decoder, and the main `SAMF_model` class.
  - `Pipeline.py`: Training pipelines (`SC_pipeline` and `SC_BC_pipeline`) with multi-agent controller integration for adaptive loss weight optimization.
  - `GLNS.py`: Graph-based local-neighborhood sampling (GLNSampler) for constructing positive/negative pairs in contrastive learning.
  - `Align.py`: Spatial coordinate alignment utilities.
  - `Clust.py`: Clustering and evaluation functions.
  - `Func.py`: Utility and helper functions.
  - `Utils.py`: Data preprocessing utilities.
  - `agent1.py`: Multi-agent controller with random exploration strategy for per-slice loss weight tuning.
- **/Basis**: Contains implementations of 12 baseline methods for comparative evaluation (CCST, DeepST, DiffusionST, GraphST, SEDR, SPACEL, SPIRAL, STAGATE, STAligner, STitch3D, SpaGCN, stDCL).
- **/Config**: YAML configuration files for each dataset (DLPFC, MERFISH, STARmap, osmFISH).
- **/run**: Jupyter notebooks to reproduce the results presented in the manuscript.
- **requirement.txt**: List of required Python packages.

---

## Environment
The SAMF code has been implemented and tested in the following development environment:

- Python == 3.10
- PyTorch == 2.2.2+cu118
- PyTorch Geometric == 2.5.2
- Scanpy == 1.10.1
- NumPy == 1.26.3
- SciPy == 1.13.0
- Scikit-learn == 1.4.2
- Pandas == 2.2.2
- Matplotlib == 3.8.4
- rpy2 == 3.5.16

```bash
pip install -r requirement.txt
```

---

## Dataset
All datasets used in the manuscript can be downloaded from:
[https://zenodo.org/records/15090086](https://zenodo.org/records/15090086)

- **DLPFC**: 12 slices of human dorsolateral prefrontal cortex (Visium 10x).  {http://spatial.libd.org/spatialLIBD/

- **STARmap**: Mouse brain data (STARmap platform).  https://www.science.org/doi/10.1126/science.aat5691
- **osmFISH**: Mouse somatosensory cortex (osmFISH platform).  https://doi.org/10.1038/s41592-018-0175-3

Configuration files for each dataset are provided in the `/Config` directory, specifying model hyperparameters, training settings, and preprocessing options.

---

## How to Run the Code
1. **Install dependencies**:
    ```bash
    pip install -r requirement.txt
    ```

2. **Download datasets** from [Zenodo](https://zenodo.org/records/15090086) and place them in the appropriate data directory.

3. **Configure the experiment** by editing the corresponding YAML file in `/Config` (e.g., `DLPFC.yaml` for DLPFC data).

4. **Run the SAMF pipeline**:
    ```bash
    python main.py --config Config/DLPFC.yaml
    ```

5. **To reproduce manuscript results**, run the notebooks in `/run` corresponding to each dataset.

---

## Contact
If you have any questions, please contact 332516060892@zzuli.edu.cn.
