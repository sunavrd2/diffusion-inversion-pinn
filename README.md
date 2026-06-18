# Diffusion-Inversion PINN
Physics-Informed Neural Network (PINN) framework for inverting moisture diffusivity from sparse X-ray radiograph profiles in cement paste samples.

## Overview

This repository contains the code accompanying the paper:

> **DETERMINING MOISTURE-DEPENDENT DIFFUSIVITY IN EARLY-AGE CEMENT PASTE: A PINN-BASED INVERSION FROM SPARSE X-RAY RADIOGRAPHS**
> Sunav Raj Dahal, Hossein Kabir, Alexandre M. Tartakovskya, and Nishant Garg
> *Journal Name*, [year]. DOI: [to be added]

The framework infers the nonlinear diffusivity function D(θ) from sparse, noisy moisture content profiles measured at discrete times obtained by processing in-situ X-ray radiographs collected during capillary absorption experiment. Three saturation conditions in early-age cement paste specimens are studied: Low Sorptivity (LS), Medium Sorptivity (MS), and High Sorptivity (HS).

## Repository Structure

```
.
├── data/                     # Input X-ray attenuation (.npy) and time at which the radiographs are recorded. Note these profiles are in X-ray attenuation difference terms, 
                                need to apply correct attenuation factor and scaling to convert to water content. 
│   ├── LS_log_profiles.npy       # exerpimental X-ray attenuation for LS specimen 
│   ├── MS_log_profiles.npy       # exerpimental X-ray attenuation for MS specimen 
│   └── HS_log_profiles.npy       # exerpimental X-ray attenuation for HS specimen
│   ├── LS_difference_times.npy       # exerpimental times at which radiographs are recorded and water content is determined for LS specimen 
│   ├── MS_difference_times.npy       # exerpimental times at which radiographs are recorded and water content is determined for MS specimen 
│   └── HS_difference_times.npy       # exerpimental times at which radiographs are recorded and water content is determined for HS specimen 
├── FDM/                                # Finite Difference Method solutions using the learned D(θ); * = corresponding spceimen [HS, MS, LS]
│   ├── *_FDM_Xray_sample.ipynb         # numerical validation of PINN solution/inference of X-ray ragiography experiment   
│   └── *_FDM_grav_cuboid.ipynb         # experimental validation of D(θ) on independant specimens of same material/mix but different geometry to test the transferability or generalizability of the learned constitutive law
├── PINN_model/             # Core PINN implementation
│   ├── PINN_de.py                          # Main PINN architecture
│   ├── adaptive_activations.py             # Adaptive activation functions
│   ├── early_stop_callback_two_nets.py     # Training callbacks
│   ├── {LS,MS,HS}_de_main.py              # Training scripts per condition - run this file to train the PINN model
│   ├── {LS,MS,HS}_visualize.ipynb         # Results visualization
│   ├── LS_outputs/                         # Trained model outputs (LS)  - notes only 5 sample outputs. Run as many, adjust the visulaization code accordingly. 
│   └── MS_outputs/                         # Trained model outputs (MS)
└── Regression/                             # D(θ) regression fits i.e. determine the D(θ), by performing fits on the ensemble of PINN inferred D and corresponding θ distributions.
    └── {LS,MS,HS}_D_theta_fit.ipynb        # Download the PINN inference/ensemble data from zenodo/ Alternatively, generate your own data by running the PINN model 'HS_outputs/', 'MS_output', 'LS_output/' .... | change file links.  
└── X_ray_data_preprocessing/               # Contains .ipynb files to visualize, and preprocess the X-ray radiographs, i.e. generate average water content profiles from 2D radiographs. 
    └── {LS,MS,HS}_profiles_for_PINN.ipynb  # generates the profiles that can be used for trainig PINNs. Can be modified as required. Download the raw X-ray radiograph data from 'zenodo' and change file links accordingly.        
```

## Data

### Included in this repository
The `data/` folder contains preprocessed moisture content profiles from X-ray radiography experiments on cement paste samples at three saturation conditions (LS, MS, HS):

- `*_log_profiles.npy` — moisture content profiles at discrete time points
- `*_difference_times.npy` — corresponding time stamps

These are the direct inputs to the PINN training scripts.

### Raw radiograph data to be used with X_ray_data_prprodessing and regression data are hosted in separately (Zenodo)

> **[Dataset title]**
> Sunav Dahal et al.
> Zenodo. DOI: [https://doi.org/10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)

Download and place the files in the `Regression/` folder or any location of your chose. Be sure to update the file links in the notebooks or python scripts as required. 

## Dependencies

Python 3.8+ with atleast the following packages:

```
numpy
scipy
matplotlib
tensorflow >= 2.x
```

optional:
```
tqdm
......
.....

```



## Usage

### 1. Train the PINN model

```bash
python LS_de_main.py   # for LS
python MS_de_main.py   # for MS
python HS_de_main.py   # for HS
```

### 2. Visualize results

Open the corresponding notebook:

```bash
jupyter PINN_model/*_visualize.ipynb               # * = HS, MS, LS
```

### 3. FDM solve using learned D(θ) 

```bash
jupyter FDM/*_FDM_Xray_sample.ipynb            # run to check if PINN model inference and numerical solution is consistent   | * = HS, MS, LS
jupyter FDM/*_grav_cuboid.ipynb                 # run to check if the PINN learned D(θ) can extend or reproduce independant experiments of same mixes but different geometry/BCs etc. | * = HS, MS, LS
```

### 4. D(θ) regression fitting

After downloading the Zenodo dataset:

```bash
jupyter notebook Regression/*_D_theta_fit.ipynb         #* = HS, MS, LS
```

## Citation

If you use this code, please cite:

```bibtex
@article{______________________________,
  title   = {[______________________________]},
  author  = {________________________________},
  journal = {______________________},
  year    = {_______},
  doi     = {}
}
```

## License

MIT License

Copyright (c) 2026 Sunav Raj Dahal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Contact

Sunav Raj Dahal — PhD Student, Civil and Environmental Engineering, The Grainger College of Engineering, University of Illinois Urbana-Champaign
GitHub: [sunavrd2](https://github.com/sunavrd2)
LinkedIn: [in/sunavd](https://linkedin.com/in/sunavd)
