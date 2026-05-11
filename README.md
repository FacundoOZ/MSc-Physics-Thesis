# MSc Thesis Project — Automated Bow Shock Detection Using Machine Learning

## Overview

This repository contains the code and data-processing pipeline developed for my MSc Physics thesis.

The project focuses on the automated detection of planetary bow shock crossings (Mars) using magnetic field measurements from NASA MAVEN spacecraft data and supervised machine learning techniques.



The pipeline includes:



* spacecraft data acquisition and preprocessing,
* statistical feature extraction
* supervised classification with K-Nearest Neighbors (KNN) and postprocessing,
* empirical bow shock modeling,
* and scientific visualization tools.



This project combines scientific computing, time-series analysis, and machine learning for space physics applications.

## Scientific Background

Planetary bow shocks form upstream of magnetospheres due to the interaction between the solar wind and the planetary obstacle.



Detecting bow shock crossings in spacecraft observations is important for studying:

* solar wind–magnetosphere coupling,
* plasma dynamics,
* shock physics,
* and the interaction between planetary atmospheres and the solar wind.



In particular, the MAVEN spacecraft was designed to study the loss of the Martian upper atmosphere and its interaction with the solar wind, providing important insights into how Mars lost its global magnetic field over billions of years. This work uses magnetic field observations from the MAVEN mission to automatically identify candidate bow shock crossings.

## Requirements

* Python 3
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* cdflib
* Scientific Computing
* Machine Learning
* Time-series analysis
* NASA MAVEN mission data



## Repository Structure

The project is organized into logical modules reflecting the data processing
pipeline, physical modeling, machine learning, and visualization stages.

```
├── ajustes/
│   ├── bow\_shock.py          # Cylindrical bow shock (non-linear) fitting and visualization
│   └── Vignes.py              # Vignes-like hyperbolic bow shock models
├── base\_de\_datos/
│   ├── conversiones.py        # Shared time conversion utilities
│   ├── descarga.py            # MAVEN MAG data download functions from LASP Institute, Colorado
│   ├── lectura.py             # File reading utilities
│   ├── promedio.py            # Average post-processing functions
│   ├── recorte.py             # Filtering functions
│   └── unión.py               # Data merging utilities
├── machine\_learning/
│   ├── clasificador\_KNN.py   # Binary kNN Classifier for bow shock detection
│   ├── estadistica.py         # Statistical feature extraction
│   ├── métricas.py            # Machine learning evaluation metrics
│   └── validación\_cruzada.py # Cross-validation utilities for kNN
├── plots/
│   ├── animación\_3D.py       # 3D MAVEN trajectory animations
│   ├── estilo\_plots.py       # Shared plotting style configuration
│   ├── MAG.py                 # MAG instrument visualization
│   ├── SWEA.py                # SWEA instrument visualization
│   └── SWIA.py                # SWIA instrument visualization
├── main.py                    # Main execution script
└── README.md
```

## Methodology

The workflow consists of the following stages:



1. Magnetic field data are downloaded and filtered according to spacecraft location, Martian crustal magnetic field regions, and empirical bow shock geometry.
2. Sliding windows are applied to spacecraft measurements.
3. Statistical and temporal features are extracted, including variance, percentiles, extrema, and local magnetic field fluctuations.
4. A supervised KNN classifier is trained using manually labeled bow shock crossings spanning six years of observations (2014–2019).
5. The trained model predicts candidate crossings on previously unseen spacecraft intervals from 2014 to 2025.
6. Predictions are compared against empirical bow shock models and spacecraft trajectories.
7. No significant correlation is observed between the average bow shock position and Mars’ orbital position, seasonal variations, or the solar activity cycle.

## Installation

**Requirements**

* Python 3.10+
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Requests
* cdflib



Install dependences with:

```
pip install numpy pandas matplotlib scikit-learn requests cdflib
```

## Usage

Run the complete pipeline with:

```
python main.py
```

and change the variable "ruta" to your folder location.

## Results

The classifier successfully identifies manually labeled bow shock crossings using only magnetic field measurements.



The detected events are physically consistent with empirical bow shock models such as the Vignes et al. model and reproduce expected spacecraft crossing signatures.



Example outputs include:



* bow shock crossing detections,
* magnetic field time-series visualizations,
* 3D spacecraft trajectory animations,
* and empirical bow shock fitting plots.

## Limitations

* The classifier depends on manually labeled training events, which are not complete.
* Performance decreases during low signal-to-noise intervals.
* Only magnetic field data are currently used.
* The model does not yet incorporate plasma moments or deep learning approaches.

## Future Improvements

Potential future extensions include:

* incorporating plasma instrument measurements,
* testing additional machine learning models,
* automated depuration of incorrectly identified bow shock events,
* and real-time event detection pipelines.

## Citation

If you use this repository, please cite:



Facundo Otero Zappa (2026).

Identificación del bow shock marciano mediante un clasificador $k$-NN binario supervisado.
Master's Thesis, Universidad de Buenos Aires.

## Author

Facundo Otero Zappa

MSc Physics Graduate | MSc Computer Science Student

Machine Learning • Scientific Computing • Space Physics

