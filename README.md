# MSc-Physics-Thesis

This repository contains the code and data processing pipeline developed for my Master's thesis.
The project focuses on detecting planetary bow shock crossings using magnetic field data
and a supervised K-Nearest Neighbors classifier.

## Scientific Background

Bow shocks form upstream of planetary magnetospheres due to the interaction
between the solar wind and the planetary obstacle.
Identifying bow shock crossings in spacecraft data is essential for
studying solar wind–magnetosphere coupling.

## Repository Structure

The project is organized into logical modules reflecting the data processing
pipeline, physical modeling, machine learning, and visualization stages.

```
├── ajustes/
│   ├── bow_shock.py           # Cylindrical bow shock plotting & non-linear fits
│   └── Vignes.py              # Vignes-like hyperbolas functions
├── base_de_datos/
│   ├── conversiones.py        # Time conversions functions shared by project
│   ├── descarga.py            # Download MAG functions from LASP Colorado
│   ├── lectura.py             # Reading file functions
│   ├── recorte.py             # Cutting file functions
│   └── unión.py               # Merging file functions
├── machine_learning/
│   ├── clasificador_KNN.py    # Binary KNN Classifier for BS detection
│   ├── estadistica.py         # Feature extraction functions
│   └── validación_cruzada.py  # Cross validation algorithm for KNN
├── plots/
│   ├── animación_3D.py        # 3D trajectory animation from MAVEN MAG
│   ├── estilo_plots.py        # Style conventions shared by project
│   ├── MAG.py                 # Full 2D, 3D and temporal series plotting for MAG instrument
│   ├── SWEA.py                # Plotting functions for SWEA instrument
│   └── SWIA.py                # Plotting functions for SWIA instrument
├── main.py                    # Main file to run every command
└── README.md
```

## Methodology

1. MAG data are read and time-filtered around candidate events.
2. Statistical features (standard deviation, percentiles, extrema) are extracted
   from sliding windows.
3. A KNN classifier is trained using manually labeled bow shock crossings.
4. The trained model predicts crossings on unseen intervals.

## Requirements

- Python ≥ 3.10
- os
- numpy
- pandas
- matplotlib, mpl_toolkits
- datetime, timedelta
- requests
- cdflib
- scikit-learn

Install dependencies with:

bash

---

## 6. How to run (minimal working example)

Show the **shortest path from zero to result**.

markdown
## Usage

To train the classifier:

bash
python src/clasificador_KNN.py


If something is slow, fragile, or manual—**say it**.

---

## 7. Results (visual, not verbose)

One figure or short description is enough.

markdown
## Results

The classifier achieves a true positive rate of XX% on the test set.
Detected crossings are consistent with the empirical model of Vignes et al. (2000).

## Limitations

- The model depends on manually labeled events.
- Performance degrades for low signal-to-noise intervals.
- Only magnetic field data are used (no plasma moments).

## Citation

If you use this code, please cite:

Facundo Otero Zappa (2026).
*Machine Learning Detection of Bow Shock Crossings*.
Master's Thesis, Universidad de Buenos Aires.
