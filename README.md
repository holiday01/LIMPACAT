LIMPACAT Pipeline
Overview
This repository contains a two-stage pipeline for analyzing immune cell infiltration in Liver Hepatocellular Carcinoma (LIHC) using:

CCD Model: Cell type deconvolution from bulk RNA-seq data
LIMPACAT Model: Whole Slide Image (WSI) analysis for immune cell abundance prediction

Repository Structure
```
.
├── CCD/
│   ├── deconvolution.py
│   ├── lihc_data.csv
│   ├── scRNA_log/
│   │   └── log_scrna.h5
│   ├── simulated_data/
│   │   ├── simulated_gex.csv
│   │   └── simulated_pro.csv
│   └── weight/
│       ├── model1.pt
│       ├── model2.pt
│       └── model3.pt
├── WSI/
│   ├── LIMPACAT.py
│   ├── Monocyte_svs.json
│   ├── NK_cell_svs.json
│   ├── Pro-B_cell_CD34+_svs.json
│   ├── log_B_cell_svs.json
│   ├── model/
│   │   ├── b.pt
│   │   ├── b34.pt
│   │   ├── mo.pt
│   │   └── nk.pt
│   ├── result/
│   │   └── hcc_immune_level.csv
│   └── run.sh
└── lihc_fraction/
    └── lihc_fraction.csv
```

CCD Model - Cell Type Deconvolution
Workflow

Data Simulation: The model generates simulated training data from single-cell RNA-seq reference (scRNA_log/log_scrna.h5)

Output: simulated_gex.csv (gene expression)
Output: simulated_pro.csv (cell proportions)


Model Training: CCD model is trained using the simulated data

Pre-trained weights: weight/model1.pt, model2.pt, model3.pt


Deconvolution: Execute CCD/deconvolution.py to:

Input: lihc_data.csv (bulk RNA-seq data)
Output: lihc_fraction/lihc_fraction.csv (immune cell fractions)
Classification: Assigns binary labels (0/1) based on immune cell abundance thresholds



Usage
```
bashcd CCD
python deconvolution.py
```


LIMPACAT Model - WSI Analysis
Overview
LIMPACAT analyzes Whole Slide Images to predict immune cell infiltration levels in LIHC samples.
Input Files
Four JSON configuration files contain LIHC sample annotations:

Monocyte_svs.json: Monocyte abundance groups
NK_cell_svs.json: NK cell abundance groups
Pro-B_cell_CD34+_svs.json: Pro-B cell (CD34+) abundance groups
log_B_cell_svs.json: B cell abundance groups

Each JSON maps WSI slide IDs to immune cell abundance classifications.
Pre-trained Models
Cell-type specific models located in model/:

b.pt: B cell model
b34.pt: Pro-B cell (CD34+) model
mo.pt: Monocyte model
nk.pt: NK cell model

Execution
```
cd WSI
bash run.sh
```
Output
Results are saved to WSI/result/hcc_immune_level.csv:

Binary classification (0/1)
1 = High immune cell abundance
0 = Low immune cell abundance

Pipeline Integration
The complete workflow:

CCD deconvolution → Estimates immune cell fractions from bulk RNA-seq
Binary classification → Determines high/low immune groups
LIMPACAT analysis → Validates predictions using WSI data
Final output → Integrated immune abundance predictions
