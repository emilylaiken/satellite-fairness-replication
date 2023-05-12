# Satellite Fairness Paper Replication Code

This repo contains replication code for the paper "Fairness and representation in satellite-based poverty maps: Evidence of urban-rural disparities and their impacts on downstream policy" by Emily Aiken, Esther Rolf, and Joshua Blumenstock (to appear in the proceedings of IJCAI '23). The paper is available on [arXiv](https://arxiv.org/abs/2305.01783). 

This replication repo contains the code used for the paper, but does not include the data. Data can be downloaded from the following sources:
- ACS survey in the US: Download via [Folktable](https://github.com/socialfoundations/folktables)
- Census in Mexico: Download via [IPUMS](https://international.ipums.org/international-action/sample_details/country/mx#tab_mx2015a)
- DHS surveys: Download via the [DHS website](https://dhsprogram.com/data/dataset_admin/login_main.cfm;jsessionid=B7C43ED6CE1E98AE6EB293C2014090B5.cfusion?CFID=95037839&CFTOKEN=2685203820ce7881-A5429BAB-920C-AEFF-52E446B83FE381DE)
- India SECC survey: Download via [SHRUG](https://www.devdatalab.org/shrug)
- Satellite featurizations from MOSAIKS: Download via [siml.berkeley.edu](https://siml.berkeley.edu/home/index/)

The code in this repo is organized as follows:
- Notebooks 1a - 1d: Cleaning survey data and extracting geometries to match to satellite imagery. 
- Notebooks 2a - 2c: Sampling tiles to obtain satellite featurizations from MOSAIKS, aggregating featurization together to a single representation per geometry. Includes replication code for Table S1, Figure 2, Figure S1. 
- Notebooks 3a - 3b: Machine learning simulations for predicting poverty and predicting urban/rural indicator. 
- Notebook 4: Analysis of ML simulation results. Includes replication code for Tables S2 and S3 and Figures 1, 3, S3, S4, and S5.
- Notebook 5: Recalibration approaches. Includes replication code for Figures 4 and S6.
- Notebook 6: Map visualizations. Includes replication code for Figure S2. 
