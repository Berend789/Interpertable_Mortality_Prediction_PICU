# Interpertable_Mortality_Prediction_PICU

This is a complete preprocessing, model training, and figure generation repo for "Interpretable machine learning on Paediatric Intensive Care datasets to
improve mortality prediction models"

All files are build up similar. Due to the size of the data from PICURED, the dataset combinations of picured alone and picured combined with pice are processed in chunks rather than all at once for the pre-processing step.

All files can analayse the data from the three different timepoints by changing the hour variable to 6, 12 or 72.
All results will be saved in the set ouput folder

CHUNK_PICURED file to process PICURED database;
CHUNK_PICUR_PICE file to process PICURED database and PICE database;
PICE file to process PICE database;
PICE_LABPICURED file to process lab values from PICURED database and PICE;

All standard variables and all own functions consists of supporting variables and functions for the other files.

License
This project is licensed under the MIT License - see the LICENSE.md file for details
