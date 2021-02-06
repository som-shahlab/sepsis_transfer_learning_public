## Prediction Utils

This library is composed of the following modules:
* `extraction_utils`: API for connecting to databases and extracting features
* `cohorts`: Definition for cohorts defined on the OMOP CDM
* `pytorch_utils`: Pytorch models for supervised learning

### Installation
0. If you are Nero a pre-existing conda environment is available at `/share/pi/nigam/envs/anaconda/envs/prediction_utils`. Otherwise, continue with the following steps.
1. Clone the repository
2. `pip install -e .` from within the directory

### Modules

#### extraction_utils
* Connect to databases using the BigQuery client library


#### pytorch_utils
* Several pipelines are implemented
    * Dataloaders
        * Sparse data in scipy.sparse.csr_matrix format
    * Layers
        * Input layers that efficiently handle sparse inputs
        * Feedforward networks
    * Training pipelines
        * Supervised learning for binary outcomes