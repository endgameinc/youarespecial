# Create your own machine learning model from scratch
Playground/tutorial for machine learning malware models, including end-to-end deep learning for malicious file detection.

# Notebooks
For a gentle walk through the BSidesLV 2017 talk, it's recommended that you peruse through the notebooks in the following order
1. [BSidesLV -- your model isn't that special -- (1) MLP](BSidesLV%20--%20your%20model%20isn't%20that%20special%20--%20(1)%20MLP.ipynb)
2. [BSidesLV -- your model isn't that special -- (2) End-to-End](BSidesLV%20--%20your%20model%20isn't%20that%20special%20--%20(2)%20End-to-End.ipynb)
3. [BSidesLV -- your model isn't that special -- (3) MalwaResNet](BSidesLV%20--%20your%20model%20isn't%20that%20special%20--%20(3)%20MalwaResNet.ipynb)

# Getting started
## Requirements
### Python dependencies
This code was developed using Python 3.6.  Necessary packages can be installed by typing
```python
pip install -r requirements.txt
```
in your shell.

### Bring your own samples
Create subdirectories `malicious/` and `benign/` off of the main branch, and populate them with malicious and benign samples, respectively.  (Hint: this may be the most important step in creating your machine learning model.  For deep learning models, you should make sure to have at least 100K samples between the two subdirectories.)

## Training models
You may then try any of the malware models contained in the `classifier/` directory.  For example ```python classifier/modeltest_multilayer.py``` will extract features for all the the samples in `malicious/` and `benign/` and cache them into `sample_index.json` and `X.dat` numpy array, then build a multilayer percpetron on top of those features.

The end-to-end deep learning models do not require any feature extraction, however, can take a very long time (and a lot of data!) to train.  On a single Titan X GPU, we trained the simple end-to-end model in about 24 hours on 100K malicious and benign samples.  For the end-to-end deep learning models, it is recommended that you significantly increase the number of training epochs beyond what is contained in the model test scripts.
