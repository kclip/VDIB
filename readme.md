# snn
Code for Learning to Time-Decode in Spiking Neural Networks Through the Information Bottleneck.

# Run example
Code for our experiments can be run from the scripts

`python image_reconstruction.py`

`python mnistdvs_reconstruction.py`

`python predictive_coding_softmax.py`


# Data preprocessing
Preprocessing/loading for the MNIST-DVS dataset requires the `neurodata` module from github.com/kclip/neurodata
 
# Dependencies

numpy v1.18.1

pytorch v1.7.1

tables v3.6.1

scipy v1.4.1

neurodata (from github.com/kclip/neurodata)

snn (from github.com/kclip/snn) - up to commit 2da2eaa8626c8d361783fb3b9488181087cf6c33

# Support
The master branch contains the paper version of the code for reproducibility. 
This version is now obsolete, please have a look at the eprop-fixed branch for an updated version, including improved performance, support of minibatching, and fixes in the
 implementation of e-prop.


Written by Nicolas Skatchkovsky