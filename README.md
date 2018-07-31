# smfish_m2
A git repo gathering some modules for analysis of smFISH bio images. This project constituted my masters project, defended on June 2018 at Mines Paristech, Paris, France. I was hosted by the CBIO at Institut Curie, and supervised by Thomas Walter.

The core of this project is to study subcellular localization patterns of mRNAs. It involved the crafting of the whole treatment pipeline, from image filtering to image classification, passing by several segmentation substeps. A detailed report is given in *report.pdf*.

## Functionality
### Low level modules
- I/O
- Image filtering
- Spot detection (mRNAs detection)
- Unsupervised Nuclei semantic segmentation from DAPI
- Unsupervised Cytoplasm semantic segmentation from cell mask channel
- SNR (signal to noise ratio) computation for microscopy types comparison

### Mid level modules
- Single-cell image classification into identified localization patterns
- Unsupervised semantic segmentation of nuclei from the **DAPI channel**
- Supervised semantic segmentation of individual cells from the cell mask channel (not completely mature)

### High level modules
- Luigi pipelines for image db handling, model training and evaluation

## Example of low level modules 
Below are typical images from confocal microscopy we work with (channels left to right: smFISH, cell mask, DAPI): <br/>
<img src="/readme_images/multichannel.png" width="700">

### Spot detection 
(DoG filtering + local max filter in the [x, y, z, kernel size] dimensions) <br/>
Zoom in:  
<img src="/readme_images/spot_detection_in_situ.png" width="450">

### Nuclei segmentation from DAPI 
(bi class Otsu thresholding) <br/>
Zoom in: <br/>
<img src="/readme_images/segmented_DAPI.png" width="450">

### Nuclei segmentation from cell mask 
(bi class UNET trained on images segmented with the above method for the ground truth (checked by hand), raw cell mask images as input):  
Zoom in:  
<img src="/readme_images/segmented_DAPI_CM.png" width="450">

### Cytoplasm segmentation from cell mask 
(GMM thresholding for mask segmentation + watershed segmentation of touching cells):  
Zoom in:  
<img src="/readme_images/combined.png" width="700">

## Code architecture
### Spot detection module
#### Image sub module
Contains all the low level features such as: filtering (gaussian flavored filters or FFT filtering) in *filters*, segmentation (morphological segmentation of nuclei or cytoplasms) (segmentation.py), spot detection in *spotdetector.py*, along with some visualization scripts (basic) in *vizu.py* , and all the utils. The *tools.py* file contains some parallelization tools.   *Tools.py*, *useable_functions.py*, *fitters.py*, *smFISHanalyser.py* could be disregarded as they implement deprecated methods.  
   
The **Image.py** file defines the mother classes (i.e. FQimage for the smFISH channel, DAPIimage, CYTimage) which will call all the methods defined in the files mentioned above. It implements the I/O for readig tiff images, use case:  
1) For FQ images
```python 
fqimage = FQimage()
fqimage.load("path_to_image", shape=(1024, 1024, 35), dtype='uint16')
fqimage.show_images()

# FFT filter an image
fqimage.filter(op=FFT)

# Detect mRNAs
fqimage.detect_and_fit(detector=DoG)

# Show spots
fqimage.show_spots()
```

2) For DAPI images
```python
dapi = DAPIimage()
dapi.load("path", shape=(512, 512, 35))

# Segment using Otsu (fastest method but not the most robust)
dapi.segment(sg=FastNuclei())

# Show nuclei
dapi.show_nuclei()
```

3) For cell mask images, which need to be loaded along with identified nuclei for segmentation.
```python
cyt = CYTimage()
cyt.load("path", dapi.nuclei)
cyt.segment(CytoSegmenter())
```
  
#### Segmentation sub module
Contains script for training the UNET applied to segmentation of nuclei on cell mask images. Implementation is in keras with several Tensorflow extra operators such as the loss function design, or the various callbacks (such as the tensorboard)
- *feed_func.py* defines data generator and data augmentation operators.
- *unet.py* defines the unet architecture. Strategy to segmend touching cells is the one proposed in the original paper (i.e. inserting an inter-cell furrow which corresponds to a specific class with a much higher weight). In practice, this weight is changed dynamically from balanced (all class get the same weight) to its final value (usually 5x more weight given to the furrow class) through a keras callback.
- *main.py* runs.  
The tensorboard implemented allows one to track segmentation results on the test set epoch per epoch (with ground truth and prediction enlightened).


### Sequences
This module implements attemps to generate cell landscape using variational autoencoders. No approach has yielded satisying results as of now. 

### Classification
Contains all the script necessary to train classification model. Are implemented a pre-trained Inception v3 and a Squeezenet, along with various sub models used to reproduce some papers.
- *Augmentation.py* defines all the data augmentation operations used for training the model. All those operations were heavily optimized for speed.
- *Preprocessing.py* defines the Luigi tasks called before training the model, i.e. turning the raw data (multiple unstructured jsons) into cleaner dataframes. Those dataframes carry only indices (sets of points defining the cell membrane, nuclei membrane, mRNAs).
- *Callbacks.py* defines the keras callbacks used during training (tensorboard and others)
- *mnist_poc.py* contains objects and methods used to implement the *Unsupervised domain adaptation by backpropagation* and *Deep Reconstruction-Classification Networks for Unsupervised Domain Adaptation* papers for domain adaptation, tested against the MNIST dataset (reproducing the papers)
- *models.py* defines the keras models used.
- *Tools.py* could be disregarded.
- *utils.py* 
- **main.py** defines the various Luigi tasks such as training any of the models, evaluating them, predicting on real data with some trained model, etc.
 
