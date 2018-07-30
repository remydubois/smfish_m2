# smfish_m2
A git repo gathering some modules for analysis of smFISH bio images. This project constituted my masters project, defended on June 2018 at Mines Paristech, Paris, France. I was hosted by the CBIO at Institut Curie, and supervised by Thomas Walter.

The core of this project is to study subcellular localization patterns of mRNAs. It involved the crafting of the whole treatment pipeline, from image filtering to image classification, passing by several segmentation substeps.

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
- Supervised semantic segmentation of nuclei from the **cell mask channel**
- Supervised semantic segmentation of individual cells from the cell mask chanenl (not completely mature)

### High level modules
- Luigi pipelines for image db handling, model training and evaluation

## Example of low level modules 
Below are typical images from confocal microscopy we work with (channels left to right: smFISH, DAPI, cell mask):
# smfish_m2
A git repo gathering some modules for analysis of smFISH bio images. This project constituted my masters project, defended on June 2018 at Mines Paristech, Paris, France. I was hosted by the CBIO at Institut Curie, and supervised by Thomas Walter.

The core of this project is to study subcellular localization patterns of mRNAs. It involved the crafting of the whole treatment pipeline, from image filtering to image classification, passing by several segmentation substeps.

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
- Supervised semantic segmentation of nuclei from the **cell mask channel**
- Supervised semantic segmentation of individual cells from the cell mask chanenl (not completely mature)

### High level modules
- Luigi pipelines for image db handling, model training and evaluation

## Example of low level modules 
Below are typical images from confocal microscopy we work with (channels left to right: smFISH, cell mask, DAPI): <br/>
<img src="/readme_images/multichannel.png" width="450">

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
Contains all the low level features such as: filtering (gaussian flavored filters or FFT filtering) in *filters*, segmentation (morphological segmentation of nuclei or cytoplasms) (segmentation.py), spot detection in *spotdetector.py*, along with some visualization scripts (basic) in *vizu.py* , and all the utils. The *tools.py* file contains some parallelization tools.  *Tools.py*, *useable_functions.py*, *fitters.py*, *smFISHanalyser.py* could be disregarded as they implement deprecated methods.  
   
The **Image.py** file defines the mother classes (i.e. FQimage for the smFISH channel, DAPIimage, CYTimage) which call all the methods defined in the files mentioned above.
- Contains parallelization scripts for segmentation  __under__
