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
Below are typical images from confocal microscopy we work with (channels left to right: smFISH, DAPI, cell mask):
![](/readme_images/multichannel)

### Spot detection 
(DoG filtering + local max filter in the [x, y, z, kernel size] dimensions)
Zoom in:
![](/readme_images/spot_detection_in_situ.png)

### Nuclei segmentation from DAPI 
(bi class Otsu thresholding)
Zoom in:
![](/readme_images/segmented_DAPI.png)

### Nuclei segmentation from cell mask 
(bi class UNET trained on images segmented with the above method):
Zoom in

### Cytoplasm segmentation from cell mask 
(GMM thresholding for mask segmentation + watershed segmentation of touching cells):
Zoom in:







### Spot detection 
(DoG filtering + local max filter in the [x, y, z, kernel size] dimensions)
Zoom in:

### Nuclei segmentation from DAPI 
(bi class Otsu thresholding)
Zoom in:

### Nuclei segmentation from cell mask 
(bi class UNET trained on images segmented with the above method):
Zoom in

### Cytoplasm segmentation from cell mask 
(GMM thresholding for mask segmentation + watershed segmentation of touching cells):
Zoom in:







