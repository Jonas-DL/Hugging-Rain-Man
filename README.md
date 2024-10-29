# Hugging-Rain-Man

## Introduction

This repository contains the annotated Action Unit (AU) labels for the HRM dataset, along with pre-trained models for facial expression analysis. The dataset consists of 131,758 frames, organized into 1,535 segments. The images themselves are not publicly available due to privacy and ethical considerations. However, the AU labels,  and pre-trained models are provided to facilitate research and development in the field of facial expression analysis, particularly for Autism Spectrum Disorder (ASD).

## Dataset Description

### Overview
- **Total Frames**: 131,758
- **Segments**: 1,535
- **Action Units and Action Descriptors**: 22AUs + 10ADs
- **Atypical Rating**: Marked by 5 people.
- **Facial Expression**: It is obtained by soft voting using 3 algorithms
## Labels, Machine-extracted Features, and Pre-trained Models

### Overview
- **Model**: ResNet-50, JAA-Net, EmoFAN, ME-GraphAU, MAE-Face, FMAE
- **Training Data**: HRM dataset
- **Performance Metrics**: Accuracy, F1-Score
- **Machine-extracted Features**: InsightFace and OpenFace features (5 key points, head pose and bounding box, etc)

### Download Links
- **Baidu Cloud**: [Download Link](https://pan.baidu.com/s/1hMCuq4L892kl092uiDjrvw), pwd:CCNU
- **Mega Cloud (Only Pre-trained Models)**: [Download Link](https://mega.nz/folder/GVYRmbKa#5vfygvAm0mYl_h-6YbFzAQ
)

### Pre-trained Models Usage
You can access the original repositories of each project through the provided links. 
To use the pre-trained models, you can slightly modify the "evaluation" method of each model to 
change the output dimension to 17. Alternatively, you can use the code snippets we provide to 
perform AU testing. Any missing libraries can be found in the original repositories.

### Data Format
- **AU Labels**: CSV file with columns for frame number, AU activations, basic facial expression category and atypical rating .
- **Pre-trained Models**: PyTorch `.pth` files.


## Acknowledgment
We would like to express our gratitude to the following excellent open-source projects: [JAA-Net](), [EmoFAN](https://github.com/jingyang2017/aunet_train), 
[ME-GraphAU](https://github.com/CVI-SZU/ME-GraphAU), [MAE-Face](), 
[FMAE](https://github.com/forever208/FMAE-IAT), [EAC](), 
[Poster++](), and [DDAMFN++]().
## Citation
if the data or method help you in the research, please cite the following paper:
```
@article{your_paper,
  title={Your Paper Title},
  author={Your Name and Co-authors},
  journal={Journal Name},
  year={Year}
}
```
