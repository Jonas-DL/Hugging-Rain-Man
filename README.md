# Hugging-Rain-Man

## Introduction

This repository contains the annotated Action Unit (AU) and Action Descriptor (AD) labels for the HRM dataset, 
along with pre-trained models for facial action detection and atypical expression regression. The dataset consists of 131,758 frames, 
organized into 1,535 segments. The images themselves are not publicly available temporarily due to privacy and 
ethical considerations. However, the AU labels,  and pre-trained models are provided to facilitate 
research and development in the field of children facial expression analysis, particularly for Autism Spectrum Disorder (ASD).

## Dataset Description

### Overview
- **Total Frames**: 131,758
- **Segments**: 1,535
- **Action Units and Action Descriptors**: 22AUs + 10ADs
- **Atypical Rating**: Annotated by 5 people
- **Facial Expression**: It is obtained by soft voting using 3 algorithms
## Labels, Machine-extracted Features, and Pre-trained Models

### Overview
- **Model**: ResNet-50, EmoFAN, ME-GraphAU, MAE-Face, FMAE
- **Training Data**: HRM dataset
- - **Selected AUs/ADs for detection**: AU1, AU2, AU4, AU6, AU7, AU9, AU10, AU12, AU14, AU15, AU16, AU17, AU18, AD19, AU20, AU23, AU24, AU25, AU2X (AU26/27), AU28, AD32, and AU43E.
- **Performance Metrics**: Accuracy, F1-Score
- **Machine-extracted Features**: InsightFace and OpenFace features (5 key points, head pose and bounding box, etc)

### Download Links
- **Baidu Cloud**: [Download Link](https://pan.baidu.com/s/1hMCuq4L892kl092uiDjrvw), pwd:CCNU
- **Mega Cloud (Only Pre-trained Models)**: [Download Link](https://mega.nz/folder/GVYRmbKa#5vfygvAm0mYl_h-6YbFzAQ
)

### Pre-trained Models Usage
You can access the original repositories of each project through the provided links. 
To use the pre-trained models, you can slightly modify the "evaluation" method of each model to 
change the output dimension to 17 or 22. Alternatively, you can use the code snippets we provide in the [Predict](https://github.com/Jonas-DL/Hugging-Rain-Man/tree/main/Predict) directory to 
perform AU testing. Any missing libraries can be found in the original repositories.

### Data Format
- **AU Labels**: CSV file with columns for frame number, AU activations, basic facial expression category and atypical rating .
- **Pre-trained Models**: PyTorch `.pth` files.
- **Machine-extracted Features**: `.csv` files.

## Acknowledgment
We would like to express our gratitude to the following excellent open-source projects: [JAA-Net](https://github.com/ZhiwenShao/PyTorch-JAANet),[EmoFAN](https://github.com/face-analysis/emonet), [EmoFAN4AU-Detection](https://github.com/jingyang2017/aunet_train), 
[ME-GraphAU](https://github.com/CVI-SZU/ME-GraphAU), [MAE-Face](https://github.com/FuxiVirtualHuman/MAE-Face), 
[FMAE](https://github.com/forever208/FMAE-IAT), [EAC](https://github.com/zyh-uaiaaaa/Erasing-Attention-Consistency), 
[Poster++](https://github.com/talented-q/poster_v2), and [DDAMFN++](https://github.com/SainingZhang/DDAMFN).
## Citation
if the data or method help you in the research, please cite the following paper:
```
@article{Wait for an update,
  title={Paper Title},
  author={Name and Co-authors},
  journal={Journal Name},
  year={Year}
}
```
