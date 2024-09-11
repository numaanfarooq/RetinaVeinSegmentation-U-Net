# RetinaVeinSegmentation-U-Net
A deep learning project for accurate retinal vein segmentation using U-Net model. This repository includes detailed steps for data preprocessing, model training, and evaluation on the DRIVE dataset. 

![01_test_0](https://github.com/user-attachments/assets/ef79dad7-2286-4f45-9a01-7188ace317dd)




## Overview

This project focuses on segmenting retinal veins from medical images using a UNET-based deep learning model implemented in PyTorch. The goal is to assist in early diagnosis and monitoring of retinal diseases, which could potentially help healthcare professionals in detecting conditions like diabetic retinopathy, glaucoma, and hypertension.

The model is trained using the **DRIVE (Digital Retinal Images for Vessel Extraction) dataset consisting of retina vessel images along with their annotated binary mask. The dataset is consist of 20 images for both training and testing set.



#UNET architecture 

U-Net is a U-shaped encoder-decoder network consisting of four encoder blocks and four decoder blocks connected via a bridge, where the encoder network halves spatial dimensions and doubles feature channels, and the decoder network doubles spatial dimensions and halves feature channels, with skip connections between encoder and decoder blocks, making it a symmetric architecture efficient for image segmentation tasks, particularly in medical imaging and computer vision applications.




##Dataset

**Name**: DRIVE (Digital Retinal Images for Vessel Extraction)
**Structure**: 
  - Training images: 20
  - Testing images: 20
**Dataset Link** You can download the dataset from https://www.kaggle.com/datasets/zionfuo/drive2004/code






