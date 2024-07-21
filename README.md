# Deep-Learning-Projects
Detection of Hateful memes project
## Introduction
The rise of social media has led to an increase in harmful content, including hateful memes. In response, Facebook launched a competition to develop methods for detecting such memes.

The primary objective of this project is to enhance the performance of hateful meme detection models. This involves not only achieving high accuracy but also ensuring that the models are capable of generalizing well across diverse datasets and meme formats. The project explores a range of cutting-edge deep learning architectures, and employs various data preprocessing and augmentation techniques to optimize performance. 

[meme_example](Images/example_memes.png)

## Environment
For this project, PyTorch Lightning was utilized as the primary framework. PyTorch Lightning streamlines the implementation of PyTorch models by providing a high-level interface for organizing code, managing training loops, and handling common tasks such as checkpointing and logging. This enabled efficient experimentation and model development, ensuring a scalable and reproducible workflow.
## Dataset

## Data Preparation
Experimented with three separate approaches to remove text from the images to avoid learning the low-level features from the bold text on the images instead of images themselves.
1. Include an inpainting model based on Stable Diffusion through a Python packaged called Detextify.
2. An inpainting model which used OpenCV’s built-in processing capabilities and Frozen East Text Detection as seen in Hate-CLIPper.
3. Image cropping with Frozen East Text Detection.

Text augmentation technique for creating synthetic data is used. These techniques were implemented with python's NLPaug Package.

Experiments were perfromed with both datasets original and the modified ones as explained above.

## Model Development
In this project, the ConcatBERT architecture was used as the baseline model for hateful meme detection. ConcatBERT combines image and text features to classify memes effectively.It uses Bert as a language module and Resnet 152 as Vision module.
Below figure shows the implementation of the concatBERT architecture in pytorch lightning module. 

[architecture](Images/architecture.pdf)

To enhance performance, developed similar architecture as that of baseline (Concat BERT) and used different vision and language modules.
1. Concat DeBerta: Language module (DeBERTa)+ Vision Module(Resnet152)
2. Concat Roberta: Language module (RoBERTa)+ Vision Module(Resnet152) 
3. Clip: Used Multimodal Clip model (this model designed to understand visual and textual information jointly)

In addition, perfromed experimentation with different vision modules such as VGG16 and Resnet25 combination with different language modules.

These comparisons allowed to evaluate the strengths and weaknesses of each approach, ultimately aiming to identify the most effective model for hateful meme detection.
## References

## Contact