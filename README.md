# Alzheimer's MRI Classification

This repository provides a complete pipeline for classifying stages of Alzheimer's Disease from structural MRI slices using deep learning. Eight models are implemented and evaluated, including custom CNNs, pretrained CNN backbones (DenseNet, ResNet, MobileNet), and Vision Transformers (ViTs).

## Project Structure

The project is structured into clearly defined directories to support modular development and easy navigation. The models/ directory contains the implementation of all eight models (model_1.py through model_8.py), including both custom convolutional neural networks and pretrained architectures like DenseNet, ResNet, MobileNet, and Vision Transformers. Training scripts corresponding to each model are located in the train/ folder, with filenames following the pattern train_model_X.py. The preprocessing/ directory includes three Python files: base_preprocessing.py for standard image transformations( to be used with models 1 anf 7 ) , advanced_preprocessing.py for enhanced filtering using histogram equalization and bilateral filtering( to be used with models 2 3 4 5 6 8), and split_by_subject.py, which ensures subject-level dataset splitting to prevent data leakage. Utility functions used across models are stored in the utils/ folder, which contains data_loaders.py for loading image datasets and metrics.py for computing performance metrics. The root directory also includes an evaluate.py script for testing trained models and generating evaluation reports, and a README.md file that documents the usage and structure of the repository.

## Models Overview

| ID | Model             | Preprocessing                       |
|----|-------------------|-------------------------------------|
| 1  | Custom CNN        | Standard                            |
| 2  | Custom CNN        |advanced |
| 3  | DenseNet201       | Advanced                            |
| 4  | ResNet18       | Advanced                            |
| 5  | ResNet50  | Advanced                            |
| 6  | MobileNetV3-Large | Advanced                            |
| 7  | Vision Transformer| Standard                            |
| 8  | Vision Transformer| Advanced                            |

## Dataset Format

The input dataset should be placed under a root folder named `ImgDataset/` with the following subfolders
MildDemented, ModerateDemented, NonDemented, VeryMildDemented/


Use `split_by_subject.py` along with the provided metadata Excel file (`meta_data.xlsx`) to perform subject-level splitting. The resulting dataset will be saved under:final_data/ train/val/ test/


## Training

Each model is trained using a script from the `train/` directory. Example:

python train/train_model_1.py


Evaluation
To evaluate a trained model:
python evaluate.py --model_path path/to/best_model.pth --model_id 1
Outputs include accuracy, precision, recall, F1-score, and a confusion matrix.
Requirements
Install dependencies:

pip install -r requirements.txt
