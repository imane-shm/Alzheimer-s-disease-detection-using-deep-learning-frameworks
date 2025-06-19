# Alzheimer's MRI Classification

This repository provides a complete pipeline for classifying stages of Alzheimer's Disease from structural MRI slices using deep learning. Eight models are implemented and evaluated, including custom CNNs, pretrained CNN backbones (DenseNet, ResNet, MobileNet), and Vision Transformers (ViTs).

## Project Structure

├── models/ # model_1.py to model_8.py
├── train/ # train_model_1.py to train_model_8.py
├── preprocessing/
│ ├── base_preprocessing.py
│ ├── advanced_preprocessing.py
│ └── split_by_subject.py
├── utils/
│ ├── data_loaders.py
│ └── metrics.py
├── evaluate.py # test evaluation
└── README.md

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
ImgDataset/
├── MildDemented/
├── ModerateDemented/
├── NonDemented/
└── VeryMildDemented/


Use `split_by_subject.py` along with the provided metadata Excel file (`meta_data.xlsx`) to perform subject-level splitting. The resulting dataset will be saved under:
final_data/
├── train/
├── val/
└── test/


## Training

Each model is trained using a script from the `train/` directory. Example:

python train/train_model_1.py


Evaluation
To evaluate a trained model:
python evaluate.py --model_path path/to/best_model.pth --model_id 1
Outputs include accuracy, precision, recall, F1-score, and a confusion matrix.
Requirements
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt