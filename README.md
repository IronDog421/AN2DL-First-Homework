# Create the README content in English
readme_content = """
# AN2DL - First Homework: FeedForward Force

This repository contains the code, report, and supplementary materials for the **First Homework** project in the **Artificial Neural Networks and Deep Learning (AN2DL)** course. The project focuses on classifying 96x96 RGB images of blood cells into eight distinct classes using deep learning techniques.

## Authors
- Aman Saini (amannnammm)
- Marine Peuzet (mapeuzet7)
- Carlos Ruiz Aguirre (irondog421)
- Ariadna García Lorente (ariadnagarcia)

---

## Project Overview

This project applies transfer learning and data augmentation techniques to develop a robust classification model for eight types of blood cells:
- Basophils
- Eosinophils
- Erythroblasts
- Immature granulocytes
- Lymphocytes
- Monocytes
- Neutrophils
- Platelets

### Goals
- Develop an accurate model for multi-class classification.
- Address challenges like unbalanced data and anomalies.
- Optimize the model's architecture and training pipeline.

---

## Repository Structure

├── .git/ # Git repository files ├── Other Tested Models/ # Additional models used for evaluation ├── Predictions.ipynb # Notebook for making predictions ├── Preprocessing.ipynb # Notebook for data preprocessing ├── Training.ipynb # Notebook for model training ├── report.pdf # Final report detailing the project └── image.png # Visualization (e.g., class distributions, results)


---

## Key Features

1. **Dataset Preprocessing**:
   - Removal of anomalies (e.g., Shrek or Rickroll images) using perceptual hashing.
   - Data augmentation using AugMix and RandAugment techniques.

2. **Model Architecture**:
   - Built on EfficientNetV2 with ImageNet pre-trained weights.
   - Includes data augmentation layers and partial layer freezing for fine-tuning.

3. **Training Configuration**:
   - Weighted categorical cross-entropy loss to handle class imbalance.
   - Optimizer: AdamW with learning rate scheduling.

4. **Evaluation**:
   - Metrics: Accuracy, Precision, Recall, and F1 Score.
   - Comparison of VGG-19, ResNet101V2, and EfficientNetV2S models.

---

## Results

- **Best Model**: EfficientNetV2S with fine-tuning.
- **Accuracy**: 77% (17% improvement over baseline VGG-19).
- **Key Insights**:
  - Data augmentation significantly improved generalization.
  - Class weights helped balance the impact of unbalanced data.

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-url.git
   cd your-repo
