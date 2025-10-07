# CPTS528-MLAdversary-Project
This is the project repository for CPTS528 Project.

## Overview
This is a project for **CPTS 428/528: Software Security and Reverse Engineering** course.  
The goal of this project is to explore the vulnerability of machine learning models to adversarial attacks and to study methods that can improve adversarial robustness.

## Project Objectives
1. Demonstrate how adversarial examples can affect image classifiers (especially CIFAR-10 in this project).
2. Analyze the performance degradation caused by different attacks.
3. Implement and evaluate defenses to increase robustness against adversarial perturbations.

## Team Information
- **Team Members:**  
  - John Ye

## Project Structure
```text
CPTS528-MLAdversary-Project/
├── data_loader.py                 # DatasetManager – loads and preprocesses CIFAR-10 data
├── model_trainer.py               # ModelTrainer – trains, validates, and saves CNN models
├── adversary_engine.py            # AdversarialEngine – generates adversarial examples for robustness testing (To be implemented in next milestone)
├── defense_module.py              # DefenseModule – applies defense strategies to improve model resilience (To be implemented in next milestone)
├── evaluator.py                   # Evaluator – compares model performance on clean and adversarial data (To be implemented in next milestone)
├── main.py                        # Experiment Orchestrator – runs the full pipeline
│
├── config.yaml                    # YAML configuration file for experiment settings
├── trained_model.pth              # Saved model after training (auto-generated)
│
├── data/                          # CIFAR-10 dataset (auto-downloaded here)
├── logs/                          # Optional folder for logs and metrics
│
└── utils/                         # Helper utilities
    ├── config.py                  # YAML config loader
    └── seed.py                    # Random seed setup for reproducibility
│
└── README.md
```

## Requirements

Before running the project, install the required Python libraries:

```bash
pip install torch torchvision pyyaml
```

CIFAR-10 will be automatically downloaded to ./data during the first run.

Trained model weights will be saved to the path specified in save_path.

## How to Run
```bash
python main.py
```
For this milestone2-2, this will:
1.Load configurations and set random seeds.
2.Prepare CIFAR-10 training and test data loaders.
3. Build a simple CNN model for classification.
4. Train and validate the model.
5. Save trained weights to trained_model.pth