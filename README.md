# Fractal-Based Neural Network Research Project

## Project Overview
Research project exploring fractal-based approaches to neural network architectures.

## Project Structure
- `docs/`: Project documentation
- `src/`: Source code
- `tests/`: Unit and integration tests
- `data/`: Research data and datasets
- `weekly_reports/`: Weekly progress tracking
- `attachments/`: Supporting materials
- `papers/`: Research papers and publications

## Setup Instructions
1. Clone the repository
2. Install dependencies (see requirements.txt)
3. Run initial setup script

## Collaborators
- [Your Name]
- Supervisor: [Supervisor Name]

## License
[Choose an appropriate open-source license]

# Mushroom Classification Project

A deep learning project for classifying mushrooms as edible or poisonous.

## Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/FractalN/blob/main/Mushroom_Classifier.ipynb)

To run this project in Google Colab:

1. Click the "Open in Colab" button above
2. Make sure GPU runtime is enabled:
   - Runtime > Change runtime type > Hardware accelerator > GPU
3. Run all cells in order
4. Results will be saved to your Google Drive in 'FractalN_Results' folder

## Local Setup

```bash
git clone https://github.com/YOUR_USERNAME/FractalN.git
cd FractalN
pip install -r requirements.txt
```

## Dataset
The dataset is included in the repository using Git LFS. To clone with the dataset:

1. Install Git LFS:
```bash
git lfs install
```

2. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/FractalN.git
```

The dataset structure:
```
data/
├── poisonous mushroom sporocarp/
└── edible mushroom sporocarp/
```

Note: The total dataset size is approximately [SIZE]MB.

## Model Training

The complete training pipeline is available in `Mushroom_Classifier.ipynb`
