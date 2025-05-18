# Software Requirements Classification

This repository contains the code, data, and results for the research paper: "Multi-Level Software Requirements Classification: Assessing BERT Variants and their Hybrid Models" by Aryan K C and Unnati Shah.

## Overview

This research evaluates six transformer-based models (BERT, SBERT, RoBERTa, NorBERT, DeBERTa, and XLNet) and two hybrid configurations (RoBERTa–DeBERTa and BERT–XLNet) for multi-level Software Requirements (SRs) classification. The study uses the PROMISE and Extended PROMISE (PROMISE_EXP) datasets to assess model performance in distinguishing between Functional Requirements (FRs) and various types of Non-Functional Requirements (NFRs).

## Key Features

- Implementation of six individual BERT-based models for multi-level SR classification
- Novel hybrid model configurations combining complementary transformer architectures
- Two-phase RoBERTa pipeline for FR/NFR classification
- Comprehensive evaluation using Macro F1-score to address class imbalance
- Comparative analysis across PROMISE and PROMISE_EXP datasets

## Repository Structure

- `src/`: Source code for all models, including data preprocessing, training, and evaluation
- `Data/`: Contains raw and processed datasets
- `Promise_results/`: Experimental results for the PROMISE dataset
- `Promise_exp_results/`: Experimental results for the Extended PROMISE dataset
- `2phase_pipeline_results/`: Results from the two-phase RoBERTa pipeline
- `requirements.txt`: List of Python dependencies

## Installation

```bash
# Clone the repository
git clone https://github.com/Aryan-B25/Software-Requirements-Classification.git
cd Software-Requirements-Classification

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Data Preprocessing

```bash
python src/preprocessing.py --input Data/promise_NFR.csv --output Data/promise_processed.csv
```

### Training Individual Models

```bash
python src/train_model.py --model roberta --dataset Data/promise_processed.csv
```

### Training Hybrid Models

```bash
python src/train_hybrid.py --models roberta,deberta --dataset Data/promise_processed.csv
```

### Evaluation

```bash
python src/evaluate.py --model_dir src/models/roberta --test_file Data/promise_processed.csv
```

### Running the Two-Phase Pipeline

```bash
python src/two_phase_pipeline.py --data Data/promise_processed.csv
```

## Results

Our experiments show that while individual models like RoBERTa and DeBERTa demonstrate strong performance (Macro F1 up to 0.5545 on PROMISE_EXP for RoBERTa), the RoBERTa–DeBERTa hybrid achieves superior results with a Macro F1-score of 0.6315 on the PROMISE dataset. Additionally, the two-phase RoBERTa pipeline attains perfect accuracy in distinguishing functional from non-functional requirements.

For detailed results, please refer to the `Promise_results/`, `Promise_exp_results/`, and `2phase_pipeline_results/` directories.

## Citation

If you use this code or find our research helpful, please cite our paper:

```
@inproceedings{kc2025multi,
  title={Multi-Level Software Requirements Classification: Assessing BERT Variants and their Hybrid Models},
  author={KC, Aryan and Shah, Unnati},
  booktitle={Proceedings of the 37th International Conference on Software Engineering and Knowledge Engineering (SEKE)},
  year={2025},
  organization={KSI Research}
}
``

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- We thank the creators of the PROMISE and PROMISE_EXP datasets for making their data publicly available.
- This research was conducted at Utica University.


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15459346.svg)](https://doi.org/10.5281/zenodo.15459346)

