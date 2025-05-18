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

- `data/`: Contains raw and processed datasets
- `src/`: Source code for data preprocessing, models, training, and evaluation
- `experiments/`: Configuration files for experiments
- `results/`: Experimental results and metrics
- `models/`: Saved model checkpoints
- `docs/`: Additional documentation, including the research paper

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
python -m src.data.preprocessing --input data/raw/promise_NFR.csv --output data/processed/promise_processed.csv
```

### Training Individual Models

```bash
python -m src.training.train --config experiments/individual_models/roberta_config.json
```

### Training Hybrid Models

```bash
python -m src.training.train --config experiments/hybrid_models/roberta_deberta_config.json
```

### Evaluation

```bash
python -m src.evaluation.metrics --model_dir models/individual/roberta --test_file data/processed/promise_processed.csv
```

### Running the Two-Phase Pipeline

```bash
python -m src.models.two_phase_pipeline --data data/processed/promise_processed.csv
```

## Results

Our experiments show that while individual models like RoBERTa and DeBERTa demonstrate strong performance (Macro F1 up to 0.5545 on PROMISE_EXP for RoBERTa), the RoBERTa–DeBERTa hybrid achieves superior results with a Macro F1-score of 0.6315 on the PROMISE dataset. Additionally, the two-phase RoBERTa pipeline attains perfect accuracy in distinguishing functional from non-functional requirements.

For detailed results, please refer to the `results/` directory or the research paper in `docs/paper.pdf`.



## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- We thank the creators of the PROMISE and PROMISE_EXP datasets for making their data publicly available.
- This research was conducted at Utica University.
