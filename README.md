# GOOG Stock Trading Strategy

This project implements a machine learning-based trading strategy for Google's stock (GOOG) using technical indicators and a Random Forest Classifier.

## Project Structure

- `data/`: Contains raw and processed data.
- `src/`: Source code for data processing, feature engineering, model training, evaluation, and backtesting.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and interactive experiments.
- `scripts/`: Scripts to run the entire pipeline.
- `plots/`: Generated plots and visualizations.
- `tests/`: Unit tests for ensuring code reliability.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/your-project.git
    cd your-project
    ```

2. Create a virtual environment:
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the data pipeline:
    ```bash
    python scripts/run_pipeline.py
    ```

2. View generated plots in the `plots/` directory.

## Testing

Run unit tests using:
```bash
python -m unittest discover tests
