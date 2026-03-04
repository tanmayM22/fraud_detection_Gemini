# Explainable Fraud Detection with Gemini

This project demonstrates a production-ready approach to credit card fraud detection, focusing on **False Positive Reduction** and **Explainable AI** (SAR generation).

## Project Structure
- `Dataset/`: Contains the ULB `creditcard.csv` file.
- `Source/`: Contains the main Jupyter notebook `fraud_explainable_gemini.ipynb`.
- `README.md`: This file.

## Getting Started
1. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn xgboost==2.1.3 shap google-generativeai matplotlib seaborn jupyter
   ```
   *Note: XGBoost 2.1.3 is required for SHAP compatibility.*
2. Set your Gemini API key:
   ```bash
   export GEMINI_API_KEY='your-api-key'
   ```
3. Run the notebook:
   ```bash
- `Dataset/`: Source data repository (not included; requires ULB `creditcard.csv`).
- `Source/`: 
  - `fraud_explainable_gemini.ipynb`: End-to-end research and model development.
  - `run_demo.py`: Slimmed-down terminal interface for performance validation.

## Environment Setup
The project requires Python 3.9+ and the following core libraries:
```bash
pip install pandas numpy scikit-learn xgboost==2.1.3 shap google-generativeai matplotlib seaborn jupyter
```
*Note: `xgboost==2.1.3` is specifically pinned for stable integration with the `shap` library's tree explainer.*

## Usage
1. **API Key Configuration**:
   ```bash
   export GEMINI_API_KEY='your_key_here'
   ```
2. **Notebook Execution**: Launch the Jupyter server and navigate to `Source/fraud_explainable_gemini.ipynb` to view the comprehensive analysis and visualizations.
3. **Quick Demo**: Run `python Source/run_demo.py` for a rapid evaluation of model metrics and a sample SAR generation.
