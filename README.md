# Weather Prediction System 🌤️

A complete Machine Learning pipeline and Streamlit web application for predicting rainfall.

## Features
- Full ML pipeline (Pandas, Scikit-learn, Matplotlib)
- Comparative analysis of Random Forest and Logistic Regression
- Detailed EDA saved as visual charts
- Modern Streamlit UI for user predictions
- Error handling and confidence scoring

## Installation
```bash
pip install -r requirements.txt
```

## Usage
1. **Generate Data (if needed):**
   ```bash
   python generate_data.py
   ```
2. **Train Model:**
   ```bash
   python train_model.py
   ```
3. **Run App:**
   ```bash
   streamlit run app.py
   ```

## Folder Structure
- `generate_data.py`: Dataset generator
- `train_model.py`: ML training pipeline
- `app.py`: Streamlit web application
- `eda/`: Generated charts
- `models/`: Trained model files
