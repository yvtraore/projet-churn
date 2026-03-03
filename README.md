# Projet Churn 📊

A machine learning project for predicting and analyzing customer churn using data science techniques, interactive visualizations, and API endpoints.

## Project Overview

This project implements a complete churn prediction pipeline, from exploratory data analysis to model deployment. It combines Jupyter notebooks for analysis, machine learning models for prediction, and modern web interfaces for visualization and interaction.

## 📁 Project Structure

```
projet-churn/
├── notebooks/              # Jupyter notebooks for analysis and experimentation
│   └── churn_model.ipynb  # Main churn prediction model and analysis
├── model/                  # Trained ML models and model artifacts
├── api/                    # REST API endpoints for model inference
├── streamlit/             # Interactive Streamlit dashboard
├── data/                  # Dataset files and data sources
├── img/                   # Images and visualizations
└── README.md
```

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- pip or conda

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yvtraore/projet-churn.git
cd projet-churn
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Components

### Notebooks (`notebooks/`)
- **churn_model.ipynb**: Comprehensive analysis including:
  - Exploratory Data Analysis (EDA)
  - Feature engineering and preprocessing
  - Model training and evaluation
  - Churn prediction insights

### Model (`model/`)
Serialized machine learning models ready for deployment and inference.

### API (`api/`)
RESTful API endpoints for making predictions on new data.

**Usage Example:**
```bash
python -m api.main
```

### Dashboard (`streamlit/`)
Interactive web interface built with Streamlit for visualizing churn patterns and making predictions.

**Launch the dashboard:**
```bash
streamlit run streamlit/app.py
```

### Data (`data/`)
Raw and processed datasets used for training and testing the churn prediction models.

## 🔍 Key Features

- **Predictive Models**: ML models trained to identify customers at risk of churn
- **Interactive Dashboard**: Real-time visualization and analysis with Streamlit
- **REST API**: Deploy predictions via HTTP endpoints
- **Data Analysis**: Comprehensive EDA in Jupyter notebooks
- **Reproducible Results**: Full analysis pipeline documented in notebooks

## 📈 Churn Prediction Workflow

1. **Data Loading & Exploration** → Understand customer data patterns
2. **Feature Engineering** → Create meaningful features for the model
3. **Model Training** → Train multiple ML models and select the best performer
4. **Evaluation** → Assess model performance on test data
5. **Deployment** → Deploy via API or dashboard for business use

## 🛠️ Technologies Used

- **Python** - Core programming language
- **Pandas & NumPy** - Data manipulation and analysis
- **Scikit-learn** - Machine learning models
- **Jupyter** - Interactive notebooks
- **Streamlit** - Web dashboard framework
- **Flask/FastAPI** - API framework (suggested)

## 📝 Usage

### Run Analysis
Open and execute the Jupyter notebook:
```bash
jupyter notebook notebooks/churn_model.ipynb
```

### Launch Web Dashboard
```bash
streamlit run streamlit/app.py
```

### Make Predictions via API
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [...]}'
```

## 📊 Expected Outputs

- Churn prediction models with performance metrics
- Feature importance analysis
- Visualizations of customer segments
- Interactive dashboard for exploring predictions
- API endpoints for integration with other systems

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for improvements.

## 📄 License

This project is open source and available under the MIT License.

## 📧 Contact

For questions or support, please open an issue in the repository.

---

**Last Updated**: March 2026
