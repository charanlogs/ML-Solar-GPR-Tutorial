# Uncertainty-Aware Solar Forecasting using Gaussian Process Regression (GPR)

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 📌 Project Overview
This repository contains a professional-standard tutorial on **Gaussian Process Regression (GPR)**. The project demonstrates how to use Bayesian inference to forecast solar power generation while quantifying model **Uncertainty**. 

Unlike standard "black-box" models, this GPR implementation provides a 95% confidence interval, allowing grid operators to manage risk and predict renewable energy yield more effectively.

## 🚀 Key Technical Features
- **Data Synergy:** Fusing weather sensor data (Irradiation) with power plant generation logs.
- **Kernel Comparison:** Evaluating three different architectures (Simple RBF, Periodic, and Composite) to find the optimal fit.
- **Hyperparameter Optimization:** Automatic tuning of length-scales and noise levels via Log-Marginal Likelihood (LML).
- **Professional Workflow:** Automated data scaling, daytime filtering, and result visualization.

## 📁 Project Structure
```text
ML-GPR-Solar-Tutorial/
├── data/                   <-- Place Kaggle CSV files here
├── output/                 <-- Auto-generated folder for plots
├── main.py                 <-- The clean Python script version
├── GPR_Tutorial.ipynb      <-- The interactive tutorial notebook
├── Tutorial_Final.pdf      <-- The academic report
├── README.md               <-- You are here
├── requirements.txt        <-- Dependencies
└── LICENSE                 <-- MIT License
```

## 📊 Visual Results
The tutorial generates two primary comparative visualizations (saved in the `/output` folder):

1. **Model Comparison:** Demonstrates why a Composite Kernel ($RBF + WhiteKernel$) is necessary to handle noisy solar data compared to basic kernels.
2. **Final Forecast:** A professional-grade plot showing the mean prediction and the 95% Confidence Interval.

## 🛠️ Installation & Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/[Your-Username]/ML-GPR-Solar-Tutorial.git
   cd ML-GPR-Solar-Tutorial
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Data:**
   Download the [Solar Power Generation Data](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data) and place the CSV files into the `/data` folder.

## ♿ Accessibility & Inclusive Design
In compliance with the assignment rubric, this project emphasizes inclusivity:
- **High-Contrast Plots:** We use a Blue/Gray/Red palette optimized for color-blind users (Protanopia/Deuteranopia).
- **Clear Visual Coding:** We use distinct markers (dots) for raw data and solid lines for predictions, ensuring the graph is interpretable in black-and-white.
- **Human-Readable Code:** Comments are written in simple, clear English to ensure the tutorial is a functional teaching tool for everyone.

## ⚖️ Ethical AI: Sustainability
By choosing GPR (a "Shallow Learner") over massive Deep Learning models, we reduce the carbon footprint of our AI training process. This supports **Green AI** practices by providing a mathematically efficient solution for localized renewable energy tasks.

## 📚 References
1. **Rasmussen, C. E., & Williams, C. K. I. (2006).** *Gaussian Processes for Machine Learning*. MIT Press.
2. **Pedregosa, F., et al. (2011).** *Scikit-learn: Machine Learning in Python*. JMLR.
3. **Kaggle Dataset:** *Solar Power Generation Data* by Anivind Kanwal.

## 📄 License
This project is licensed under the **MIT License**.