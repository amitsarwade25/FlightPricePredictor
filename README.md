# Flight Fare Intelligence Tool

A predictive analytics tool that estimates flight prices using historical pricing data and a **Linear Regression** model built with **Scikit-learn**.

---

# Overview

This project analyzes historical flight pricing data including **travel date**, **day of week**, **number of stops**, **flight duration**, and **airline** to understand fare patterns and predict prices for new flight scenarios.

It demonstrates a complete, lightweight machine learning workflow including:

- Data loading
- Data preprocessing
- Model training
- Model evaluation
- Reusable predictions
- Visualization of actual vs. predicted prices

---

# Features

- **Data-driven fare analysis** – Loads and processes historical flight pricing data from CSV.
- **Linear Regression model** – Uses Scikit-learn to predict ticket prices.
- **Train-test evaluation** – Uses an **80/20 split** with **Mean Squared Error (MSE)**.
- **Reusable prediction function** – Predict flight prices using flight details.
- **Visualization** – Compares actual historical prices with model predictions.

---

# Tech Stack

- **Python**
- **Scikit-learn** – Model training and evaluation
- **Matplotlib** – Data visualization

---

# Dataset

The dataset **`flight_prices_extended.csv`** contains the following columns:

| Column | Description |
|---------|-------------|
| `date` | Travel date |
| `day_of_week` | Day of the week (0–6) |
| `num_stops` | Number of flight stops |
| `duration` | Flight duration (hours) |
| `airline` | Airline (0 or 1) |
| `price` | Ticket price (Target Variable) |

> **Note:** This is a demo-scale dataset created to demonstrate the machine learning workflow and is **not** scraped from a live flight pricing source.

---

# Installation & Setup

## Clone the Repository

```bash
git clone https://github.com/amitsarwade25/flight-fare-intelligence.git
cd flight-fare-intelligence
```

## Install Dependencies

```bash
pip install pandas scikit-learn matplotlib numpy
```

## Add the Dataset

Place **`flight_prices_extended.csv`** in the project root directory (same folder as the Python script).

## Run the Project

```bash
python flight_price_prediction.py
```

The program will:

- Train the machine learning model
- Print the **Mean Squared Error (MSE)**
- Predict an example flight price
- Display a graph comparing actual and predicted prices

---

# ⚙️ How It Works

1. **Load and prepare data** from the CSV file.
2. **Select features** (`day_of_week`, `num_stops`, `duration`, `airline`) and the target (`price`).
3. **Split the dataset** into training and testing sets (80/20).
4. **Train** a Linear Regression model.
5. **Evaluate** performance using **Mean Squared Error (MSE)**.
6. **Predict** ticket prices using the `predict_price()` function.
7. **Visualize** actual vs. predicted prices using Matplotlib.

---

by - Amit
