

#  Glucose & Xylose Yield Prediction using ANN

This project builds and trains an **Artificial Neural Network (ANN)** model using TensorFlow to predict the **glucose and xylose yields** from biomass based on experimental conditions like **biomass loading**, **particle size**, and **reaction time**.
The algorithm aims to replicate the study 
##  Input

* **Dataset**: `synthetic_data.csv`
  A CSV file with the following columns:

  * `biomass_loading`
  * `particle_size`
  * `time`
  * `glucose` (target)
  * `xylose` (target)

## Model Overview

* **Architecture**:

  * Input layer: 3 neurons
  * Hidden layer: 8 neurons, `sigmoid` activation
  * Output layer: 2 neurons (`glucose` & `xylose`), `linear` activation
* **Loss Function**: Mean Squared Error (MSE)
* **Optimizer**: Adam
* **Epochs**: 1000

# Workflow

1. **Data Normalization**: Using MinMaxScaler for both inputs and outputs.
2. **Train-Test Split**: 70% training, 15% validation, 15% testing.
3. **Model Training**: Trained on scaled inputs/outputs.
4. **Evaluation**:

   * Mean Squared Error (MSE)
   * R² score for glucose and xylose
     
5. **Visualization**:

   * Plots training and validation loss curves across epochs.

## 📊 Output

* **Performance Metrics** :

  * Test MSE
  * R² for glucose and xylose
* **Loss Curve Plot**: Visualization of  training and validation loss over epochs.

# Requirements

* Python ≥ 3.7
* `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `tensorflow`

# Use Case

Useful for:

* Bioethanol process modeling
* Bioprocess optimization
* Predictive analytics for sugar yield from biomass.

