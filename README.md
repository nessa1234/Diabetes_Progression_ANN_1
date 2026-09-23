# Diabetes Progression Prediction Using Artificial Neural Network

## Project Overview

This project aims to model the progression of diabetes using an **Artificial Neural Network (ANN)** and the Diabetes dataset available in the Scikit-learn library.

The objective is to understand how different patient-related variables influence diabetes progression and to build a regression model capable of predicting the diabetes disease progression score.

The project covers:

* Data loading
* Data preprocessing
* Missing-value analysis
* Exploratory Data Analysis (EDA)
* Feature normalization
* ANN model building
* Model training
* Model evaluation
* Architecture experimentation
* Performance comparison

---

## Dataset

The **Diabetes dataset from Scikit-learn** was used for this project.

The dataset contains:

* **442 observations**
* **10 independent features**
* **1 continuous target variable**

### Features

The independent variables are:

* `age`
* `sex`
* `bmi`
* `bp`
* `s1`
* `s2`
* `s3`
* `s4`
* `s5`
* `s6`

The target variable represents a quantitative measure of **diabetes disease progression**.

The dataset was loaded using:

```python
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
```

No external dataset download was required.

---

## Dataset Shape

The original feature matrix contains:

```text
(442, 10)
```

The target contains:

```text
(442,)
```

After converting the dataset into a Pandas DataFrame and adding the target variable, the dataset contains **11 columns**.

---

## Data Preprocessing

### Missing Values

The dataset was checked for missing values.

No missing values were found in any of the features or the target variable. Therefore, no missing-value imputation or row removal was required.

### Feature and Target Separation

The dataset was separated into:

```python
X = df.drop('target', axis=1)
y = df['target']
```

---

## Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the distribution of the variables and their relationship with diabetes progression.

The following visualizations were created:

* Distribution of diabetes progression
* Distribution of all independent variables
* Correlation analysis
* Correlation heatmap
* BMI vs Diabetes Progression
* Blood Pressure vs Diabetes Progression
* S5 vs Diabetes Progression

---

## Correlation Analysis

The correlation of each feature with the target was calculated.

The strongest relationships observed were:

| Feature | Correlation with Target |
| ------- | ----------------------: |
| BMI     |                  0.5865 |
| S5      |                  0.5659 |
| BP      |                  0.4415 |
| S4      |                  0.4305 |
| S6      |                  0.3825 |
| S3      |                 -0.3948 |

Among the variables, **BMI and S5 showed the strongest positive correlations with diabetes progression**.

S3 showed a moderately negative relationship with the target.

---

## Train-Test Split

The dataset was divided into training and testing sets using an **80:20 split**.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

This resulted in:

```text
Training observations: 353
Testing observations: 89
```

---

## Feature Normalization

The input features were standardized using `StandardScaler`.

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The scaler was fitted only on the training data and then applied to the testing data to prevent data leakage.

Feature scaling is important for neural networks because it allows the optimization algorithm to work with variables on comparable scales.

---

# Artificial Neural Network

The Artificial Neural Networks were created using Scikit-learn's:

```python
MLPRegressor
```

`MLPRegressor` implements a feed-forward multilayer perceptron neural network for regression problems.

---

## Basic ANN Model

The first ANN architecture contained two hidden layers.

### Architecture

```text
10 Input Features
        ↓
32 Neurons
   ReLU
        ↓
16 Neurons
   ReLU
        ↓
1 Regression Output
```

The model was defined as:

```python
ann_model = MLPRegressor(
    hidden_layer_sizes=(32, 16),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42
)
```

### Model Configuration

* Hidden Layers: **32 and 16 neurons**
* Activation Function: **ReLU**
* Optimizer: **Adam**
* Maximum Iterations: **1000**
* Problem Type: **Regression**

---

## Basic ANN Performance

The model was evaluated using:

* Mean Squared Error
* Root Mean Squared Error
* Mean Absolute Error
* R² Score

The basic ANN achieved:

| Metric   |      Result |
| -------- | ----------: |
| MSE      | **2775.81** |
| RMSE     |   **52.69** |
| MAE      |   **42.21** |
| R² Score |  **0.4761** |

The R² score of approximately **0.476** indicates that the model explained approximately **47.6% of the variation in diabetes progression within the test dataset**.

An Actual vs Predicted scatter plot was also created to visualize the model's prediction performance.

---

# ANN Architecture Experiment

A second ANN architecture was tested to determine whether increasing network complexity would improve model performance.

The experimental network used three hidden layers.

### Architecture

```text
10 Input Features
        ↓
64 Neurons
   ReLU
        ↓
32 Neurons
   ReLU
        ↓
16 Neurons
   ReLU
        ↓
1 Regression Output
```

The model was created using:

```python
improved_ann = MLPRegressor(
    hidden_layer_sizes=(64, 32, 16),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=2000,
    random_state=42
)
```

Changes made compared with the basic ANN included:

* Increased first hidden layer from 32 to 64 neurons
* Added an additional hidden layer
* Used hidden layers of 64, 32 and 16 neurons
* Increased maximum iterations from 1000 to 2000
* Explicitly specified a learning rate of 0.001

---

## Experimental ANN Performance

The experimental architecture produced:

| Metric   |      Result |
| -------- | ----------: |
| MSE      | **6243.42** |
| RMSE     |   **79.02** |
| MAE      |   **58.05** |
| R² Score | **-0.1784** |

---

## Model Comparison

| Model            |         MSE |      RMSE |       MAE |   R² Score |
| ---------------- | ----------: | --------: | --------: | ---------: |
| Basic ANN        | **2775.81** | **52.69** | **42.21** | **0.4761** |
| Experimental ANN |     6243.42 |     79.02 |     58.05 |    -0.1784 |

For MSE, RMSE and MAE, **lower values indicate better performance**.

For R², **higher values indicate better performance**.

The Basic ANN therefore performed substantially better on the unseen testing data.

The larger experimental ANN did not improve performance. Instead, its prediction errors increased and its R² score became negative.

This experiment demonstrates that increasing the number of neurons and hidden layers does not automatically improve neural-network performance.

Because the Diabetes dataset contains only **442 observations**, the larger architecture may have been unnecessarily complex for the available training data and showed poorer generalization on unseen observations.

---

## Training Observation

Both ANN configurations reached their specified maximum number of iterations before the optimization algorithm reported complete convergence.

The Basic ANN reached:

```text
1000 iterations
```

The Experimental ANN reached:

```text
2000 iterations
```

Further improvements could therefore investigate:

* Early stopping
* Different learning rates
* Regularization
* Different hidden-layer sizes
* Alternative activation functions
* Different convergence tolerance values

---

## Technologies and Libraries

The project was developed using:

* Python
* Jupyter Notebook
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

Important Scikit-learn components used include:

```text
load_diabetes
train_test_split
StandardScaler
MLPRegressor
mean_squared_error
mean_absolute_error
r2_score
```

---

## Conclusion

This project successfully developed and evaluated an Artificial Neural Network for predicting diabetes progression using the Scikit-learn Diabetes dataset.

The dataset contained **442 patient observations and 10 independent variables**, with no missing values.

Exploratory Data Analysis was used to investigate feature distributions and relationships with diabetes progression. BMI and S5 showed some of the strongest positive relationships with the target variable.

The features were standardized using `StandardScaler`, and the dataset was divided into 80% training data and 20% testing data.

The first ANN consisted of hidden layers containing **32 and 16 neurons** and achieved:

```text
MSE  = 2775.81
RMSE = 52.69
MAE  = 42.21
R²   = 0.4761
```

A more complex architecture containing **64, 32 and 16 neurons** was subsequently tested. However, the experimental model produced significantly poorer test performance with an R² score of **-0.1784**.

Therefore, the experiment showed that the simpler ANN architecture generalized better to unseen data than the larger network.

Overall, the project demonstrates a complete machine-learning workflow for ANN-based regression, including preprocessing, exploratory analysis, normalization, neural-network development, training, evaluation and hyperparameter experimentation.

---

## Repository Structure

```text
Diabetes-Progression-ANN/
│
├── Diabetes_Progression_ANN.ipynb
└── README.md
```

---

## How to Run

Open the Jupyter Notebook:

```text
Diabetes_Progression_ANN.ipynb
```

Install the required packages if necessary.

For the browser-based Jupyter/WASM environment used in this project:

```python
!mamba install pandas numpy matplotlib scipy scikit-learn
```

Then run all notebook cells sequentially.

---

## Author

**Nessa Kallikkad Madhu**

Deep Learning Module End Project
Diabetes Progression Prediction Using Artificial Neural Network
