# 🍷 Wine Dataset Classification with Neural Networks

## 📌 Overview
This project uses the **UCI Wine Dataset** to perform **multi-class classification** using a **Neural Network** built with TensorFlow/Keras.  
The dataset contains chemical analysis results of wines from three different cultivars (classes).

The goal is to train a model that can classify wine samples into **Class 0**, **Class 1**, or **Class 2** based on 13 numerical features.

---

## 📊 Dataset
- **Source**: UCI Machine Learning Repository (built into `scikit-learn`)
- **Samples**: 178
- **Features**: 13 numerical (chemical composition measurements)
- **Classes**: 3 wine cultivars
- **Type**: Multi-class classification

---

## 🛠️ Steps
1. **Load Dataset**
   - Use `sklearn.datasets.load_wine()` to load data and labels.
   
2. **Split Data**
   - Use `train_test_split()` to divide into training (80%) and test (20%) sets.

3. **Preprocess**
   - Standardize features with `StandardScaler`.
   - One-hot encode target labels using `to_categorical`.

4. **Build Model**
   - Input layer: 13 features  
   - Hidden layers: Dense layers with ReLU activation  
   - Dropout for regularization  
   - Output layer: Softmax activation (3 classes)

5. **Compile Model**
   - Loss: `categorical_crossentropy`
   - Optimizer: `adam`
   - Metric: `accuracy`

6. **Train**
   - Use validation split (20%) to monitor overfitting.
   - Apply `EarlyStopping` to stop training when validation loss stops improving.

7. **Evaluate**
   - Test accuracy is printed on unseen data.

---

## 📦 Requirements
Install dependencies:
```bash
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn
