# Iris Flower Classification  

A machine learning project focused on identifying the species of flowers based on observed characteristics of sepals and petals. The project aims to experiment with different classification methods and evaluate the performance of the generated models.  

## 📊 Project Details  

- **Objective:**  
  The main objective of this project is to design and analyze machine learning models to correctly classify the species of flowers based on their sepal and petal characteristics. This includes data preprocessing, experimenting with classification methods, and evaluating model performance.  

- **Context:**  
  The project uses the well-known **Iris dataset**, originally collected by Ronald A. Fisher in 1936. The dataset contains information about three species of Iris flowers:  
  - *Iris setosa*  
  - *Iris versicolor*  
  - *Iris virginica*  

  The dataset consists of 150 samples, with 50 samples per species, each having the following four features:  
  - **SepalLengthCm:** Sepal length  
  - **SepalWidthCm:** Sepal width  
  - **PetalLengthCm:** Petal length  
  - **PetalWidthCm:** Petal width  

  The target variable in the dataset is **Species**, which indicates the type of flower.  

### ⚙️ Key Techniques Used

In this Iris flower classification project, several data processing and machine learning techniques were employed to optimize the accuracy of classifying the flower species. Below are the main approaches used:

#### 1. **Loading and Preparing Data**
   - The **Iris** dataset was loaded and stored in a DataFrame.
   - Categorical variables (flower species) were encoded using **LabelEncoder** to transform them into numerical values.
   - Data normalization was performed using **MinMaxScaler** to ensure that input features (such as petal and sepal length and width) were on the same scale.

#### 2. **Splitting Data into Training and Test Sets**
   - The dataset was split into 75% for training and 25% for testing using the `train_test_split` function from `sklearn`.
   - This split ensures that the model is evaluated fairly using data that it hasn’t seen during training.

#### 3. **Training Models with Different Algorithms**
   Several machine learning algorithms were tested to find the best model for classifying the flowers. Below are the key models used:

   - **Random Forest (RF):**
     - **RandomizedSearchCV** was used to optimize hyperparameters such as the number of trees, the maximum depth of the trees, and the minimum number of samples required to split a node.
   
   - **Support Vector Machine (SVM):**
     - **SVC** was employed with a random search over hyperparameters like the kernel type and the `C` parameter, which controls model regularization.

   - **Naive Bayes:**
     - **GaussianNB** was used, adjusting the variance smoothing and the initial class probabilities.

   - **K-Neighbors Classifier (KNN):**
     - **KNeighborsClassifier** was fine-tuned with parameters like the number of neighbors and the distance function, also performing hyperparameter search using **RandomizedSearchCV**.

   - **XGBoost (Extreme Gradient Boosting):**
     - **XGBClassifier** was used with optimization for hyperparameters like the number of trees (`n_estimators`), the maximum tree depth, and the learning rate.

#### 4. **Model Evaluation**
   After training, the model was evaluated using the following techniques:

   - **Evaluation Metrics:**
     - **Accuracy:** Measures the proportion of correct predictions.
     - **Precision:** Measures the accuracy of positive predictions.
     - **Recall:** Measures the model's ability to identify all positive instances.
     - **F1 Score:** Combines precision and recall into a balanced metric.

   - **Confusion Matrix:**
     - A confusion matrix was generated to visualize model performance, showing the number of correct and incorrect predictions for each class.

#### 5. **Hyperparameter Tuning**
   The **RandomizedSearchCV** process was used across all models for hyperparameter tuning. This allowed for finding the best parameter combinations, resulting in a more robust and accurate model.

#### 6. **Final Training**
   After choosing the best model, final training was performed and the model was fine-tuned with the best-found parameters.

### Important Functions

- **`apresentando_metricas`:** Calculates and displays metrics like **accuracy**, **precision**, **recall**, and **f1 score**.
- **`gera_matriz_confusao`:** Generates and displays a confusion matrix to evaluate model performance.
- **`Modelo`:** A class that encapsulates the entire process of data loading, preprocessing, model training, evaluation, and hyperparameter tuning.

This set of techniques formed the foundation for building effective and well-evaluated machine learning models for Iris flower classification.

 
## 🛠️ Repository Structure  

The project follows this organized folder structure:  

```
├── specification.pdf    # Project informations
├── iris_project.ipynb   # Jupyter Notebooks with exploratory analysis,pre-processing, modeling, testing and results  
├── iris_data            # Data used to analysis  
├── main.py              # Main code with the model and tests
├── README.md            # Project documentation  
```  

## 🚀 Results and Insights  

### Model Performance

All models in this project were trained using **cross-validation** and **hyperparameter tuning** techniques to find the best-performing configurations. Given that the Iris dataset consists of only 150 samples, the models were at a higher risk of **overfitting**—this is when a model performs exceptionally well on the training data but fails to generalize to unseen data.

The following points summarize the insights derived from model evaluation:

1. **Overfitting Concern:**
   - Due to the relatively small dataset, most models achieved **100% accuracy** on the training data. However, this is likely an indicator of **overfitting**, where the models simply memorized the data instead of learning the underlying patterns.
   - While the high accuracy is impressive, it doesn't necessarily reflect the model's real-world performance since overfitting tends to result in poor generalization.

2. **Data Limitation:**
   - The Iris dataset consists of only 150 instances (50 samples for each flower species). The small size of the dataset makes it harder to assess the true quality of each model.
   - Models with complex structures, like **Random Forest** or **XGBoost**, tend to perform exceedingly well on small datasets by memorizing data patterns. However, this behavior is not always desirable for deployment in real-world applications where new, unseen data is expected.

3. **Strategies for Improvement:**
   - **Increasing Data Size:** A straightforward approach to reduce overfitting would be to gather more data. By increasing the number of flower samples, the model would have more examples to learn from, helping it generalize better to new data.
   - **Data Augmentation:** Another viable alternative is to create synthetic data samples using **data augmentation** techniques. This would artificially expand the dataset by generating variations of the existing data, giving the models more diversity to learn from.
   
4. **Evaluation Results:**
   - Given the overfitting observed in the results (with 100% accuracy), it is challenging to fully assess the models' performance with the current dataset. The models likely performed well during training, but their ability to generalize to new, unseen data remains uncertain.
   - Additional testing with a larger and more diverse dataset would be needed to evaluate the models’ true capabilities.

### Conclusion

The models developed in this project demonstrated high performance, but this could be due to the limitations of the dataset, leading to overfitting. Future work will include collecting more data or using techniques like data augmentation to create a more robust model capable of generalizing better to unseen data. This will ensure the models perform well not only on training data but also in real-world scenarios.

## 📁 Dataset  

- **Data Source:**  
  The dataset used is the publicly available *Iris Dataset*, which can be accessed [here](https://archive.ics.uci.edu/ml/datasets/Iris).  

## 🤷‍♂️ How to Use  

To run the project locally, follow these steps:  

1. Clone the repository:  
   ```bash  
   git clone https://github.com/GabrielIamato/iris_classification_model.git
   ```  

2. Install the required dependencies:  
   ```bash  
   pip install pandas scikit-learn xgboost matplotlib seaborn numpy  
   ```  

3. Run the main script to train and test models:  
   ```bash  
   python main.py  
   ```  

## 📈 Technologies Used  

The project utilizes the following technologies and libraries:  

- **Programming Language:** Python  
- **Libraries:**  
  - `pandas` – Data manipulation and analysis  
  - `numpy` – Numerical computing and array operations  
  - `matplotlib` and `seaborn` – Data visualization and statistical plotting  
  - `scikit-learn` – Machine learning algorithms, preprocessing, and model evaluation  
  - `xgboost` – Gradient boosting algorithms for classification and regression  
  - `pprint` – Pretty-printing Python data structures  
  - `randomizedsearchcv` – Hyperparameter tuning for machine learning models  
 
## 🤝 Contributions  

Contributions are welcome! To contribute, follow these steps:  

1. Fork the repository  
2. Create a new branch for your feature (`git checkout -b my-feature`)  
3. Commit your changes and push to the branch  
4. Submit a Pull Request  

## 📄 License  

This project is licensed under the GNU General Public License v3.0, published by the Free Software Foundation.
