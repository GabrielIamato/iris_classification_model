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

- **Key Techniques Used:**  
 

## 🛠️ Repository Structure  

The project follows this organized folder structure:  

```
├── specification.pdf    # Project informations
├── iris_project.ipynb   # Jupyter Notebooks with exploratory analysis,pre-processing, modeling, testing and results  
├── iris_data            # Data used to analysis  
├── main.py              # 
├── README.md            # Project documentation  
```  

## 🚀 Results and Insights  


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

This project is licensed under the MIT License.  
