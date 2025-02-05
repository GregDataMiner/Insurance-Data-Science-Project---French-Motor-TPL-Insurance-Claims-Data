# 📊 **Insurance Data Science Project - French Motor TPL Insurance Claims Data**
 This project is an adapted version of Karan Sarpal's project published on Kaggle, where I implement Machine learning and data science tool to evaluate risk profils from a french moto.r dataset.


![Cover Image](Images/Cover.png)

---

## **Background and Context**  
Traditionally, actuaries have used statistical and mathematical methods based on well-established theories to predict:  
- The severity of claims  
- The probability of claims  

These predictions fueled a feedback loop between:  
1. Underwriting  
2. Actuarial science  
3. Claims management  

In recent years, actuarial methods have evolved with the integration of **Data Science (DS)** and **Machine Learning (ML)**, which improve:  
- The accuracy of cost predictions  
- Insurance portfolio management  

Currently, commonly used models include **Generalized Linear Models (GLM)** and **decision trees**, which provide:  
- Better computational efficiency  
- Greater prediction accuracy  

---

## **Project Objective** 🎯  
This project aims to demonstrate the use of the **DS/ML** workflow to:  
- Predict claim costs with accuracy  
- Optimize insurance portfolio management  

This is crucial due to the significant increase in claims, such as those related to business interruptions caused by the **COVID-19** pandemic.

### **Case Study**  
We are using a French **motor third-party liability (MTPL)** insurance dataset to predict claim severity.

---

## **Project Content** 🛠️  

The project includes:  
- 🔄 **Preprocessing** and **encoding** of the MTPL dataset  
- 🧮 **Selection of relevant** risk variables  
- 🤖 **Training** and **testing** of ML regression models  

---

## **Additional Resources** 📚  

- **Swiss Association of Actuaries:** [Case Studies Page](https://www.actuarialdatascience.org/ADS-Tutorials/)  
- **Kaggle Dataset:** [French Motor TPL Insurance Claims](https://www.kaggle.com/datasets/karansarpal/fremtpl-french-motor-tpl-insurance-claims)

---

## **Data Presentation**  

The datasets **freMTPLfreq** and **freMTPLsev** contain risk characteristics for **413,169** motor third-party liability insurance policies, observed over a one-year period.  

### **Key Information**  
- Each policy includes the number of claims.  
- Some amounts in **freMTPLsev** are fixed, based on the French **IRSA-IDA** convention.

---

## **Technical Overview** ⚙️  

### **Models Used**  
- 🌲 **Random Forest Regression**  
- 📈 **Poisson Regression (GLM)**  
- 🔀 **Tweedie Regression**  
- 🚀 **XGBoost Regression**

---

## **Author** ✍️  
Created and maintained by **[@GregDataMiner]**  
