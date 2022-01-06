# Project_Python_Data_Analysis
# BENYEMNA-Hamza_HACHEM-Mohamad-ESILV-Python-project-2022

Context:

This project focuses on concerning diabetics in the US from 1999 to 2008. The aim is to settle a Machine Learning model that predicts if a patient has to be readmitted in the hospital (within 30 days, 30 days later or not).

This is a supervised classification problem (3-class).


Description:

We first worked on the dataset to clean it. This helps us to work more easily and efficiently. 
Besides this, we analyzed it through plots and correlation rates between the features and the target. 
After that, we worked on the Machine Learning. We tried 3 models:
 - Naive Bayes
 - Random Forest
 - Gradient Boosting

We made many tests in order to improve the accuracy obtained through different methods but the best one was to work with the Gradient Boosting. We improved it with the tuning model method and a gread search. 
Finally, we obtained an accuracy about 62.5%.


Tasks:

Notebook:
  - load the dataset
  - clean it
  - show plots and analyze it
  - find different Machine Learning models and improve it through tests and different methods.
 
Report: we explained as well as possible the fulfillment of the project.

Flask:
In the Flask part we have <ins>4 essentiels files and 1 folder</ins>
* Main.py
* GradientBoostingModel.pkl
* templates 

These 3 are the most important files to make the flask app work smoothly.
* first we need to set the right enviroment;therefore make sure that all the packages are well installed on your PC.
* second include these file in your project and make sure the "Index.html"(found in Templates that it stays like this)
```
Once you run the application you might run into a portal problem to solve that simply change the portal number from 5000 to an available portal
for you
```
Once the application runs. you will find in the terminal a link press on it and a HTML survey will open where you can
put your input and receive the prediction directly. 