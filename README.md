# KG-Fraud-Detection
## This repository contains my work for my Bachelor's thesis. The main idea behind it to create a Fraud detection (FD) system utilizing a Knowledge graph (KG) and utilize Machine Learning to further improve the results. ##

**Components:**

1) thesisTradML.ipynb, which contains the inital experiments done with simple ML algorithms without the KG or ensemble learning. This is purely a first attempt to compare it with the results of the finalized model.
2) Main.py the main function to run. There are several available ML algorithms which can be used. For this some manual changes need to be made. More specifically on lines 128 & 131, the string parameter need to change following the form "clfx" with x being a number from 1 to 7 depending on which ML algorithm you want to execute. The algorithms are all initialized on the dictionary above for more info refer to it!
3) Train_Test.py this contains all the necessary functions for the ML training and testing.
