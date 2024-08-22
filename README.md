# Project for Data Scientist Beginner

## Table of Contents
1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Implemented Features](#implemented-features)
4. [Status of Project](#status-of-project)

## Introduction
The objective of this project is to address several tasks typical for a data scientist working with the PTB-XL dataset. As my first project, I have encountered challenges such as noise filtering and anomaly detection in ECG signals. My goal is to demonstrate basic skills and problem-solving abilities with the help of tools like ChatGPT and the StackOverflow community.

## Setup
This project requires the PTB-XL dataset to be installed on your local machine. You can download it from [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/). The code for this project is located in the file named `projekt1.ipynb` within this repository. To ensure the code functions correctly, update the `data_folder_path` variable in the code to match the directory where your dataset is stored.

## Implemented Features
- Loaded the dataset and removed outliers.
- Visualized ECG signals.
- Checked the dataset for any missing values.
- Attempted to classify health conditions based on ECG signals. Unfortunately, this part was unsuccessful due to a technical issue with loading the appropriate records from the dataset.
- Performed noise filtering, ECG signal segmentation, and anomaly detection.
- Extracted features from the signals and their anomalies.
- Compared the effectiveness of various signal detection algorithms.

## Process and Execution
1. The 'remove_outliers' function is designed to filter out outliers from a given signal based on the Interquartile Range (IQR) method. It works in a following way: The function calculates the first quartile (Q1) and the third quartile (Q3) of the signal, which represent the 25th and 75th percentiles, respectively. These quartiles help identify the central portion of the data. Next, the Interquartile Range (IQR) is computed as the difference between Q3 and Q1. The IQR measures the spread of the middle 50% of the data. The function then calculates the lower and upper bounds using the IQR and a specified threshold (default is 1.5). These bounds define the acceptable range for data points:
   - **Lower Bound:** `Q1 - (IQR * threshold)`
   - **Upper Bound:** `Q3 + (IQR * threshold)`
Next step is to create a mask to identify the data points within these bounds. In the end the function returns a cleaned version of the signal, containing only the data points that lie within the calculated bounds.


## Status of Project
The project has been theoretically completed, except for one non-functioning part of the code. For this reason, it may be updated in the near future to fix the malfunctioning code or to add new functionalities. This largely depends on the vision of my supervisor, whom I would like to thank for their time, motivation, and overall support. I also hope to have the opportunity to demonstrate my skills in a much better way, as I do not believe this project showcased my abilities as well as I would have liked.
