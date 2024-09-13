# Telecom-Customer-Churn-Prediction:
This project uses machine learning to predict customer churn in the telecom industry. Churn means when customers leave a service provider. The model helps telecom companies identify which customers are likely to leave, allowing them to take actions to retain those customers.

# Who Is This Model For?:
- Telecom companies looking to reduce customer churn.
- Data analysts and data scientists exploring machine learning for customer behavior.

# How It Helps:
- Predicts customers likely to churn.
- Helps businesses design targeted retention strategies.

# Key Files:
- telecom_customer_churn.csv: The original dataset used for training.

- processed_data.csv:
    - This is the modified version of your original dataset (telecom_customer_churn.csv).
    - Changes made include:
	    ~ Missing values were replaced with 0.
	    ~ Categorical columns (like Gender, City, etc.) were converted to numerical values using label encoding.
		      why : Converting categorical columns (like Gender or City) to numerical values is necessary because most machine learning algorithms cannot process text                   data directly.
                These algorithms require numerical inputs to perform mathematical operations. By converting categories to numbers, the model can better understand                   relationships between features.
		
		              For example:
                  -Gender can be encoded as 0 for Male and 1 for Female.
		              -City can be converted to unique numbers for each city.

    - Selected numerical columns (Monthly Charge, Total Charges) were scaled using standardization to ensure all features are on a similar scale. This file               represents the cleaned and processed data used for model training.

- finalized_model.sav:
    - This file contains the trained machine learning model (Random Forest Classifier) after it was fitted on the processed dataset.
    - You can load this model in the future to make predictions on new data without retraining the model.
    - finalized_model.sav: The trained Random Forest model ready for use.

# Dataset Description: 
- This project uses the 'Telecom Customer Churn dataset', which contains information about customer demographics, services subscribed, and usage patterns. The dataset includes columns like Customer ID, Gender, Age, Monthly Charges, Contract Type, and whether the customer churned or not.
- The model is trained to predict customer churn using both numeric and categorical data. Categorical columns (e.g., Gender, Contract Type) are encoded numerically to allow the model to process them effectively.

# Contributing
Feel free to fork the repository and create pull requests. Issues and improvements are welcome.
