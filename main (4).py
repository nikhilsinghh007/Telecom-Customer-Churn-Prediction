# Import of Required Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from google.colab import files
import pickle

# Load the Dataset
uploaded = files.upload()  # Upload your dataset (CSV file)
df = pd.read_csv('telecom_customer_churn.csv')  # Replace with your dataset name
df.head()

# Explore the Data
df.info()  # Get the structure of the data
df.describe()  # Get summary statistics
df.isnull().sum()  # Check for missing values

# Clean the Data
df.fillna(0, inplace=True)  # Replace missing values with 0

# Convert categorical columns to numerical
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Scaling selected features
scaler = StandardScaler()
df[['Monthly Charge', 'Total Charges']] = scaler.fit_transform(df[['Monthly Charge', 'Total Charges']])

# Visualize the Data
sns.pairplot(df)
plt.show()

# Heatmap for correlation of numeric columns
numeric_df = df.select_dtypes(include=[float, int])
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Prepare Data for Machine Learning
X = df.drop(['Churn Category'], axis=1)  # Features
y = df['Churn Category']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Machine Learning Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the Model
accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the Processed Data
df.to_csv('processed_data.csv')
files.download('processed_data.csv')

# Save the trained model
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

# To load the saved model in the future
loaded_model = pickle.load(open(filename, 'rb'))
