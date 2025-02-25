import gradio as gr
import os
import pandas as pd
import lightgbm as lgb
from datasets import load_dataset
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ------------------ Kaggle Authentication ------------------ #
api = KaggleApi()
api.authenticate()

# Download dataset from Kaggle (replace <dataset-path>)
os.system("kaggle datasets download -d <dataset-path>")  # Replace <dataset-path>

# Extract the dataset (replace <dataset-name>)
os.system("unzip ./<dataset-name>.zip -d ./data/")  # Replace <dataset-name>

# ------------------ Load Datasets ------------------ #
# Load Hugging Face dataset (replace 'dataset_name')
hf_dataset = load_dataset('dataset_name')  # Replace with actual dataset
hf_df = pd.DataFrame(hf_dataset['train'])  # Convert to pandas DataFrame

# Load Kaggle dataset
kaggle_df = pd.read_csv('./data/kaggle_dataset.csv')  # Replace with actual Kaggle dataset path

# Merge datasets
merged_df = pd.concat([hf_df, kaggle_df], ignore_index=True)

# ------------------ LightGBM Model Setup ------------------ #
# Example: Assuming the last column is the target
X = merged_df.iloc[:, :-1]  # Features
y = merged_df.iloc[:, -1]   # Target

# Handle missing values if any
X.fillna(0, inplace=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Train LightGBM model
params = {
    'objective': 'binary',   # Change if it's multi-class or regression
    'metric': 'binary_error',  # Adjust metric based on the task
    'boosting_type': 'gbdt',
    'verbose': -1
}

model = lgb.train(params, train_data, num_boost_round=100, valid_sets=[test_data], early_stopping_rounds=10)

# ------------------ Gradio Interface ------------------ #
# Function to display merged data
def display_data():
    return merged_df.head()

# Function to make predictions
def predict(input_data):
    input_df = pd.DataFrame([input_data], columns=X.columns)
    prediction = model.predict(input_df)
    # Convert probability to binary (0/1) if needed
    return "Positive" if prediction[0] > 0.5 else "Negative"

# Gradio Interface for Data Display and Prediction
with gr.Blocks() as iface:
    gr.Markdown("### Merged Dataset")
    data_display = gr.Dataframe(value=merged_df.head(), interactive=False)

    gr.Markdown("### LightGBM Prediction")
    inputs = [gr.Textbox(label=col) for col in X.columns]
    predict_button = gr.Button("Predict")
    output = gr.Textbox(label="Prediction Result")

    predict_button.click(fn=predict, inputs=inputs, outputs=output)

# ------------------ Launch Gradio App ------------------ #
iface.launch()