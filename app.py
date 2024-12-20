import gradio as gr
import os
import pandas as pd
from datasets import load_dataset
from kaggle.api.kaggle_api_extended import KaggleApi

# Kaggle authentication
api = KaggleApi()
api.authenticate()

# Download dataset from Kaggle (example dataset)
os.system("kaggle datasets download -d <dataset-path>")  # Replace <dataset-path>

# Extract the dataset (if needed)
os.system("unzip ./<dataset-name>.zip -d ./data/")  # Replace <dataset-name> with actual dataset name

# Load the Hugging Face dataset
hf_dataset = load_dataset('dataset_name')  # Replace 'dataset_name' with the name of the dataset
hf_df = pd.DataFrame(hf_dataset['train'])  # Convert to pandas DataFrame

# Load Kaggle dataset
kaggle_df = pd.read_csv('./data/kaggle_dataset.csv')  # Replace with the appropriate path

# Merge datasets
merged_df = pd.concat([hf_df, kaggle_df], ignore_index=True)

# Example function to display merged data (this can be modified based on your needs)
def display_data():
    return merged_df.head()

# Create Gradio interface to display data
iface = gr.Interface(fn=display_data, inputs=[], outputs="dataframe")

# Launch the Gradio app
iface.launch()
