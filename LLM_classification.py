### OpenAI classification 

import pandas as pd
import string
import openai
import os
from getpass import getpass
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

class TextClassificationPipeline:
    def __init__(self, file_path, model_name="gpt-3.5-turbo"):
        self.file_path = file_path
        self.model_name = model_name
        self.df = None
        self.api_key = None
        self._setup_api_key()

    def _setup_api_key(self):
        # Fetch the OpenAI API key securely.
        try:
            from google.colab import userdata
            self.api_key = userdata.get('OPENAI_API_KEY')
        except ImportError:
            self.api_key = os.getenv('OPENAI_API_KEY')
            if self.api_key is None:
                self.api_key = getpass("Enter your OpenAI API Key: ")
        openai.api_key = self.api_key

    def load_data(self):
        print("Loading data...")
        self.df = pd.read_excel(self.file_path)
        display(self.df.head())
        display(self.df.info())

    def preprocess_data(self):
        print("Preprocessing data...")
        def remove_punctuation(text):
            """Removes punctuation from a string."""
            if isinstance(text, str):
                return text.translate(str.maketrans('', '', string.punctuation))
            return text 

        self.df['sentence_cleaned'] = self.df['sentence'].apply(remove_punctuation)
        
        # Drop rows with missing values
        initial_len = len(self.df)
        self.df.dropna(inplace=True)
        print(f"Dropped {initial_len - len(self.df)} rows with missing values.")

    def _classify_single_sentence(self, sentence):
        try:
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
              model=self.model_name,
              messages=[
                    {"role": "system", "content": "You are a helpful assistant that classifies sentences as 'descriptive' or 'normative'."},
                    {"role": "user", "content": f"Classify the following sentence as either 'descriptive' or 'normative': {sentence}"}
                ],
              max_completion_tokens=10,
              n=1,
              stop=None,
              temperature=0.5,
            )
            classification = response.choices[0].message.content.strip().lower()
            if 'descriptive' in classification:
                return 'descriptive'
            elif 'normative' in classification:
                return 'normative'
            else:
                return None
        except Exception as e:
            print(f"Error classifying sentence: {e}")
            return None

    def classify_data(self, subset_size=None):
        print(f"Classifying data (subset_size={subset_size})...")
        self.df['predicted_label'] = None
        
        target_df = self.df.head(subset_size) if subset_size else self.df
        
        predictions = []
        for index, row in target_df.iterrows():
            sentence = row['sentence']
            predicted_label = self._classify_single_sentence(sentence)
            predictions.append(predicted_label)
            
        # Assign back to the main dataframe (careful with indices if subsetting)
        if subset_size:
             self.df.loc[target_df.index, 'predicted_label'] = predictions
        else:
             self.df['predicted_label'] = predictions

        display(self.df.head(subset_size if subset_size else 5))

    def evaluate_model(self):
        print("Evaluating model...")
        # Filter for rows where we actually made a prediction
        comparison_subset = self.df.dropna(subset=['predicted_label']).copy()
        
        if len(comparison_subset) == 0:
            print("No predictions found to evaluate.")
            return

        # Ensure labels are consistent strings
        comparison_subset['label'] = comparison_subset['label'].astype(str)
        comparison_subset['predicted_label'] = comparison_subset['predicted_label'].astype(str)

        # Calculate metrics
        accuracy = accuracy_score(comparison_subset['label'], comparison_subset['predicted_label'])
        f1 = f1_score(comparison_subset['label'], comparison_subset['predicted_label'], average='weighted')
        
        print(f"\nAccuracy: {accuracy:.2f}")
        print(f"F1 Score: {f1:.2f}")

        # Confusion Matrix
        cm = confusion_matrix(comparison_subset['label'], comparison_subset['predicted_label'])
        labels = sorted(list(set(comparison_subset['label'].unique()) | set(comparison_subset['predicted_label'].unique())))

        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()

    def run(self, subset_size=5):
        self.load_data()
        self.preprocess_data()
        self.classify_data(subset_size=subset_size)
        self.evaluate_model()

# Usage
if __name__ == "__main__":
    pipeline = TextClassificationPipeline(file_path="sample_data/validation_sample_limited.xlsx")
    pipeline.run(subset_size=5)