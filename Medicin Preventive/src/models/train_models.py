"""
Model Traning Script for Disease Prediction

This script is responsible for training two machine learning models, 
namely Decision Tree and Random Forest, on a preprocessed dataset for 
disease prediction. It evaluates both models on a validation set and 
optionally saves the trained models as `.pkl` files for later use.
"""

import argparse

import yaml #type: ignore
import joblib #type: ignore
import pandas as pd #type: ignore
from sklearn.metrics import accuracy_score #type: ignore
from sklearn.tree import DecisionTreeClassifier #type: ignore
from sklearn.ensemble import RandomForestClassifier #type: ignore
from sklearn.linear_model import LogisticRegression #type: ignore
from sklearn.model_selection import train_test_split #type: ignore

def evalute_model(y_true, y_pred):
    """
        Evaluation du modèle à travers des métrics comme accuracy score
    """
    norm_accuracy = accuracy_score(y_true, y_pred)
    count_accuracy = accuracy_score(y_true, y_pred, normalize=False)
    print(f"Accuracy score : {norm_accuracy}")
    print(f"Validation Score (count): {count_accuracy}")


def train_models(dump_models: bool = True) -> bool:
    """
        Entrainement des modèles avec Random forest et Decision Tree
    """
    try:
        with open('config.yaml', 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)

        # Load and preprocess the dataset
        df = pd.read_csv(config['processed']['training'])

        # Split the dataset into training and validation sets
        x_train, x_val, y_train, y_val = train_test_split(
            df[df.columns[:-1]], df['prognosis'], test_size=0.2, random_state=42
        )

        # Initialize the models
        dtc = DecisionTreeClassifier(random_state=42)
        rfc = RandomForestClassifier(random_state=42)
        lgr = LogisticRegression()  

        # Fit the models
        dtc.fit(x_train, y_train)
        rfc.fit(x_train, y_train)
        lgr.fit(x_train, y_train)

        # Evaluate the models
        print("\nDecision Tree Model:")
        y_pred = dtc.predict(x_val)
        evalute_model(y_val, y_pred)

        print("\nRandom Forest Model:")
        y_pred = rfc.predict(x_val)
        evalute_model(y_val, y_pred)

        print("\nLogistic Reg")

        # Save the models to disk if required
        if dump_models:
            joblib.dump(dtc, config['models']['dtm'])
            print('\nDecision tree model saved to /models directory.')
            joblib.dump(rfc, config['models']['rfm'])
            print('Random forest model saved to /models directory.')

        return True

    except (FileNotFoundError, ValueError, KeyError) as e:
        print("(train_models)", e)
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and optionally save the model.')
    parser.add_argument('--dump-models', action='store_true', help='Save the model after training!')
    train_models(dump_models=parser.parse_args().dump_models)
