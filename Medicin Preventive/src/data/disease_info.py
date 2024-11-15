import yaml  # type: ignore
import pandas as pd  # type: ignore
from src.utils.wikipedia import wiki_api
from googletrans import Translator  # Importer la bibliothèque de traduction

class DiseaseInfo:
    """A class to provide disease-related information based on a dataset and Wikipedia API."""

    def __init__(self) -> None:
        self.__dinfo = self.load_dataset()
        self.translator = Translator()  # Initialiser le traducteur

    def load_dataset(self) -> pd.DataFrame | None:
        """Load the disease information dataset from the specified path."""
        try:
            with open('config.yaml', encoding='utf-8') as file:
                data_path = yaml.safe_load(file)['metadata']['disease_info']
            return pd.read_csv(data_path).set_index('disease_name')
        except KeyError as e:
            print(f'(dinfo_module) Unknown config key: {e}')
        return None

    def safe_lookup(self, row: str, col: str) -> str | None:
        """Helper method to safely lookup data in the dataset with exception handling."""
        try:
            return self.__dinfo.loc[row, col]  # type: ignore
        except AttributeError:
            print(f"(dinfo-module)[Error -1]: looking up '{col}' for '{row}'")
            return None

    def translate_to_french(self, text: str) -> str:
        """Translate the given text to French using Google Translate."""
        try:
            translated = self.translator.translate(text, src='en', dest='fr')
            return translated.text
        except Exception as e:
            print(f"(dinfo-module)[Error]: Translation failed - {e}")
            return text  # Retourner le texte original en cas d'échec

    def precautions(self, disease_name) -> list[str] | None:
        """Return a list of precautions for the given disease, translated to French if available."""
        precautions_str = self.safe_lookup(disease_name, 'precautions')
        if precautions_str:
            precautions_list = precautions_str.split(',')
            # Traduire chaque précaution en français
            return [self.translate_to_french(precaution) for precaution in precautions_list]
        return None

    def image_link(self, disease_name) -> str | None:
        """Return the image link for the given disease, if available."""
        return self.safe_lookup(disease_name, 'wiki_img')

    def short_summary(self, disease_name: str) -> str | None:
        """Return a short summary of the disease from Wikipedia, translated to French."""
        pageid = self.safe_lookup(disease_name, 'wiki_pageid')
        if pageid is None:
            return None
        try:
            res = wiki_api(pageids=pageid, exintro=True, explaintext=True, exsentences=5)
            if res and res.ok:
                summary = res.json()['query']['pages'][f'{pageid}']['extract']  # type:ignore
                # Traduire le résumé en français
                return self.translate_to_french(summary)
            return None
        except (KeyError, TypeError) as e:
            print('(dinfo-module)[Error -1]: Unable to locate', e)
            return None

if __name__ == "__main__":
    di = DiseaseInfo()
