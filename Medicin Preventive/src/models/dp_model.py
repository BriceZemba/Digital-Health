import yaml #type: ignore
import joblib #type: ignore
import pandas as pd #type: ignore
from scipy import stats #type: ignore

class DiseasePredictModel():

    def __init__(self) -> None:
        self.__config = self.load_config()
        self.__symptoms, self.__diseases = self.load_labels()
        self.__dt_model, self.__rf_model = self.load_models()

    def load_config(self) -> dict:
            with open('config.yaml', 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)

    def load_models(self):
        try:
            with open(self.__config['models']['dtm'], 'rb') as file:
                dt_model = joblib.load(file)
            with open(self.__config['models']['rfm'], 'rb') as file:
                rf_model = joblib.load(file)
            return dt_model, rf_model
        except OSError as e:
            print("(dp_module)", e)
        except KeyError as e:
            print(f"(dp_model) Unknown config key: {e}")
        return None, None

    def load_labels(self) -> tuple[tuple[str], tuple[str]]:
        try:
            df = pd.read_csv(self.__config['metadata']['labels'])
            return tuple(df['symptom_name'].tolist()), tuple(df['disease_name'].tolist())
        except FileNotFoundError as e:
            print("(dp_module)", e)
        except KeyError as e:
            print(f"(dp_model) Unknown label key: {e}")
        return tuple(), tuple()

    def validate_models(self) -> bool:
        if not self.__dt_model or not self.__rf_model:
            print("(dp_module) Modéle ne sont chargé correctement")
            return False
        if not self.__symptoms or not self.__diseases:
            print("(dp_module) Data labels are not loaded properly.")
            return False
        return True

    def symptoms_to_df(self, psymptoms: list[str]) -> pd.DataFrame:
        binary_vector = [1 if symp in psymptoms else 0 for symp in self.__symptoms]
        return pd.DataFrame([binary_vector], columns=self.__symptoms)

    @property
    def symptoms(self) -> tuple[str]:
        """Symptom labels tuple read-only."""
        return self.__symptoms

    @property
    def diseases(self) -> tuple[str]:
        """Disease labels tuple read-only."""
        return self.__diseases

    def predict(self, symptoms_names: list[str]) -> int | None:
        if not self.validate_models():
            return None

        x_pred = self.symptoms_to_df(symptoms_names)
        dt_pred = self.__dt_model.predict(x_pred) # type:ignore
        rf_pred = self.__rf_model.predict(x_pred) # type:ignore
        return stats.mode([dt_pred, rf_pred], keepdims=False)[0][0]

    def predict_proba(self, symptoms_names: list[str]) -> list[float] | None:
        """
        Predire la probabilité de chaque maladie à travers ces symptômes
        """
        if not self.validate_models():
            return None

        x_pred = self.symptoms_to_df(symptoms_names)
        dt_proba = self.__dt_model.predict_proba(x_pred) # type:ignore
        rf_proba = self.__rf_model.predict_proba(x_pred) # type:ignore
        return ((dt_proba + rf_proba) / 2)[0]

if __name__ == "__main__":
    dpm = DiseasePredictModel()
