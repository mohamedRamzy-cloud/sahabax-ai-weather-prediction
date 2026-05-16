
import pandas as pd

from src.model_io import load_model, predict


def main():

    model_dict = load_model()

    # Only provide important features
    sample = pd.DataFrame([
        {
            "Humidity3pm": 55.0,
            "Humidity9am": 80.0,
            "Rainfall": 5.0,
            "Sunshine": 8.0,
            "Cloud3pm": 4.0,
            "Cloud9am": 5.0,
            "Pressure9am": 1012.0,
            "WindGustSpeed": 40.0,
            "Temp3pm": 22.0,
            "RainToday": 0.0,
        }
    ])

    label, probability = predict(model_dict, sample)

    print("\n" + "=" * 40)
    print(f"Prediction  : {label}")
    print(f"Probability : {probability:.2%}")
    print("=" * 40)


if __name__ == "__main__":
    main()