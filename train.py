
import argparse
import json
import os
from src.trainer import train_pipeline

RESULTS_PATH = "outputs/benchmark_results.json"

MODEL_DISPLAY = {
    "Logistic Regression": "Logistic Regression  ",
    "SVM":                 "SVM                  ",
    "Naive Bayes":         "Naive Bayes          ",
    "Decision Tree":       "Decision Tree        ",
    "Random Forest":       "Random Forest        ",
    "Gradient Boosting":   "Gradient Boosting    ",
    "XGBoost":             "XGBoost              ",
    "LightGBM":            "LightGBM             ",
    "MLP":                 "MLP (Neural Net)     ",
}


def show_ranking(results: dict):
    print("\n" + "=" * 55)
    print("Benchmark Ranking:")
    print("=" * 55)
    print(f"  {'#':<4} {'Model':<25} {'F1':>6}  {'Accuracy':>8}")
    print("-" * 55)

    for i, row in enumerate(results["ranking"], 1):
        flag = "  <- Best" if i == 1 else ""
        print(
            f"  {i:<4} {row['model']:<25} "
            f"{row['f1']:>6.4f}  {row['accuracy']:>8.4f}{flag}"
        )

    print("=" * 55)


def ask_confirmation(best_name: str, best_f1: float, best_acc: float) -> bool:
    print(f"\nSelected Model : {MODEL_DISPLAY.get(best_name, best_name)}")
    print(f"F1 Score       : {best_f1}")
    print(f"Accuracy       : {best_acc}")
    print()

    while True:
        choice = input(
            "Do you want to train and save this model? [y/n]: "
        ).strip().lower()

        if choice == "y":
            return True

        elif choice == "n":
            return False

        else:
            print("Please enter y or n only.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the best model from benchmark"
    )

    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to CSV dataset"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Force a specific model name"
    )

    parser.add_argument(
        "--k",
        type=int,
        default=15,
        help="Number of features to select"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(RESULTS_PATH):
        print("Benchmark results file is missing.")
        print("Run benchmark first: python benchmark.py\n")
        exit(1)

    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)

    show_ranking(results)

    if args.model:
        best_name = args.model

        matching = [
            r for r in results["ranking"]
            if r["model"] == best_name
        ]

        if not matching:
            print(f"\nModel '{best_name}' was not found in benchmark results.")
            print(
                f"Available models: "
                f"{[r['model'] for r in results['ranking']]}"
            )
            exit(1)

        best_f1 = matching[0]["f1"]
        best_acc = matching[0]["accuracy"]

    else:
        best_name = results["best_model"]
        best_f1 = results["best_f1"]
        best_acc = results["best_accuracy"]

    confirmed = ask_confirmation(best_name, best_f1, best_acc)

    if not confirmed:
        print("\nAvailable Models:")

        for i, row in enumerate(results["ranking"], 1):
            print(f"  {i}. {row['model']}")

        print()

        num = input("Enter model number: ").strip()

        try:
            idx = int(num) - 1

            best_name = results["ranking"][idx]["model"]
            best_f1 = results["ranking"][idx]["f1"]
            best_acc = results["ranking"][idx]["accuracy"]

        except (ValueError, IndexError):
            print("Invalid model number. Operation cancelled.")
            exit(1)

        confirmed = ask_confirmation(best_name, best_f1, best_acc)

        if not confirmed:
            print("\nTraining cancelled.")
            exit(0)

    print(f"\nTraining model: {best_name} ...\n")

    kwargs = {
        "k_features": args.k,
        "best_model_name": best_name
    }

    if args.data:
        kwargs["data_path"] = args.data

    train_pipeline(**kwargs)

    print("\nTraining and saving completed successfully!")
    print("Run prediction using: python predict.py")

