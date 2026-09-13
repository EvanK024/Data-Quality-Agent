import json
import joblib
import pandas as pd
import os

model = None


def init():
    global model

    model_path = os.path.join(
        os.environ["AZUREML_MODEL_DIR"],
        "data_quality_agent_model.pkl"
    )

    model = joblib.load(model_path)


def run(raw_data):
    try:
        data = json.loads(raw_data)

        if isinstance(data, dict):
            data = [data]

        df = pd.DataFrame(data)

        probabilities = model.predict_proba(df)[:, 1]

        threshold = 0.70
        predictions = (probabilities >= threshold).astype(int)

        results = []

        for prediction, probability in zip(predictions, probabilities):
            results.append({
                "needs_review": int(prediction),
                "review_probability": round(float(probability), 4),
                "decision": (
                    "REVIEW REQUIRED"
                    if prediction == 1
                    else "NO REVIEW REQUIRED"
                )
            })

        return results

    except Exception as e:
        return {
            "error": str(e)
        }
