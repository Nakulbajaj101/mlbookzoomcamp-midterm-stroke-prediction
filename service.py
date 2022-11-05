import os

import bentoml
import pandas as pd
from bentoml.io import JSON
from pydantic import BaseModel


MODEL_NAME = os.getenv("MODEL_NAME", "stroke_detection_model")
SERVICE_NAME = os.getenv("SERVICE_NAME", "stroke_detection_classifier")
class StrokeServiceData(BaseModel):

    id: int
    gender : str
    age : float
    hypertension: int
    heart_disease : int
    ever_married: str
    work_type: str
    residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str



model_ref = bentoml.xgboost.get(
    tag_like=f"{MODEL_NAME}:latest"
)

preprocessor = model_ref.custom_objects["preprocessor"]
transformer = model_ref.custom_objects["transformer"]

runner = model_ref.to_runner()

svc = bentoml.Service(
    name=f"{SERVICE_NAME}",
    runners=[runner]
)

@svc.api(input=JSON(pydantic_model=StrokeServiceData), output=JSON())
async def classify(raw_request):
    """Function to classify and make stroke prediction"""

    app_data = pd.DataFrame(raw_request, index=[0])
    vector_processed = preprocessor.transform(app_data)
    vector_transformed = transformer.transform(vector_processed.to_dict(orient='records'))

    prediction = await runner.predict_proba.async_run(vector_transformed)
    result = prediction[0][1]

    if result > 0.7:
        return {
            "status": 200,
            "probability_of_stroke": result,
            "stroke_risk": "HIGH"
        }
    elif result > 0.5:
        return {
            "status": 200,
            "probability_of_stroke": result,
            "stroke_risk": "MEDIUM"
        }
    elif result > 0.25:
        return {
            "status": 200,
            "probability_of_stroke": result,
            "stroke_risk": "LOW"
        }
    else:
       return {
            "status": 200,
            "probability_of_stroke": result,
            "stroke_risk": "UNLIKELY"
        }
    