import os

import bentoml
import pandas as pd
from bentoml.io import JSON
from pydantic import BaseModel


MODEL_NAME = os.getenv("MODEL_NAME", "stroke_detection_model")
SERVICE_NAME = os.getenv("SERVICE_NAME", "stroke_detection_classifier")
class StrokeServiceData(BaseModel):

    id: int = 49833
    gender : str = "Female"
    age : float = 42.0
    hypertension: int = 1
    heart_disease : int = 1
    ever_married: str = 0
    work_type: str = "Govt_job"
    residence_type: str = "Rural"
    avg_glucose_level: float = 112.98
    bmi: float = 37.2
    smoking_status: str = "formerly smoked"


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

    app_data = pd.DataFrame(raw_request.dict(), index=[0])
    vector_processed = preprocessor.transform(app_data)
    vector_transformed = transformer.transform(vector_processed.to_dict(orient='records'))

    prediction = await runner.predict_proba.async_run(vector_transformed)
    result = round(prediction[0][1],3)

    if result > 0.5:
        return {
            "status": 200,
            "probability_of_stroke": result,
            "stroke_risk": "HIGH"
        }
    elif result > 0.3:
        return {
            "status": 200,
            "probability_of_stroke": result,
            "stroke_risk": "MEDIUM"
        }
    elif result > 0.2:
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
    