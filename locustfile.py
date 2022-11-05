from locust import task
from locust import between
from locust import HttpUser


sample = {
  "id": 49833,
  "gender": "Female",
  "age": 42.0,
  "hypertension": 0,
  "heart_disease": 0,
  "ever_married": "Yes",
  "work_type": "Govt_job",
  "residence_type": "Rural",
  "avg_glucose_level": 112.98,
  "bmi": 37.2,
  "smoking_status": "formerly smoked"
}

class CreditRiskTestUser(HttpUser):
    """
    Usage:
        Start locust load testing client with:

            locust -H http://localhost:3000

        Open browser at http://0.0.0.0:8089, adjust desired number of users and spawn
        rate for the load test from the Web UI and start swarming.
    """

    @task
    def classify(self):
        self.client.post("/classify", json=sample)

    wait_time = between(0.01, 2)