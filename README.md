# Stroke Prediction Service
The prupose of this project is to provide a stroke prediction service, which could be used by GPs, doctors and health providers to understand the risk of stroke. By picking early indication of risk, health service providers can provide intervention and help patients
understand the risk. They can further work with the patients to design a plan or provide a prescription to reduce the risk of stroke

The service will be provided as an API, and the IT teams at Health providers can use it to support web portals and forms. Where the health providers don't have IT to support them, service will be provided as a streamlit app

[StreamlitApp]![alt text](https://github.com/Nakulbajaj101/mlbookzoomcamp-midterm-stroke-prediction/blob/main/images/StreamlitStrokePrediction.png)

# Motivation 
Amongst the major cardiovascular diseases, stroke is one of the most dangerous and life-threatening disease, but the life of a patient can be saved if the stroke is detected during early stage. The literature reveals that the patients always experience ministrokes which are also known as transient ischemic attacks (TIA) before experiencing the actual attack of the stroke. Most of the literature work is based on the MRI and CT scan images for classifying the cardiovascular diseases including a stroke which is an expensive approach for diagnosis of early strokes. In India where cases of strokes are rising, there is a need to explore noninvasive cheap methods for the diagnosis of early strokes. [Sroke prediction](https://www.hindawi.com/journals/bn/2022/7725597/)

Hence a simple service that can use tabular form of patient data about their health and lifestyle factors, would be cheaper and can scale.

It is believed there are early factors that could help predict stroke. Imagine clinics and GPs have your medical data, and everytime you
go to the GP, with the latest data they collect on you, as soon data is enterted in the system, they get notified on risk of suffering a stroke based on current health and lifestyle factors. By equipping clinics to utilise a ML model, can help them to prioritise health of patients that have high risk of suffering stroke, and provide an intervention.


# Data Source 
The dataset is available on kaggle at [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
For the ease of discovery, dataset has been downloaded and made available in the repo [here](https://github.com/Nakulbajaj101/mlbookzoomcamp-midterm-stroke-prediction/blob/main/healthcare-dataset-stroke-data.csv)

# Build With
The section covers tools used to run the project
1. Python for data exploration with pandas, seaborn and matplotlib
2. Python for ML pipeline with sklearn, xgboost and bentoml
3. Bentoml framework in python to build the deployment service
4. Bash for orchatrating model training, building deployment and pushing to ECR on the cloud
5. AWS Fargate for deploying the model as a service on AWS
6. Locust for local load testing of bentoml api
7. Streamlit for prediction service app

# Project structure
![alt text](https://github.com/Nakulbajaj101/mlbookzoomcamp-midterm-stroke-prediction/blob/main/images/ProjectTree.png)

# How to run training

1. In the root directory run `pipenv shell`
2. Then run `python training.py`

# How to build the service and containerize it

1. Run the following commands in order in bash from the root directory
Note: Make sure `jq` is installed and `docker` is installed and running

Activate the pipenv virtual env shell
```bash
pipenv shell
```


```bash
echo "Building the bento"
bentoml build

# Containerise the application

echo "Containerise bento and building it"

export MODEL_TAG=$(bentoml get stroke_detection_classifier:latest -o json | jq -r .version)
cd ~/bentoml/bentos/$SERVICE_NAME/$MODEL_TAG && bentoml containerize $SERVICE_NAME:latest
```

# How to run the project end to end on AWS Fargate

Note: Make sure `jq` is installed and `docker` is installed and running, also make sure AWS profile is configured locally which has privelages to create ECR repo and create an image

Activate the pipenv virtual env shell
```bash
pipenv shell
```

Run the following bash script in the root directory
```bash
bash ./create_bento_artifacts.sh
```

This will train the model, build the service, containerize it and take it to ecr repo
The name of the repo will be `stroke_detection_classifier:latest`

Now we can follow the [video 7.6](https://www.youtube.com/watch?v=aF-TfJXQX-w&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=72) to deploy it behind fargate

We can then fetch the url provided by the service and update the streamlit app file
Navigate to app folder, and then inside the app.py file update the line 31 url

Then run the command in the app directory
```bash
pipenv run streamlit run app.py
```

Once the service is up navigate to `localhost:8501` in the browser, change the member id to 1 and click on "Predict Stroke Risk". The app will send the request to AWS Fargate, and a prediction will be returned

![Fargate Task Creation](https://github.com/Nakulbajaj101/mlbookzoomcamp-midterm-stroke-prediction/blob/main/images/FargateTaskCreation.png)

![Bento Deployed Swagger Api](https://github.com/Nakulbajaj101/mlbookzoomcamp-midterm-stroke-prediction/blob/main/images/BentoDeployedFargateSwaggerApi.png)

![Api Test](https://github.com/Nakulbajaj101/mlbookzoomcamp-midterm-stroke-prediction/blob/main/images/SwaggerApiResponse.png)

![Streamlit test with deployed](https://github.com/Nakulbajaj101/mlbookzoomcamp-midterm-stroke-prediction/blob/main/images/StreamlitStrokePrediction.png)

# How to do load testing

1. Run the bento prediction service locally from the root directory
```bash
pipenv run bentoml serve --production --reload -p 3000
```

2. Then in the other terminal from same root directory run Locust
```bash 
pipenv run locust -H http://localhost:3000
```

Navigate to `http://0.0.0.0:8089` in the browser and start the load testing

![Load Testing Result](https://github.com/Nakulbajaj101/mlbookzoomcamp-midterm-stroke-prediction/blob/main/images/LocalLoadTesting.png)

![Load Testing Terminal](https://github.com/Nakulbajaj101/mlbookzoomcamp-midterm-stroke-prediction/blob/main/images/LocustTerminalResults.png)



# Data exploration, model selection and EDA

1. Open the `exploratory_analysis.ipynb` file and `training.ipynb` file to see data exploration and model selection strategy
2. Open the `training.ipynb` and `training.py` script to see model selection and model building as ML pipelines

Note: Three models were trained, Decision tree, logistic regression and XGBoost, and best one was chosen with highest ROC.
