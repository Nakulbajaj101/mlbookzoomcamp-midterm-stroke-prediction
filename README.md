# Stroke Prediction Service
The prupose of this project is to provide a stroke prediction service, which could be used by GPs, doctors and health providers to understand the risk of stroke. By picking early indication of risk, health service providers can provide intervention and help patients
understand the risk. They can further work with the patients to design a plan or provide a prescription to reduce the risk of stroke

The service will be provided as an API, and the IT teams at Health providers can use it to support web portals and forms.

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

# Project structure
![alt text](https://github.com/Nakulbajaj101/mlbookzoomcamp-midterm-stroke-prediction/blob/main/images/ProjectTree.png)
