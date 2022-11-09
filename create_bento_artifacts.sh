#!bin/bash

################SET AWS DEFAULT PROFILE#################
# export AWS_PROFILE=your_profile_with_ECS_PERMISSIONS #
# Watch the video https://www.youtube.com/watch?v=aF-TfJXQX-w&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=72 #

####Once the profile is ready you can take the image to AWS####

export MODEL_NAME="stroke_detection_model"
export SERVICE_NAME="stroke_detection_classifier"

# Running the training
echo "Running training"
python training.py

# Building the application

echo "Building the bento"
bentoml build

# Containerise the application

echo "Containerise bento and building it"

export MODEL_TAG=$(bentoml get stroke_detection_classifier:latest -o json | jq -r .version)
cd ~/bentoml/bentos/$SERVICE_NAME/$MODEL_TAG && bentoml containerize $SERVICE_NAME:latest

echo "Tagging the docker image"
# Defining docker image tag and account id
export DOCKER_IMAGE=$SERVICE_NAME:$MODEL_TAG
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity | jq -r .Account)
export AWS_REGION=$(aws configure get region)


echo "AWS docker login and pushing the latest image"
# Docker login for ecr
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com


echo "Creating the repo"
# Create a reporsitory
aws ecr create-repository \
    --repository-name $SERVICE_NAME \
    --image-scanning-configuration scanOnPush=true \
    --region $AWS_REGION

# Tag the image

docker tag $DOCKER_IMAGE $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$SERVICE_NAME

# Pushin the image


echo "Pushing the image"
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$SERVICE_NAME
