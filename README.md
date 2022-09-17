#  Finetuning for Turkish-English translation and online-batch deployment in Vertex AI

This code shows how to **finetune a BERT model** (stored in Hugging Face) for **Turkish to English translation**.
This code uses **Vertex AI Training with 1xV100 GPU** for finetuning, and **Vertex AI prediction** for online and batch predictions.


## Finetuning

Using libraries from Hugging Face, the code sample **fine tunes a Bert model** on the **WMT dataset (wmt16) for Turkish to English** [stored in Hugging Face](https://huggingface.co/datasets/wmt16/viewer/tr-en). The model will be then deployed in Vertex AI Model registry. The Bert model is the **Helsinki-NLP/opus-mt-tr-en** [stored in Huggig face](https://huggingface.co/Helsinki-NLP/opus-mt-tr-en).

The model is fine tuned on **Vertex AI with 1xV100 NVIDIA GPU**, using Vertex AI Training with autopackaging. Similar code for a different use case can be seen in [this codelab](https://codelabs.developers.google.com/vertex-training-autopkg#1). Command for training with autopackaging:
```sh
gcloud ai custom-jobs create \
--region=us-central1 \
--display-name=fine_tune_bert_tr_en \
--args=--job_dir=argolis-vertex-uscentral1 \
--worker-pool-spec=machine-type=n1-standard-4,replica-count=1,accelerator-type=NVIDIA_TESLA_V100,executor-image-uri="us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-7:latest",local-package-path='autopackage',python-module=trainer.task
```


## Custom Container image

A Custom Container image for  predictions is required. Custom Container image [requires that the container must run an HTTP server](https://cloud.google.com/ai-platform-unified/docs/predictions/custom-container-requirements#image). 
Specifically, the container must listen and respond to liveness checks, health checks, and prediction requests.

This repo uses **FastAPI and Uvicorn** to implement the HTTP server. 
The HTTP server must listen for requests on `0.0.0.0`. [Uvicorn](https://www.uvicorn.org) is an ASGI web server implementation for Python. 
Uvicorn currently supports HTTP/1.1 and WebSockets. 
[Here](https://github.com/tiangolo/uvicorn-gunicorn-fastapi-docker) is a docker image with Uvicorn managed by Gunicorn for high-performance FastAPI web applications in Python 3.6+ with performance auto-tuning. 
An uvicorn server is launched with:
```bash
uvicorn main:app --host 0.0.0.0 --port 8080
```


## Set up 

1. After finetuning, move the resulting model to the `predict/model-output-tr-en` directory (for example, with `gsutil`). The `predict` directory contains the `Dockerfile` to generate the Custom Container image. 
2. Follow the instructions below under section **Online predictions in Vertex AI**. It will upload the model to Vertex AI and will create an online endpoint.
3. Make a Batch prediction following instructions under section below **Batch predictions on Vertex AI**.

The model must be stored in the `predict/model-output-tr-en` directory, with a similar content like this:
```sh
-rw-r--r--   1 rafaelsanchez  primarygroup       1399 14 Sep 11:19 config.json
-rw-r--r--   1 rafaelsanchez  primarygroup  304670597 14 Sep 11:19 pytorch_model.bin
-rw-r--r--   1 rafaelsanchez  primarygroup     839750 14 Sep 11:19 source.spm
-rw-r--r--   1 rafaelsanchez  primarygroup         65 14 Sep 11:19 special_tokens_map.json
-rw-r--r--   1 rafaelsanchez  primarygroup     796647 14 Sep 11:19 target.spm
-rw-r--r--   1 rafaelsanchez  primarygroup        296 14 Sep 11:19 tokenizer_config.json
-rw-r--r--   1 rafaelsanchez  primarygroup       3183 14 Sep 11:19 training_args.bin
-rw-r--r--   1 rafaelsanchez  primarygroup    1688744 14 Sep 11:19 vocab.json
```


## Online predictions in Vertex AI

Push docker image to **Artifact Registry**:
```bash
gcloud auth configure-docker europe-west4-docker.pkg.dev
gcloud builds submit --tag europe-west4-docker.pkg.dev/argolis-rafaelsanchez-ml-dev/ml-pipelines-repo/finetuning_bert_model_tr_en
```

Upload model to Vertex AI prediction:
```python
from google.cloud import aiplatform

STAGING_BUCKET = 'gs://argolis-vertex-europewest4'
PROJECT_ID = 'argolis-rafaelsanchez-ml-dev'
LOCATION = 'europe-west4'

aiplatform.init(project=PROJECT_ID, staging_bucket=STAGING_BUCKET, location=LOCATION)

DEPLOY_IMAGE = 'europe-west4-docker.pkg.dev/argolis-rafaelsanchez-ml-dev/ml-pipelines-repo/finetuning_bert_model_tr_en' 
HEALTH_ROUTE = "/health"
PREDICT_ROUTE = "/predict"
SERVING_CONTAINER_PORTS = [8080]

model = aiplatform.Model.upload(
    display_name=f'custom-finetuning_bert_model_tr_en',    
    description=f'Finetuned BERT model with Uviron and FastAPI',
    serving_container_image_uri=DEPLOY_IMAGE,
    serving_container_predict_route=PREDICT_ROUTE,
    serving_container_health_route=HEALTH_ROUTE,
    serving_container_ports=SERVING_CONTAINER_PORTS,
)

print(model.resource_name)

# Retrieve a Model on Vertex
model = aiplatform.Model(model.resource_name)
print(model.resource_name)

# Deploy model
endpoint = model.deploy(
      machine_type='n1-standard-2', 
      sync=False
)
endpoint.wait()

# Retrieve an Endpoint on Vertex
print(endpoint.predict([["Bu odun yanmaz."]]))
# Output: Prediction(predictions=[[["This wood won't burn."]]], deployed_model_id='5089885615579725824', model_version_id='1', model_resource_name='projects/989788194604/locations/europe-west4/models/1092976130660499456', explanations=None)
```

Predict using REST API online endpoint:
```bash
curl -X POST -H "Authorization: Bearer $(gcloud auth print-access-token)" -H "Content-Type: application/json" \
https://europe-west4-aiplatform.googleapis.com/v1alpha1/projects/989788194604/locations/europe-west4/endpoints/5366610702058389504:predict \
-d "{\"instances\": [\"Bu odun yanmaz.\"]}"
{
  "predictions": [
    [
      [
        "This wood won't burn."
      ]
    ]
  ],
  "deployedModelId": "5089885615579725824",
  "model": "projects/989788194604/locations/europe-west4/models/1092976130660499456",
  "modelDisplayName": "custom-finetuning_bert_model_tr_en"
}
```

## Running docker locally

Build and run locally (for info see [here](https://cloud.google.com/ai-platform/prediction/docs/getting-started-pytorch-container#run_the_container_locally_optional))
```bash
docker build -t finetuning_bert_model_tr_en .
docker run -p 7080:7080 -e AIP_HTTP_PORT=7080 \
    -e AIP_HEALTH_ROUTE=/health \
    -e AIP_PREDICT_ROUTE=/predict \
    finetuning_bert_model_tr_en:latest
[...]
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

To access container shell:
```bash
docker run -it --rm -p 8080:8080 \
    --name=finetuning_bert_model_tr_en \
    -e AIP_HTTP_PORT=7080 \
    -e AIP_HEALTH_ROUTE=/health \
    -e AIP_PREDICT_ROUTE=/predict \
    -e AIP_STORAGE_URI='gs://argolis-vertex-europewest4' \
    0395efe5870d \
    /bin/bash
```

Prediction example:
```bash
curl -i -X POST http://localhost:7080/predict  -d "{\"instances\": [\"Bu odun yanmaz.\"]}"
HTTP/1.1 200 OK
date: Sun, 03 Jul 2022 14:54:39 GMT
server: uvicorn
content-length: 21
content-type: application/json

{"predictions":....}
```

Prediction example with local file `instance_test.json`:
```bash
curl -i -X POST -d @instances_test.json   -H "Content-Type: application/json; charset=uost:7080/predict 
HTTP/1.1 200 OK
date: Sun, 03 Jul 2022 14:56:39 GMT
server: uvicorn
content-length: 21
content-type: application/json

{"predictions":....}
```


## Batch predictions on Vertex AI

Upload `batch_input_data.jsonl` to GCS using `gsutil`:
```sh
gsutil cp batch_input_data.jsonl <YOUR_GCS_BUCKET>
```

Launch a Batch prediction on Vertex AI with `predict_batch.py`:

Results are stored in GCS in the folder `prediction-<model-display-name>-<job-create-time>`. Inside of it, multiple files of type `prediction.results-00000-of-000XX`:
```json
{"instance": "Bu odun yanmaz.", "prediction": [["This wood won't burn."], ["At least on paper, it seems like a great idea."]]}
```


## References

[1] Medium article: [How to fine-tune pre-trained translation model](https://medium.com/@tskumar1320/how-to-fine-tune-pre-trained-language-translation-model-3e8a6aace9f)      
[2] Codelab: [Vertex AI: Use autopackaging to fine tune Bert with Hugging Face on Vertex AI Training](https://codelabs.developers.google.com/vertex-training-autopkg#1)    
