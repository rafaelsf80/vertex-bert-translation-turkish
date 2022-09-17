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
#endpoint = aiplatform.Endpoint('projects/989788194604/locations/europe-west4/endpoints/5366610702058389504')
print(endpoint.predict([["Bu odun yanmaz."]]))
# Output: 
# Prediction(predictions=[[["This wood won't burn."]]], deployed_model_id='5089885615579725824', model_version_id='1', 
#     model_resource_name='projects/989788194604/locations/europe-west4/models/1092976130660499456', explanations=None)