from google.cloud import aiplatform
import logging

STAGING_BUCKET = 'gs://argolis-vertex-europewest4'
PROJECT_ID = 'argolis-rafaelsanchez-ml-dev'
LOCATION = 'europe-west4'

SOURCE_FILE_URI= 'gs://argolis-vertex-europewest4/batch_input_data.jsonl'

aiplatform.init(project=PROJECT_ID, staging_bucket=STAGING_BUCKET, location=LOCATION)

model_custom = aiplatform.Model('projects/989788194604/locations/europe-west4/models/1092976130660499456')

batch_prediction_job = model_custom.batch_predict(
    job_display_name='custom-bert-tr-en-batch',
    gcs_source=SOURCE_FILE_URI, 
    instances_format='jsonl',
    gcs_destination_prefix=STAGING_BUCKET,
    machine_type="n1-standard-4",
    #accelerator_type= "NVIDIA_TESLA_T4",
    #accelerator_count = 1
)

batch_prediction_job.wait()

logging.info('destination: %s',  STAGING_BUCKET)

logging.info('batch_prediction_job.display_name: %s', batch_prediction_job.display_name)
logging.info('batch_prediction_job.resource_name: %s', batch_prediction_job.resource_name)
logging.info('batch_prediction_job.state: %s', batch_prediction_job.state)