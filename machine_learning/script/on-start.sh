#!/bin/bash
sudo -u ec2-user -i <<'EOF'
# Check data freshness and trigger ETL if needed
python3 -c "
import boto3
from datetime import datetime, timedelta

s3 = boto3.client('s3')
glue = boto3.client('glue')

# Check last modified time of processed data
response = s3.list_objects_v2(Bucket='techmart-ml-bucket', Prefix='processed-data/')
if response['Contents']:
    last_modified = response['Contents'][0]['LastModified']
    if datetime.now(last_modified.tzinfo) - last_modified > timedelta(days=1):
        print('Data is stale. Triggering ETL pipeline...')
        glue.start_job_run(JobName='techmart-etl-pipeline')
"

# Download latest model
aws s3 cp s3://yourbucketname/models/hybrid_model.pkl /home/ec2-user/SageMaker/techmart-ml-recommendation/models/ --quiet
EOF