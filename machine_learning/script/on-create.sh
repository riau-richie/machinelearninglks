#!/bin/bash
sudo -u ec2-user -i <<'EOF'
# Install required packages
pip install scikit-learn numpy pandas boto3 pickle-mixin

# Create project structure
mkdir -p /home/ec2-user/SageMaker/techmart-ml-recommendation/{models,data,scripts,notebooks}

# Set environment variables
echo "export BUCKET_NAME=yourbucketname" >> ~/.bashrc
echo "export REGION=us-east-1" >> ~/.bashrc
EOF