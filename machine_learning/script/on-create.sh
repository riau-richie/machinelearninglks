#!/bin/bash
set -e

# 1. System Updates and Dependencies
sudo yum update -y
sudo yum install -y python3 python3-devel git wget

# 2. Configure Python Environment
sudo pip3 install --upgrade pip
sudo pip3 install boto3 pandas numpy scikit-learn

# 3. Repository Management
REPO_URL="yourrepository"
TARGET_DIR="/home/ec2-user/SageMaker/machinelearninglks"

if [ ! -d "$TARGET_DIR" ]; then
    git clone "$REPO_URL" "$TARGET_DIR"
else
    cd "$TARGET_DIR"
    git fetch origin
    git reset --hard origin/main
fi

# 4. Jupyter Notebook Configuration
JUPYTER_CONFIG_DIR="/home/ec2-user/.jupyter"
JUPYTER_CONFIG_FILE="$JUPYTER_CONFIG_DIR/jupyter_notebook_config.py"

mkdir -p "$JUPYTER_CONFIG_DIR"

cat << EOF > "$JUPYTER_CONFIG_FILE"
# Disable browser auto-opening
c.ServerApp.open_browser = False

# Set default port
c.ServerApp.port = 8888

# Allow all IPs
c.ServerApp.ip = '0.0.0.0'

# Disable token authentication
c.ServerApp.token = ''

# Set notebook directory
c.ServerApp.notebook_dir = '/home/ec2-user/SageMaker'
EOF

# 5. Permissions and Environment Setup
sudo chown -R ec2-user:ec2-user "$TARGET_DIR"
sudo chown -R ec2-user:ec2-user "$JUPYTER_CONFIG_DIR"

# 6. Install Additional Requirements (if any)
if [ -f "$TARGET_DIR/requirements.txt" ]; then
    sudo pip3 install -r "$TARGET_DIR/requirements.txt"
fi

# 7. Copy notebook to main directory (optional)
NOTEBOOK_SOURCE="$TARGET_DIR/machine_learning/training.ipynb"
NOTEBOOK_DEST="/home/ec2-user/SageMaker/training.ipynb"

if [ -f "$NOTEBOOK_SOURCE" ]; then
    cp "$NOTEBOOK_SOURCE" "$NOTEBOOK_DEST"
    chown ec2-user:ec2-user "$NOTEBOOK_DEST"
fi

# 8. Restart Jupyter (if running)
if pgrep -f "jupyter-notebook"; then
    pkill -f "jupyter-notebook"
    sleep 2
    sudo -u ec2-user nohup jupyter notebook --allow-root &> /dev/null &
fi

echo "Lifecycle configuration completed successfully"
