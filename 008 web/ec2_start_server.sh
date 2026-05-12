sudo yum update -y
sudo yum install python3-pip -y
python3 -m venv mlflow-env
source mlflow-env/bin/activate
pip install mlflow boto3 psycopg2-binary

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mlflow server \
    --backend-store-uri "sqlite:///$SCRIPT_DIR/mlflow.db" \
    --default-artifact-root "$SCRIPT_DIR/artifacts" \
    --host 0.0.0.0 \
    --port 5000

sudo yum install git docker -y
git clone https://github.com/maesedev/nlp-exposicion-servicios-rnn.git /home/ec2-user/inference-endpoint
cd /home/ec2-user/inference-endpoint/008 web/app
docker build -t inference-endpoint:latest .
docker run -d -p 8000:8000 \
    -v /home/ec2-user/ngrok_link.config:/home/ec2-user/ngrok_link.config:ro \
    inference-endpoint:latest