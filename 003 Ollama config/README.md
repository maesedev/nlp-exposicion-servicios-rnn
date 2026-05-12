# Install Ollama

wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz

sudo tar xvzf ./ngrok-v3-stable-linux-amd64.tgz -C /usr/local/bin

ngrok config add-authtoken 3D09buBp6T4rNGFvvXMKSFVgwER_5ZEc2YVLR6pcELaZ7Mz6Y

docker pull ollama/ollama

docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

docker exec -it ollama bash

ollama run phi4

ngrok http 11434
