#!/bin/bash
# Expone el endpoint de inferencia FastAPI a través de un túnel ngrok.
#
# Requisitos:
#   1. ngrok instalado:        https://ngrok.com/download
#   2. Authtoken configurado:  ngrok config add-authtoken <YOUR_TOKEN>
#
# IMPORTANTE: el plan gratuito de ngrok permite un único agente online por cuenta.
# Si el túnel de Ollama está corriendo con la misma cuenta, hay que:
#   - correr este script en una máquina distinta a la de Ollama, o
#   - usar un authtoken de otra cuenta, o
#   - upgradear a un plan pago.
#
# Uso:
#   ./start_ngrok.sh                       # puerto 8000, URL → ~/ngrok_api.config
#   ./start_ngrok.sh 8000 /tmp/api.url     # puerto y archivo custom

set -e

API_PORT="${1:-8000}"
CONFIG_FILE="${2:-$HOME/ngrok_api.config}"

# Mata cualquier agente previo en este host
pkill -f "ngrok http $API_PORT" 2>/dev/null || true
sleep 1

echo "Iniciando ngrok sobre el puerto $API_PORT ..."
ngrok http "$API_PORT" --log=stdout > /tmp/ngrok_api.log 2>&1 &
NGROK_PID=$!

# Espera a que el túnel esté listo (ngrok local API en :4040)
URL=""
for i in {1..20}; do
  sleep 1
  URL=$(curl -s http://127.0.0.1:4040/api/tunnels 2>/dev/null \
    | python3 -c "import sys,json; t=json.load(sys.stdin).get('tunnels',[]); print(next((x['public_url'] for x in t if x['proto']=='https'), ''))" 2>/dev/null \
    || true)
  [ -n "$URL" ] && break
done

if [ -z "$URL" ]; then
  echo "Error: ngrok no expuso un túnel HTTPS tras 20 s. Revisa /tmp/ngrok_api.log"
  kill "$NGROK_PID" 2>/dev/null || true
  exit 1
fi

echo "$URL" > "$CONFIG_FILE"

cat <<EOF

ngrok API tunnel listo
----------------------
URL pública  : $URL
Local        : http://127.0.0.1:$API_PORT
Guardado en  : $CONFIG_FILE
Inspector    : http://127.0.0.1:4040
PID          : $NGROK_PID

Para usar desde el frontend, abrí la galería con:
  dinosaurios.html?api=$URL

(La URL queda guardada en localStorage; refrescar sin el query param mantiene la config.)
EOF
