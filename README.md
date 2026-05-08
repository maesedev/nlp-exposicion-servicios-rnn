# 001 Preprocesamiento — Tokenización BPE de nombres de dinosaurios

Este módulo entrena un vocabulario **BPE (Byte Pair Encoding)** sobre 1 535 nombres de dinosaurios y expone utilidades para codificar texto nuevo con ese vocabulario.

## Archivos

| Archivo | Descripción |
|---|---|
| `Creacion de tokens.ipynb` | Notebook exploratorio que origina el pipeline |
| `vocabulary_creator.py` | Script principal: normaliza, entrena BPE y guarda los artefactos |
| `token_encoder.py` | Codificador: carga vocabulario + reglas y tokeniza texto libre |
| `tokens/vocabulary.txt` | Vocabulario final (un token por línea; índice = número de línea) |
| `tokens/merges.csv` | Reglas de fusión BPE ordenadas por paso |

## Pipeline paso a paso

### 1. Carga y normalización

```
"Tyrannosaurus Rex" → "tyrannosaurus"
```

Cada nombre se normaliza con la función `normalize`:

1. Minúsculas.
2. Descomposición NFD y eliminación de marcas diacríticas (acentos, tildes).
3. Retención solo de caracteres `[a-z0-9]`.

El resultado es una lista de ~1 535 palabras limpias derivadas de `000 data/Dinosours.csv`.

### 2. Entrenamiento BPE

BPE parte de cada palabra dividida en caracteres individuales y fusiona iterativamente el par de símbolos más frecuente.

```
velociraptor
→  v e l o c i r a p t o r          (inicio: caracteres)
→  v e l o c i r a p t or           (paso 1: o+r → or)
→  v e l o c i r a p tor            (paso 2: t+or → tor)
→  v e l o c i ra p tor             (paso 3: r+a → ra)
→  v e l o c i ra p tor             ...
→  velociraptor                      (tras N fusiones)
```

Parámetros:
- `--merges` (default 300): número de operaciones de fusión.
- El vocabulario resultante tiene **4 tokens especiales + tokens BPE** ordenados alfabéticamente.

### 3. Construcción de secuencias

Cada palabra se encapsula con marcadores de inicio y fin:

```
velociraptor → [<SOW>, ve, l, oci, raptor, <EOW>]
```

Si la secuencia resultante es más corta que `--min-len` (default 4), se rellena con `<PAD>` por la derecha.

### 4. Tokens especiales

| Token | Índice | Rol |
|---|---|---|
| `<PAD>` | 0 | Relleno para alcanzar longitud mínima |
| `<SOW>` | 1 | Inicio de palabra (*Start Of Word*) |
| `<EOW>` | 2 | Fin de palabra (*End Of Word*) |
| `<UNK>` | 3 | Token desconocido (no visto en vocabulario) |

### 5. Artefactos generados

**`tokens/vocabulary.txt`** — El vocabulario final (277 tokens en la ejecución por defecto):

```
<PAD>       ← índice 0
<SOW>       ← índice 1
<EOW>       ← índice 2
<UNK>       ← índice 3
a           ← índice 4
ab          ← índice 5
...
saurus      ← índice alto
raptor      ← ...
```

**`tokens/merges.csv`** — Cada fila documenta una fusión:

```
paso, merge,       token_resultante, frecuencia
1,    s + </w>,    s</w>,            1066
6,    saur + us,   saurus,           706
97,   ra + ptor,   raptor,           18
```

## Uso de los scripts

### Crear vocabulario desde cero

```bash
python vocabulary_creator.py "../000 data/Dinosours.csv"

# Opciones
python vocabulary_creator.py "../000 data/Dinosours.csv" \
    --merges 400 \   # más fusiones → tokens más largos
    --min-len 6  \   # longitud mínima de secuencia
    --out-dir tokens
```

Salida esperada:

```
Words loaded    : 1535
BPE merges done : 300
Vocabulary size : 277  (4 special + 273 BPE tokens)
Sequences saved : 1535  (X padded to T=4)
```

### Codificar texto nuevo

```bash
# Codificar una palabra
python token_encoder.py tokens/vocabulary.txt tokens/merges.csv "Velociraptor"
# → <SOW> ve l oci raptor <EOW>

# Leer desde stdin
echo "Tyrannosaurus" | python token_encoder.py tokens/vocabulary.txt tokens/merges.csv

# Sin normalizar (modo raw)
python token_encoder.py tokens/vocabulary.txt tokens/merges.csv "velociraptor" --raw

# Sin vocabulario (sin SOW/EOW ni UNK)
python token_encoder.py tokens/vocabulary.txt tokens/merges.csv "velociraptor" --no-vocab
```

## Parámetros clave y su efecto

| Parámetro | Valor bajo | Valor alto |
|---|---|---|
| `--merges` | Vocabulario pequeño, tokens cortos (casi caracteres) | Vocabulario grande, tokens largos (palabras enteras) |
| `--min-len` | Secuencias más cortas, sin padding | Secuencias uniformes, más padding |

## Dependencias

```
pandas
```

Solo la biblioteca estándar de Python más `pandas` (usado en el notebook). El script `vocabulary_creator.py` solo usa la biblioteca estándar.
