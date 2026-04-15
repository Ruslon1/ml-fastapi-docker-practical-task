# ML FastAPI Docker

Iris classification model served with FastAPI.

## Local

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 train.py
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs`.

## Docker

```bash
docker build -t ml-fastapi-docker .
docker run --rm -p 8000:8000 ml-fastapi-docker
```
