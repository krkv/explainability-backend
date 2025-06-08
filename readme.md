# XAI LLM chat backend

## About

The purpose of this Flask application is to provide an API for LLM assistant requests.

The API can be used to integrate with an instance of XAI LLM chat frontend, or another application.

## Development

### Set up a virtual environment

```
python3 -m venv venv
source venv/bin/activate
```

### Install app requirements

```
pip install -r requirements.txt
```

### Create .env file and set the HF token

You can use `.env.example` file as an example of a `.env` file.

```
HF_TOKEN="hf_***"
```

### Start dev server

```
FLASK_ENV="development" flask run --debug
```