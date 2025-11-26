# Survey Research Hackathon – CSV Classifier Web App

This project is a locally hosted web app for uploading a CSV, previewing its contents, defining topics for classification, and generating a LangChain-compatible Pydantic parser prompt. (You can later extend it to classify rows using an LLM via Ollama.)

## Features

- Upload a CSV and preview the first few rows
- Select a column to classify
- Add, edit, and remove topics and their descriptions in a table
- Generate a prompt for use with LangChain Pydantic parser

## Setup

1. **Clone the repository** (if you haven't already):

	```bash
	git clone <your-repo-url>
	cd survey-research-hackathon
	```

2. **Create and activate a Python virtual environment** (recommended):

	```bash
	python3 -m venv .venv
	source .venv/bin/activate
	```

3. **Install dependencies:**

	```bash
	pip install -r requirements.txt
	```

## Running the App

Start the FastAPI server with Uvicorn:

```bash
uvicorn app.main:app --reload
```

Then open your browser and go to:

- [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

## Usage

1. **Upload a CSV file**: Click "Upload & Preview" and select your CSV. The first few rows will be displayed.
2. **Select the column to classify**: Use the dropdown to pick the target column.
3. **Define topics**: Add, edit, or remove topics and their descriptions in the table.
4. **Build prompt**: Click "Build LangChain Pydantic prompt" to generate a prompt for use with an LLM.

## Next Steps

- Integrate with Ollama and LangChain for row classification
- Add support for saving/loading topic sets
- Add tests and more advanced error handling

---

**Developed for the Survey Research Hackathon**
