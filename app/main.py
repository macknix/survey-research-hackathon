from fastapi import FastAPI, File, UploadFile, Request, Body
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import io
import pandas as pd
import requests
import json
from typing import List, Optional
from pydantic import BaseModel

app = FastAPI()

# Templates and static files (for our single page)
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# --- Pydantic models (must be defined before use) ---
class Topic(BaseModel):
    id: int
    label: str
    description: str

class GenerateTopicsRequest(BaseModel):
    column: str
    sample_rows: Optional[List[dict]] = None

class GenerateTopicsResponse(BaseModel):
    success: bool
    topics: List[Topic]

class BuildPromptRequest(BaseModel):
    topics: List[Topic]

class BuildPromptResponse(BaseModel):
    success: bool
    prompt: str

class ClassifyRowsRequest(BaseModel):
    column: str
    rows: List[dict]
    topics: List[dict]  # Accept as dicts from frontend
    model: str
    batch_size: int = 10

class ClassifyRowsResponse(BaseModel):
    success: bool
    columns: List[str]
    rows: List[dict]
    error: Optional[str] = None

def build_batch_prompt(topics: List[dict], texts: List[str]) -> str:
    topics_desc = "\n".join(
        f"- {t['id']}: {t['label']} — {t['description']}" for t in topics
    )
    prompt = f"""You are a text classification assistant.\n\nYou will be given a list of survey responses or free-text entries.\nFor each entry, assign exactly ONE topic ID from the list below.\n\nTopics:\n{topics_desc}\n\nReturn your answer as a JSON list, one object per input, with this schema:\n\n[\n  {{\n    \"topic_id\": <number>,  // one of the IDs above\n    \"topic_label\": <string>,\n    \"rationale\": <string>  // short explanation for why this topic was chosen\n  }}\n]\n\nMake sure the JSON is valid and matches the number/order of inputs.\n\nHere are the entries:\n"""
    for i, text in enumerate(texts):
        prompt += f"{i+1}. {text}\n"
    prompt += "\n---\nJSON:"
    return prompt

def call_ollama(prompt: str, model: str) -> str:
    # Assumes Ollama is running locally on port 11434
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    print(payload)
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    result = response.json()
    return result.get("response", "")


@app.post("/classify_rows", response_model=ClassifyRowsResponse)
async def classify_rows(payload: ClassifyRowsRequest):
    """
    Classify the selected column using Ollama LLM in batches. Append results to the dataframe.
    """
    try:
        texts = [str(row.get(payload.column, "")) for row in payload.rows]
        batch_size = payload.batch_size or 10
        all_results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            prompt = build_batch_prompt(payload.topics, batch_texts)
            try:
                llm_response = call_ollama(prompt, payload.model)
                parsed = json.loads(llm_response)
                if not isinstance(parsed, list):
                    return ClassifyRowsResponse(success=False, columns=[], rows=[], error="LLM did not return a list.")
                all_results.extend(parsed)
            except Exception as e:
                return ClassifyRowsResponse(success=False, columns=[], rows=[], error=f"LLM error: {e}")
        # Append results to original rows
        output_rows = []
        for row, result in zip(payload.rows, all_results):
            new_row = dict(row)
            for k, v in result.items():
                new_row[k] = v
            output_rows.append(new_row)
        # Columns: original + new fields
        extra_cols = set()
        for r in all_results:
            extra_cols.update(r.keys())
        columns = list(payload.rows[0].keys()) + [c for c in extra_cols if c not in payload.rows[0]]
        return ClassifyRowsResponse(success=True, columns=columns, rows=output_rows)
    except Exception as e:
        return ClassifyRowsResponse(success=False, columns=[], rows=[], error=str(e))




@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Serve the main page with the upload UI.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    """
    Receive a CSV file, parse it with pandas, and return first N rows as JSON.
    """
    # Read file contents into memory
    contents = await file.read()
    buffer = io.BytesIO(contents)

    # You can refine encoding/sep later as needed
    try:
        df = pd.read_csv(buffer)
    except Exception as e:
        return {"success": False, "error": f"Failed to parse CSV: {e}"}

    # Limit rows and columns sent to front-end
    preview_rows = 5
    df_preview = df.head(preview_rows)

    # Prepare data for JSON (list of dicts) and column names
    data = df_preview.to_dict(orient="records")
    columns = list(df_preview.columns)

    return {
        "success": True,
        "columns": columns,
        "rows": data,
        "row_count": len(df),
    }


# --- Topic modeling and prompt endpoints ---
from typing import List, Optional
from pydantic import BaseModel

class Topic(BaseModel):
    id: int
    label: str
    description: str

class GenerateTopicsRequest(BaseModel):
    column: str
    sample_rows: Optional[List[dict]] = None

class GenerateTopicsResponse(BaseModel):
    success: bool
    topics: List[Topic]

class BuildPromptRequest(BaseModel):
    topics: List[Topic]

class BuildPromptResponse(BaseModel):
    success: bool
    prompt: str

@app.post("/generate_topics", response_model=GenerateTopicsResponse)
async def generate_topics(payload: GenerateTopicsRequest):
    """
    Stub: return some example topics for the given column.
    Later, replace this with real topic modeling.
    """
    stub_topics = [
        Topic(id=1, label="Positive feedback", description="Comments that express satisfaction or positive sentiment."),
        Topic(id=2, label="Negative feedback", description="Comments that express dissatisfaction or complaints."),
        Topic(id=3, label="Feature requests", description="Suggestions or requests for new features or improvements."),
    ]
    return GenerateTopicsResponse(success=True, topics=stub_topics)

@app.post("/build_prompt", response_model=BuildPromptResponse)
async def build_prompt(payload: BuildPromptRequest):
    """
    Build a LangChain-style Pydantic parser prompt for classifying rows into topics.
    This is a stub that returns a structured prompt string.
    """
    topics = payload.topics
    topics_desc = "\n".join(
        f"- {t.id}: {t.label} — {t.description}" for t in topics
    )
    prompt = f"""You are a text classification assistant.\n\nYou will be given a piece of text from a survey response or similar free-text field.\nYour task is to assign exactly ONE topic ID from the list below to the text.\n\nTopics:\n{topics_desc}\n\nReturn your answer as JSON with this schema:\n\n{{\n  \"topic_id\": <number>,  // one of the IDs above\n  \"topic_label\": <string>,\n  \"rationale\": <string>  // short explanation for why this topic was chosen\n}}\n\nMake sure the JSON is valid and does not include any additional fields.\n"""
    return BuildPromptResponse(success=True, prompt=prompt)