from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import io
import pandas as pd

app = FastAPI()

# Templates and static files (for our single page)
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")


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