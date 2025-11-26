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