from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles as S
from api.routes import router
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Pokemon Classifier API")

# Serve static files
app.mount("/static", S(directory=os.path.join(BASE_DIR, "frontend")), name="static")

# Register API routes
app.include_router(router, prefix="/api")


# Serve frontend
@app.get("/", response_class=HTMLResponse)
def home():
    with open(os.path.join(BASE_DIR, "frontend", "index.html")) as f:
        return f.read()
