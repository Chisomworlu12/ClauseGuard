from fastapi import FastAPI

from api.routes import router as api_router

app = FastAPI(title="ClauseGuard")
app.include_router(api_router)

@app.get("/health")
def health():
    return {"status": "ok"}



