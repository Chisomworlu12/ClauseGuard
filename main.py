from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router as api_router
from config import settings

app = FastAPI(title="ClauseGuard")

allowed_origins = [
    origin.strip()
    for origin in settings.frontend_origins.split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

app.include_router(api_router)


@app.get("/health")
def health():
    return {"status": "ok"}
