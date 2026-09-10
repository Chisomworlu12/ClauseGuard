from fastapi import FastAPI

app = FastAPI(title="ClauseGuard")


@app.get("/health")
def health():
    return {"status": "ok"}


# TODO(phase 6): add a DB-connectivity probe via fastapi-health once the engine exists
