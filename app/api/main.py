from fastapi import FastAPI
from app.api.routes import health, predict

app = FastAPI(title="Dogs vs Cats API")

app.include_router(health.router)
app.include_router(predict.router)