from fastapi import FastAPI
from etl.celery_app import celery_app

app = FastAPI(title="ETL Service")

@app.get("/")
async def root():
    return {"message": "Hello from ETL"}

@app.post("/task")
async def create_task():
    task = celery_app.send_task("tasks.process_data", args=[1, 2])
    return {"task_id": task.id}
