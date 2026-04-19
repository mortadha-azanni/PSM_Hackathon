from fastapi import FastAPI

app = FastAPI(title="ML Service")

@app.get("/")
async def root():
    return {"message": "Hello from ML"}
