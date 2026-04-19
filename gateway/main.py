from fastapi import FastAPI

app = FastAPI(title="Gateway Service")

@app.get("/")
async def root():
    return {"message": "Hello from Gateway"}
