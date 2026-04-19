from fastapi import FastAPI

app = FastAPI(title="Search Service")

@app.get("/")
async def root():
    return {"message": "Hello from Search"}
