import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import SERVER_HOST, SERVER_PORT
from .models.registry import ModelRegistry
from .routers import audio, models, realtime


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry = ModelRegistry()
    registry.scan()
    app.state.registry = registry
    yield


app = FastAPI(
    title="ExecuTorch Voice Studio",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(models.router, prefix="/v1")
app.include_router(audio.router, prefix="/v1")
app.include_router(realtime.router, prefix="/v1")


@app.get("/")
async def root():
    return {
        "name": "ExecuTorch Voice Studio",
        "version": "0.1.0",
        "docs": "/docs",
    }


def main():
    uvicorn.run(
        "backend.main:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=True,
    )


if __name__ == "__main__":
    main()
