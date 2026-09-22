"""Main FastAPI application entrypoint."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.endpoints.optimizer import router as optimizer_router

app = FastAPI(
    title="Portfolio Optimizer API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(optimizer_router, prefix="/api/v1", tags=["Portfolio Optimizer"])


@app.get("/health", tags=["Health"])
def health_check():
    """Health check probe endpoint."""
    return {"status": "ok"}
