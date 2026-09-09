"""FastAPI Application Entrypoint."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from cv_segment.api.routes import router

app = FastAPI(
    title="CV Stream Segmentation API",
    description="Automated boundary detection and lossless PDF splitting engine.",
    version="1.0.0",
)

# Enable CORS for frontend or web portals
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
