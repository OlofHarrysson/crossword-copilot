from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .models import MockResponse


app = FastAPI(title="Template API", version="1.0.0")

# Add CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Template API is running"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/items", response_model=MockResponse)
async def get_items():
    """Get mock items data"""
    return MockResponse(count=3)
