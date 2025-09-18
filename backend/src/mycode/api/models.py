from pydantic import BaseModel


class MockResponse(BaseModel):
    """Response model for the mock items endpoint"""

    count: int
