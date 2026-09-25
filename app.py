import os
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, validator
import uvicorn

from NexText.components.model_prediction import ModelPrediction
from NexText.logging import logger

# Global predictor instance
predictor: ModelPrediction = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to safely load model once on startup
    and clean up resources on shutdown.
    """
    global predictor
    try:
        logger.info("Initializing ModelPrediction component on application startup...")
        predictor = ModelPrediction()
        logger.info(f"ModelPrediction ready. Using device: {predictor.device}")
    except Exception as e:
        logger.exception(f"Failed to load model on startup: {e}")
        raise e
    yield
    logger.info("Shutting down NexText Application...")


# Initialize FastAPI app
app = FastAPI(
    title="NexText - AI Text Summarization API",
    description="Abstractive text summarization using fine-tuned Pegasus transformer model on SamSum dataset.",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# Request and Response schemas
class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        description="The source text, dialogue, or conversation to summarize.",
        example="Amanda: I baked cookies for our team meeting today!\nJerry: Wow Amanda, you're the best!"
    )

    @validator("text")
    def validate_text_not_empty(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Input text cannot be empty or contain only whitespace.")
        if len(v) > 20000:
            raise ValueError("Input text exceeds maximum allowed length of 20,000 characters.")
        return v.strip()


class PredictionResponse(BaseModel):
    summary: str = Field(..., description="The generated abstractive summary.")


# Exception Handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    error_messages = []
    for err in exc.errors():
        msg = err.get("msg", "Invalid input")
        # Clean up pydantic default prefix if present
        if msg.startswith("Value error, "):
            msg = msg.replace("Value error, ", "")
        error_messages.append(msg)
    
    combined_message = "; ".join(error_messages)
    logger.warning(f"Validation error on {request.url.path}: {combined_message}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": combined_message}
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTP {exc.status_code} error on {request.url.path}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled server error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred while processing the request."}
    )


# Routes
@app.get("/", response_class=HTMLResponse, summary="Serve Frontend UI")
async def index(request: Request):
    """Serve the modern interactive summarizer web interface."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/health", summary="Service Health Check")
async def health_check() -> Dict[str, Any]:
    """Check API and model availability status."""
    return {
        "status": "healthy",
        "service": "NexText Summarization API",
        "device": predictor.device if predictor else "uninitialized",
        "model": "Pegasus-SamSum"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Generate Abstractive Summary",
    status_code=status.HTTP_200_OK
)
async def predict_summary(payload: PredictionRequest) -> Dict[str, str]:
    """
    Generate an abstractive summary for the provided input text or dialogue.
    
    - **text**: Source dialogue, article, or conversation string.
    - **Returns**: `{"summary": "..."}`
    """
    global predictor
    if predictor is None:
        logger.error("Predictor component is not initialized.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is still initializing or unavailable."
        )

    try:
        text_input = payload.text
        logger.info(f"Received summarization request with {len(text_input)} characters.")
        
        summary = predictor.predict(text_input)
        
        logger.info(f"Summary generated successfully ({len(summary)} characters).")
        return {"summary": summary}
        
    except Exception as e:
        logger.exception(f"Prediction failed during model inference: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )


if __name__ == "__main__":
    logger.info("Starting NexText FastAPI application on http://127.0.0.1:8080")
    uvicorn.run("app:app", host="127.0.0.1", port=8080, reload=False)
