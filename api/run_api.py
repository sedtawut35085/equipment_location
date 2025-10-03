#!/usr/bin/env python3
"""
Script to run the FastAPI application with uvicorn
"""
import uvicorn

if __name__ == "__main__":
    print("Starting Meat Classifier API...")
    print("API will be available at: http://localhost:8000")
    print("Interactive API docs at: http://localhost:8000/docs")
    
    uvicorn.run(
        "main_api:app",  # Use import string instead of importing app object
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )
