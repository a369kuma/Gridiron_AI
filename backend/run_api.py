#!/usr/bin/env python3
"""
API server startup script for Gridiron AI.
This script initializes the database and starts the Flask API server.
"""

import os
import sys
import logging
from api.app import app, init_db, load_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Start the API server."""
    logger.info("Starting Gridiron AI API server...")
    
    try:
        # Initialize database
        logger.info("Initializing database...")
        init_db()
        
        # Load model
        logger.info("Loading trained model...")
        load_model()
        
        # Start server
        logger.info("Starting Flask server on http://localhost:5003")
        app.run(
            debug=True,
            host='0.0.0.0',
            port=5003,
            threaded=True
        )
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise

if __name__ == "__main__":
    main()
