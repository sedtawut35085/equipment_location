#!/usr/bin/env python3
"""
Configuration Management Module
Loads and manages environment variables for the equipment classification project
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class to manage all environment variables"""
    # Class Names
    EQUIPMENT_CLASS_NAMES = os.getenv('EQUIPMENT_CLASS_NAMES', 'microphone,headphone').split(',')

# Create a global instance
config = Config()

CLASS_NAMES = config.EQUIPMENT_CLASS_NAMES