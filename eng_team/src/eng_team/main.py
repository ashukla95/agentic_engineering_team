#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from eng_team.crew import EngTeam

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    inputs = {
        "product_name": "trading platform",
        "competitor_product_name": "robinhood"
    }
    
    try:
        EngTeam().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")
