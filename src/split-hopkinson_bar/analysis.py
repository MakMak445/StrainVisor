import argparse
import asyncio
import base64
import csv as _csv
import json
import math
import os
import re
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Tuple
#makes assumption that 
def parse_args():
    p = argparse.ArgumentParser(description="Analyse a folder containing the frames of a SHB experiment")
    p.add_argument("root", help="root folder for camera (njord or slave)")
def main():
    try:
        asyncio.run(main_async(parse_args()))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)