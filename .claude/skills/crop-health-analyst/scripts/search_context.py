"""Search agricultural reference documents.

Usage: python search_context.py "<query>"
"""
import os, sys
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
from agent.tools import search_agricultural_context

query = sys.argv[1] if len(sys.argv) > 1 else "crop stress interpretation"
result = search_agricultural_context(query)
print(result.summary)
