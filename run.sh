pip install uv 
uv venv .venv 
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e . --no-build-isolation
