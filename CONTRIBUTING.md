### Contributing

Thanks for your interest in contributing!

1. Fork the repo and create a feature branch.
2. Set up a virtual environment and install dependencies:
   - Full: `pip install -r requirements.txt`
   - CI/minimal: `pip install -r requirements-ci.txt`
3. Run the pipeline locally (optional): `python run_pipeline.py --step all`
4. Run the app: `streamlit run webapp/streamlit_app.py`
5. Run lint and tests:
   - `flake8 --max-line-length=120 --extend-ignore=E203,W503 .`
   - `pytest -q`
6. Open a pull request with a clear description.

Please avoid committing large data or model artifacts. See `.gitignore`.
