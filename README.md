# Paper to audio

Simple conversion of pdf to audio using LLM calls. 
`paper2audio_gemini.ipynb` contains all necessary code to load a pdf and save it to a .wav audio file. Requires gemini API key. 

### Installation

1. **Prerequisites:**
   - Python 3.13+
   - [`uv`](https://github.com/astral-sh/uv) (install with `pip install uv` or follow uv's installation guide)

2. **Clone the repository:**
   ```bash
   git clone <URL_OF_THE_REPO>
   cd paper2audio
   ```

3. **Create and activate a virtual environment (with uv):**
   ```bash
   uv venv .venv
   source .venv/bin/activate
   ```
   For Windows, use: `\.venv\Scripts\activate`

4. **Install dependencies:**
   The project dependencies are listed in `pyproject.toml`. Install them using **uv**:
   ```bash
   uv pip install .[dev]
   ```
   This will install both the main dependencies and the development dependencies needed to run the notebook.

5. **Set your Gemini API Key:**
   This project uses the Google Gemini API. You will need to obtain an API key and set it as an environment variable.
   ```bash
   export GEMINI_API_KEY="YOUR_API_KEY_HERE"
   ```
   Replace `"YOUR_API_KEY_HERE"` with your actual key.

6. **Run the Jupyter Notebook:**
   Launch Jupyter Lab or Jupyter Notebook and open `paper2audio_gemini.ipynb`.


