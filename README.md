# ScholarSwarm — AI Scholarship Finder

A Streamlit web application with four cooperating LangGraph agents:

1. **Profile Agent** — reads the profile form, resume/transcript PDF/TXT/photo, GitHub public profile, and public links.
2. **Research Agent** — plans targeted searches, searches the web, opens relevant pages, and asks for broader searches when results are sparse.
3. **Eligibility Agent** — extracts criteria and requires quotes that can be found in retrieved source text before accepting a verdict, funding claim, or deadline.
4. **Ranking & Application Agent** — ranks matches with a visible score breakdown, suggests next actions, and drafts an editable SOP.

The app only prepares drafts. It never submits university applications.

## Fastest Windows launch

1. Install Python 3.11 or newer from https://www.python.org/downloads/ (enable **Add Python to PATH**).
2. Extract this ZIP to a normal folder.
3. Double-click `run_windows.bat`. It creates `.venv`, installs packages, creates `.env` from safe placeholders if needed, and starts Streamlit. The first run needs internet access to install dependencies.
4. For AI extraction and SOP drafting, open `.env` in Notepad and set your own `GEMINI_API_KEY`. Restart the app after changing `.env`. Tavily is optional; DuckDuckGo search is used as a fallback.

## Manual Windows setup (PowerShell)

From the extracted project root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
streamlit run app/main.py
```

If PowerShell blocks activation, use the current terminal without activation:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app/main.py
```

Streamlit normally opens `http://localhost:8501`. Keep the terminal open while using the app; press `Ctrl+C` there to stop it.

## API keys

- `GEMINI_API_KEY` is used for profile extraction, eligibility analysis, and SOP drafting.
- `TAVILY_API_KEY` is optional. If missing or failing, the app attempts DuckDuckGo search.
- `GITHUB_TOKEN` is optional and only increases GitHub API rate limits.

Put real credentials in `.env` only. The repository's `.env.example` contains placeholders. If a key was shared or committed, revoke/rotate it before use.

## Verify the agent workflow

The page shows four agent cards and a live event feed. A normal run should show Profile Agent, Research Agent, Eligibility Agent, and Ranking & Application Agent events, then the research summary, opportunities, top-five ranking, and editable SOP draft. Search requires internet access; Gemini features require a valid API key and available quota.

## Run tests

```powershell
pytest -q
```

Tests mock the LLM and web services, so they do not validate current scholarship facts or require working API keys. For live research, use the app and verify each result at its linked official source.

## Included sample profiles

The Rahul and Ananya profiles are fictional demo data. They use illustrative projects and should not be presented as real student achievements. Replace them with a consented student's information before drawing eligibility conclusions.

## Known limitations

- Some websites block automated retrieval or render content with JavaScript. Such results should be treated as lower-confidence and checked manually.
- Eligibility labels are preliminary assessments, not admission predictions.
- Funding/deadline claims must be confirmed on the official program page before applying.
- The current local test environment may not have package/network access; install the listed dependencies on your machine.

See `ARCHITECTURE.md` for the graph, agent responsibilities, trust design, and demo plan.
