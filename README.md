# CourseCompass — Web‑RAG Learning Path Builder (Education)

A portfolio-ready GenAI web app that turns any learning goal into a **credible, week-by-week learning plan** using:
- **Perplexity API** for live web retrieval (no URL pasting)
- **Web‑RAG** (extract → chunk → embed → top‑k evidence)
- **OpenAI API** for grounded synthesis (plan + curated resources + notes)
- A **premium Streamlit UI** with Sources/Evidence panels and Markdown export

---

## What this app does

Enter:
- a learning goal (e.g., “Learn LLM evaluation in 6 weeks”)
- your current level
- time budget

CourseCompass will:
1. Retrieve up-to-date learning resources from the web via **Perplexity**
2. Extract and chunk the content
3. Build embeddings and retrieve the most relevant evidence
4. Generate a structured plan + curated resources with citations like **[1]** referencing the Sources panel

> This is **web-RAG**, not local-file RAG: there is **no static `data/` folder**.

---

## Tech Stack
- Python
- Streamlit
- OpenAI API (embeddings + synthesis)
- Perplexity API (web retrieval)
- Trafilatura (clean article extraction)

---

## Local Setup & Run (macOS + PyCharm)

### 1) Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Add API keys
```bash
cp .env .env
```
Edit `.env` and set:
- `OPENAI_API_KEY`
- `PERPLEXITY_API_KEY`

✅ **Do not commit `.env`** (it’s ignored by `.gitignore`).

### 4) Run
```bash
streamlit run app.py
```
Open: http://localhost:8501

---

## Docker Setup (SETUP-2)

### Build
```bash
docker build -t coursecompass-web-rag .
```

### Run
```bash
docker run -p 8501:8501 --env-file .env coursecompass-web-rag
```

Open: http://localhost:8501

---

## Deploy on Render

### 1) Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit: CourseCompass Web-RAG"
git branch -M main
git remote add origin REPO_URL_HERE
git push -u origin main
```

### 2) Create a Render Web Service
- New → **Web Service**
- Connect your GitHub repo
- Environment: **Python**

**Build Command**
```bash
pip install -r requirements.txt
```

**Start Command**
```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

### 3) Add environment variables on Render
- `OPENAI_API_KEY`
- `PERPLEXITY_API_KEY`
- (Optional) `HTTP_TIMEOUT_S=30`

---

## .gitignore Reminder (CONSTRAINT)
- ✅ Commit `.gitignore`
- ❌ Never commit `.env`  
This repo includes `.gitignore` rules for `.env` and `.env.*`.

---

## About the Author
I have **5 years of experience** and I’m **actively looking for full-time GenAI / AI Engineer roles** (US).

- GitHub: https://github.com/bhavesh-kalluru  
- LinkedIn: https://www.linkedin.com/in/bhaveshkalluru/  
- Portfolio: https://kbhavesh.com  
- Project repo: REPO_URL_HERE
