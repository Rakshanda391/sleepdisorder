# SomniCare — Sleep Disorder Classification

A beginner-friendly full-stack student project based on the uploaded paper, *Applying Machine Learning Algorithms for the Classification of Sleep Disorders*. It has a React frontend, Flask REST API, SQLite persistence, registration/login, prediction history, CSV model training, and model evaluation metrics.

> **Important:** This is an educational prototype, not a medical device or diagnostic service. The application falls back to a clearly identified synthetic dataset when no suitable CSV is bundled. Synthetic data and its scores must not be presented as real clinical performance. For research reproduction, supply the original dataset used by the paper.

## 1. Stack
- React + Vite + JavaScript — responsive single-page interface.
- Python + Flask — REST endpoints and input validation.
- scikit-learn + pandas — preprocessing, ensemble classifier, and evaluation.
- SQLite — user accounts and prediction history.

## 2. Structure
```text
sleep-disorder-fullstack/
├── backend/app.py
├── backend/requirements.txt
├── backend/.env.example
├── backend/data/                 # optional original dataset CSV goes here
├── frontend/index.html
├── frontend/package.json
├── frontend/.env.example
├── frontend/src/App.jsx
├── frontend/src/main.jsx
├── frontend/src/styles.css
├── frontend/src/services/api.js
├── database/schema.sql
├── .gitignore
└── README.md
```

## 3. Run locally
Requirements: Python 3.10+ and Node.js 18+ (current LTS recommended).

### Backend
```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
# Optional: copy .env.example to .env and configure environment variables.
python app.py
```
API starts at `http://localhost:5000`. The app creates `backend/sleep_app.db` and initializes the model at startup.

### Frontend (new terminal)
```bash
cd frontend
npm install
# Optional: copy .env.example to .env
npm run dev
```
Open `http://localhost:5173`.

## 4. Dataset format
Use a CSV containing these columns (column names are case-sensitive after trimming, with a few common unit-suffixed aliases supported):

`Gender, Age, Occupation, Sleep Duration, Quality of Sleep, Physical Activity Level, Stress Level, BMI Category, Blood Pressure, Heart Rate, Daily Steps, Sleep Disorder`

The target `Sleep Disorder` must contain at least two of `None`, `Sleep Apnea`, `Insomnia`. Upload from **Model insights → Choose CSV**. The API removes duplicates, drops missing targets, applies median/mode imputation, scales numeric features, one-hot encodes categorical features, and evaluates a voting ensemble with an 80/20 split. Uploaded training updates the in-memory model; restart returns to the bundled/fallback dataset. The original PDF describes 400 records and 13 attributes, while this app uses the 11 available predictor fields shown in its illustrated form plus the target column. If your source dataset contains an extra identifier column such as Person ID, it is ignored.

## 5. API reference
All endpoints are prefixed with `/api`. JSON errors use `{ "error": "message" }`. Authenticated endpoints expect `X-User-Id` from the successful login response. This is a simple college-demo session mechanism, not production-grade authentication; use signed tokens/sessions and CSRF protection before deployment.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | API/model readiness |
| POST | `/api/auth/register` | Create account: `username, full_name, email, password, phone?` |
| POST | `/api/auth/login` | Login: `login` (username or email), `password` |
| GET | `/api/me` | Current user profile; requires `X-User-Id` |
| POST | `/api/predict` | Predict and save result; requires `X-User-Id` and all form fields |
| GET | `/api/predictions` | List current user's latest 100 results; requires `X-User-Id` |
| DELETE | `/api/predictions/:id` | Delete own result; requires `X-User-Id` |
| GET | `/api/model/metrics` | Current model metrics and feature names |
| POST | `/api/model/train` | Multipart form with `file=<CSV>`; requires `X-User-Id` |

### Example prediction body
```json
{
  "Gender":"Female", "Age":32, "Occupation":"Engineer",
  "Sleep Duration":7, "Quality of Sleep":7,
  "Physical Activity Level":45, "Stress Level":4,
  "BMI Category":"Normal", "Blood Pressure":"120/80",
  "Heart Rate":72, "Daily Steps":7000
}
```
Response includes `prediction`, optional `probabilities`, `created_at`, `model_source`, and a medical-use disclaimer. The model currently uses a hard-voting ensemble; class probabilities are only returned when supported by the estimator.

## 6. Database
SQLite file is created automatically at `backend/sleep_app.db`. The schema is also provided in `database/schema.sql`. `users` stores password hashes (never plaintext passwords); `predictions` stores the input JSON, predicted class, optional class scores, timestamp, and user relationship.

## 7. Testing checklist
These are steps for you to run; do not claim the checks passed until you execute them on your machine.
1. `GET http://localhost:5000/api/health` should return `status: ok` and `model_ready: true`.
2. Register a user, then login. Try a duplicate username/email (expect HTTP 409) and a password shorter than 8 characters (expect HTTP 400).
3. Log in, generate a prediction, and confirm it appears in Prediction history.
4. Try age below 18, quality above 10, missing text values, and invalid CSV; expect readable 400-level errors.
5. Delete a history row and refresh history; it should remain deleted.
6. Upload a valid labelled CSV; metrics should update. Upload a non-CSV or CSV missing required columns; expect an error.
7. Run `npm run build` in `frontend` to verify the production frontend build.

## 8. Troubleshooting
- **Cannot reach backend:** start Flask first and check port 5000.
- **CORS error:** confirm `FRONTEND_ORIGIN=http://localhost:5173` and restart Flask.
- **Missing Python packages:** activate the virtual environment and rerun `pip install -r requirements.txt`.
- **CSV column error:** check exact column names and target labels; ensure at least 15 usable rows and two classes.
- **Scores differ from the paper:** expected unless the same dataset, preprocessing, split, and model configuration are reproduced. Synthetic demo data is not the paper's dataset.
- **Login isn't production secure:** this is a learning prototype; do not expose it publicly without proper session/JWT authentication, HTTPS, rate limiting, and security review.

## 9. Architecture
```text
React UI (5173) → REST API / Flask (5000) → scikit-learn pipeline
                              │                     ↑
                              └── SQLite database   └── CSV dataset / preprocessing
```

## 10. Future enhancements
- Reproduce and compare all six algorithms listed in the paper, using a fixed random seed and cross-validation.
- Add a confusion-matrix visualization and per-class metrics.
- Add secure session/JWT authentication, password reset, rate limiting, and audit controls.
- Use a larger, ethically sourced and appropriately de-identified dataset; assess class imbalance and subgroup performance.
- Explore wearable integration only with consent and validated data handling.


## 8. Windows CMD: run locally

Open **two separate CMD windows**. Install Python 3.10+ and Node.js 18+ first.

**Terminal 1 — Flask backend**
```bat
cd path\to\sleep-disorder-fullstack\backend
py -m venv .venv
.venv\Scripts\activate
py -m pip install --upgrade pip
pip install -r requirements.txt
py app.py
```
The API should be available at `http://127.0.0.1:5000/api/health`.

**Terminal 2 — React frontend**
```bat
cd path\to\sleep-disorder-fullstack\frontend
npm install
copy .env.example .env
npm run dev
```
Open `http://localhost:5173`. Keep both terminals running. If `py` is not recognized, use `python` instead.

## 9. GitHub upload from Windows CMD

Extract this ZIP first, then run:
```bat
cd path\to\sleep-disorder-fullstack
git init
git add .
git commit -m "Add sleep disorder full stack app"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```
Replace the remote URL with your own repository URL. If `origin` already exists, use `git remote set-url origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git`.

The project contains fewer than 100 source files. GitHub's 100-file limit applies to selecting/uploading files in one batch through the website, not the total number of files in a Git repository. Pushing with Git is the better option.

## 10. Deploy separately (example: Vercel frontend + Render backend)

Pushing to GitHub does **not** by itself publish the application. Create two services connected to the same GitHub repository:

**A. Flask backend on Render**
1. Create a new **Web Service** and select this repository.
2. Set **Root Directory** to `backend`.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app --bind 0.0.0.0:$PORT`
5. Add environment variables:
   - `SECRET_KEY`: a long random secret
   - `FRONTEND_ORIGIN`: your deployed frontend origin, e.g. `https://your-app.vercel.app`
6. Deploy and copy the backend service URL. Test `https://YOUR-BACKEND.onrender.com/api/health`.

**B. React frontend on Vercel**
1. Import the same GitHub repository as a new Vercel project.
2. Set **Root Directory** to `frontend`.
3. Framework preset: Vite.
4. Build command: `npm run build`; output directory: `dist`.
5. Add environment variable `VITE_API_URL` with the backend API base URL, including `/api`, e.g. `https://YOUR-BACKEND.onrender.com/api`.
6. Deploy/redeploy after setting the environment variable.

Because Vite environment variables are embedded at build time, set `VITE_API_URL` before redeploying. Also set Render's `FRONTEND_ORIGIN` to the exact deployed frontend origin (no trailing slash) and redeploy the backend if changed.

**Important deployment note:** SQLite data stored on an ephemeral hosting filesystem may be lost when a service restarts or redeploys. For persistent user accounts and prediction history, configure a persistent disk or migrate to a managed database. The demo fallback data is synthetic and is not for medical diagnosis.
