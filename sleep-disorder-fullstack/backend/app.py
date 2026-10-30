import os, io, json, sqlite3
from datetime import datetime
from functools import wraps
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
DB_PATH=os.getenv('DATABASE_PATH', os.path.join(BASE_DIR,'sleep_app.db'))
app=Flask(__name__)
app.config['SECRET_KEY']=os.getenv('SECRET_KEY','change-this-development-secret')
CORS(app, resources={r'/api/*': {'origins': os.getenv('FRONTEND_ORIGIN','http://localhost:5173')}}, supports_credentials=True)

FEATURES=['Gender','Age','Occupation','Sleep Duration','Quality of Sleep','Physical Activity Level','Stress Level','BMI Category','Blood Pressure','Heart Rate','Daily Steps']
TARGET='Sleep Disorder'
NUMERIC=['Age','Sleep Duration','Quality of Sleep','Physical Activity Level','Stress Level','Heart Rate','Daily Steps']
CATEGORICAL=['Gender','Occupation','BMI Category','Blood Pressure']
LABELS=['None','Sleep Apnea','Insomnia']
model=None; metrics={}; model_source='demo data'; training_rows=0

def connect_db():
    con=sqlite3.connect(DB_PATH); con.row_factory=sqlite3.Row; return con

def init_db():
    with connect_db() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, full_name TEXT NOT NULL, email TEXT UNIQUE NOT NULL, phone TEXT, password_hash TEXT NOT NULL, created_at TEXT NOT NULL)""")
        con.execute("""CREATE TABLE IF NOT EXISTS predictions(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, input_json TEXT NOT NULL, prediction TEXT NOT NULL, probabilities TEXT, created_at TEXT NOT NULL, FOREIGN KEY(user_id) REFERENCES users(id))""")
        con.commit()

def make_demo_data(n=420, seed=42):
    """Synthetic data only for demonstrating the application; not real patient data."""
    rng=np.random.default_rng(seed); rows=[]
    occupations=['Nurse','Doctor','Engineer','Lawyer','Teacher','Salesperson','Accountant','Scientist']
    for _ in range(n):
        age=int(rng.integers(20,70)); gender=rng.choice(['Male','Female']); duration=round(float(np.clip(rng.normal(7,1.2),4,10)),1)
        quality=int(np.clip(round(rng.normal(7,1.5)),3,9)); activity=int(rng.integers(20,90)); stress=int(rng.integers(1,10)); bmi=rng.choice(['Normal','Overweight','Obese','Normal Weight'],p=[.35,.32,.18,.15]); hr=int(rng.integers(55,100)); steps=int(rng.integers(2000,12000)); bp=rng.choice(['110/70','120/80','130/85','140/90','125/82'])
        apnea=(bmi=='Obese' or bmi=='Overweight') and (age>45 or hr>82 or bp in ['140/90','130/85'])
        insomnia=(stress>=7 and (duration<6.5 or quality<=5))
        label='Sleep Apnea' if apnea and (not insomnia or rng.random()<.6) else ('Insomnia' if insomnia else 'None')
        rows.append({'Gender':gender,'Age':age,'Occupation':rng.choice(occupations),'Sleep Duration':duration,'Quality of Sleep':quality,'Physical Activity Level':activity,'Stress Level':stress,'BMI Category':bmi,'Blood Pressure':bp,'Heart Rate':hr,'Daily Steps':steps,'Sleep Disorder':label})
    return pd.DataFrame(rows)

def train(df, source):
    global model, metrics, model_source, training_rows
    df=df.copy(); df.columns=[str(c).strip() for c in df.columns]
    # Support common alternate names in public copies of the dataset.
    aliases={'Sleep Duration (hours)':'Sleep Duration','Quality of Sleep (scale: 1-10)':'Quality of Sleep','Physical Activity Level (minutes/day)':'Physical Activity Level','Stress Level (scale: 1-10)':'Stress Level','Daily Steps (count)':'Daily Steps'}
    df=df.rename(columns=aliases)
    missing=[c for c in FEATURES+[TARGET] if c not in df.columns]
    if missing: raise ValueError('Dataset is missing required columns: '+', '.join(missing))
    df=df[FEATURES+[TARGET]].drop_duplicates().dropna(subset=[TARGET])
    df[TARGET]=df[TARGET].astype(str).str.strip().replace({'None':'None','none':'None','Sleep Apnea':'Sleep Apnea','Insomnia':'Insomnia'})
    df=df[df[TARGET].isin(LABELS)]
    if len(df)<15 or df[TARGET].nunique()<2: raise ValueError('At least 15 usable rows and two target classes are required.')
    X=df[FEATURES]; y=df[TARGET]
    num=Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())])
    cat=Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore'))])
    prep=ColumnTransformer([('num',num,NUMERIC),('cat',cat,CATEGORICAL)])
    estimators=[('dt',DecisionTreeClassifier(max_depth=8,random_state=42)),('rf',RandomForestClassifier(n_estimators=120,random_state=42,class_weight='balanced'))]
    clf=VotingClassifier(estimators=estimators,voting='hard')
    pipe=Pipeline([('preprocess',prep),('classifier',clf)])
    strat=y if y.value_counts().min()>=2 else None
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=strat)
    pipe.fit(Xtr,ytr); pred=pipe.predict(Xte)
    model=pipe; model_source=source; training_rows=len(df)
    metrics={'accuracy':round(float(accuracy_score(yte,pred)),4),'precision_weighted':round(float(precision_score(yte,pred,average='weighted',zero_division=0)),4),'recall_weighted':round(float(recall_score(yte,pred,average='weighted',zero_division=0)),4),'f1_weighted':round(float(f1_score(yte,pred,average='weighted',zero_division=0)),4),'test_records':len(yte),'training_records':len(ytr),'dataset_records':len(df),'labels':sorted(y.unique().tolist()),'confusion_matrix':confusion_matrix(yte,pred,labels=sorted(y.unique())).tolist()}
    return metrics

def require_user(fn):
    @wraps(fn)
    def wrapped(*args,**kwargs):
        uid=request.headers.get('X-User-Id')
        if not uid: return jsonify({'error':'Please log in to use this feature.'}),401
        with connect_db() as con: user=con.execute('SELECT id,username,full_name,email FROM users WHERE id=?',(uid,)).fetchone()
        if not user: return jsonify({'error':'Session user was not found. Please log in again.'}),401
        g.user=dict(user)
        return fn(*args,**kwargs)
    return wrapped

@app.get('/api/health')
def health(): return jsonify({'status':'ok','model_ready':model is not None,'model_source':model_source,'training_rows':training_rows})

@app.post('/api/auth/register')
def register():
    d=request.get_json(silent=True) or {}; username=str(d.get('username','')).strip(); full=str(d.get('full_name','')).strip(); email=str(d.get('email','')).strip().lower(); password=str(d.get('password','')); phone=str(d.get('phone','')).strip()
    if not username or not full or not email or len(password)<8: return jsonify({'error':'Username, full name, email and password (at least 8 characters) are required.'}),400
    try:
        with connect_db() as con:
            cur=con.execute('INSERT INTO users(username,full_name,email,phone,password_hash,created_at) VALUES(?,?,?,?,?,?)',(username,full,email,phone,generate_password_hash(password),datetime.utcnow().isoformat()))
            con.commit(); uid=cur.lastrowid
        return jsonify({'user':{'id':uid,'username':username,'full_name':full,'email':email}}),201
    except sqlite3.IntegrityError: return jsonify({'error':'Username or email already exists.'}),409

@app.post('/api/auth/login')
def login():
    d=request.get_json(silent=True) or {}; login_id=str(d.get('login','')).strip(); password=str(d.get('password',''))
    with connect_db() as con: u=con.execute('SELECT * FROM users WHERE username=? OR email=?',(login_id,login_id.lower())).fetchone()
    if not u or not check_password_hash(u['password_hash'],password): return jsonify({'error':'Invalid username/email or password.'}),401
    return jsonify({'user':{'id':u['id'],'username':u['username'],'full_name':u['full_name'],'email':u['email']}})

@app.get('/api/me')
@require_user
def me(): return jsonify({'user':g.user})

@app.post('/api/predict')
@require_user
def predict():
    if model is None: return jsonify({'error':'Model is not ready. Train or reload the model first.'}),503
    d=request.get_json(silent=True) or {}
    try:
        row={k:d.get(k) for k in FEATURES}
        for k in ['Age','Sleep Duration','Quality of Sleep','Physical Activity Level','Stress Level','Heart Rate','Daily Steps']:
            row[k]=float(row[k])
            if not np.isfinite(row[k]): raise ValueError(k+' must be a valid number.')
        if not 18<=row['Age']<=100: raise ValueError('Age must be between 18 and 100.')
        if not 0<row['Sleep Duration']<=24: raise ValueError('Sleep duration must be between 0 and 24 hours.')
        if not 1<=row['Quality of Sleep']<=10: raise ValueError('Quality of sleep must be between 1 and 10.')
        if not 0<=row['Physical Activity Level']<=1440: raise ValueError('Physical activity must be between 0 and 1440 minutes.')
        if not 1<=row['Stress Level']<=10: raise ValueError('Stress level must be between 1 and 10.')
        if not 30<=row['Heart Rate']<=220: raise ValueError('Heart rate must be between 30 and 220 bpm.')
        if not 0<=row['Daily Steps']<=100000: raise ValueError('Daily steps must be between 0 and 100000.')
        for k in ['Gender','Occupation','BMI Category','Blood Pressure']:
            if not str(row[k] or '').strip(): raise ValueError('Please provide '+k+'.')
            row[k]=str(row[k]).strip()
        X=pd.DataFrame([row],columns=FEATURES); label=str(model.predict(X)[0]); probs={}
        if hasattr(model,'predict_proba'):
            vals=model.predict_proba(X)[0]; probs={str(c):round(float(p),4) for c,p in zip(model.classes_,vals)}
        now=datetime.utcnow().isoformat()
        with connect_db() as con:
            con.execute('INSERT INTO predictions(user_id,input_json,prediction,probabilities,created_at) VALUES(?,?,?,?,?)',(g.user['id'],json.dumps(row),label,json.dumps(probs),now)); con.commit()
        return jsonify({'prediction':label,'probabilities':probs,'created_at':now,'model_source':model_source,'notice':'Educational decision-support estimate only; not a medical diagnosis.'})
    except (ValueError,TypeError) as e: return jsonify({'error':str(e)}),400
    except Exception as e: return jsonify({'error':'Prediction failed. Check the submitted values and try again.'}),500

@app.get('/api/predictions')
@require_user
def predictions():
    with connect_db() as con: rows=con.execute('SELECT * FROM predictions WHERE user_id=? ORDER BY id DESC LIMIT 100',(g.user['id'],)).fetchall()
    return jsonify({'predictions':[{'id':r['id'],'input':json.loads(r['input_json']),'prediction':r['prediction'],'probabilities':json.loads(r['probabilities'] or '{}'),'created_at':r['created_at']} for r in rows]})

@app.delete('/api/predictions/<int:pid>')
@require_user
def delete_prediction(pid):
    with connect_db() as con: cur=con.execute('DELETE FROM predictions WHERE id=? AND user_id=?',(pid,g.user['id'])); con.commit()
    if cur.rowcount==0: return jsonify({'error':'Prediction not found.'}),404
    return jsonify({'message':'Prediction deleted.'})

@app.get('/api/model/metrics')
def get_metrics(): return jsonify({'metrics':metrics,'model_source':model_source,'training_rows':training_rows,'features':FEATURES,'target':TARGET})

@app.post('/api/model/train')
@require_user
def upload_train():
    if 'file' not in request.files: return jsonify({'error':'Upload a CSV file using the file field.'}),400
    f=request.files['file']
    if not f.filename.lower().endswith('.csv'): return jsonify({'error':'Only CSV files are supported.'}),400
    try:
        df=pd.read_csv(io.BytesIO(f.read())); result=train(df,'uploaded: '+f.filename)
        return jsonify({'message':'Model trained successfully.','metrics':result,'model_source':model_source})
    except Exception as e: return jsonify({'error':str(e)}),400

@app.errorhandler(404)
def not_found(_): return jsonify({'error':'API endpoint not found.'}),404

init_db()
try: train(pd.read_csv(os.path.join(BASE_DIR,'data','sleep_health_lifestyle.csv')),'bundled CSV dataset')
except Exception: train(make_demo_data(),'synthetic demo data (not real patient records)')

if __name__=='__main__': app.run(host='0.0.0.0',port=int(os.getenv('PORT','5000')),debug=os.getenv('FLASK_DEBUG','false').lower()=='true')
