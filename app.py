from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

templates = Jinja2Templates(directory="templates")
model = xgb.XGBClassifier()
model.load_model('best_xgb_model.json')
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_classes.npy', allow_pickle=True)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse('home.html', {"request": request})

@app.post("/recommendation", response_class=HTMLResponse)
async def recommendation(request: Request, 
                         gender: str = Form(...), 
                         age: str = Form(...), 
                         taste: str = Form(...), 
                         season: str = Form(...)):
    
    # 입력 데이터를 모델에 맞게 전처리
    input_data = process_input(gender, age, taste, season)
    menu_prediction = predict_menu(input_data)
    
    return templates.TemplateResponse('recommendation.html', {"request": request, "menu": menu_prediction})

@app.get("/settings", response_class=HTMLResponse)
async def settings(request: Request):
    return templates.TemplateResponse('settings.html', {"request": request})

def process_input(gender, age, taste, season):
    # 여기에 입력 데이터를 처리하는 코드를 작성합니다.
    processed_data = [...]  # 예시: 적절한 전처리 코드를 추가합니다.
    return processed_data

def predict_menu(input_data):
    prediction = model.predict(input_data)
    predicted_label = label_encoder.inverse_transform(prediction)
    return predicted_label[0]

