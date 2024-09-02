from flask import Flask, render_template, request, redirect, url_for
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# 머신러닝 모델 및 레이블 인코더 불러오기
model = xgb.XGBClassifier()
model.load_model('best_xgb_model.json')
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_classes.npy',allow_pickle=True)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/recommendation', methods=['GET', 'POST'])
def recommendation():
    if request.method == 'POST':
        gender = request.form['gender']
        age = request.form['age']
        taste = request.form['taste']
        season = request.form['season']
        
        # 입력 데이터를 모델에 맞게 전처리
        input_data = process_input(gender, age, taste, season)
        menu_prediction = predict_menu(input_data)
        
        return render_template('recommendation.html', menu=menu_prediction)
    return render_template('recommendation.html')

@app.route('/settings')
def settings():
    return render_template('settings.html')

def process_input(gender, age, taste, season):
    # 여기에 입력 데이터를 처리하는 코드를 작성합니다.
    processed_data = [...]  # 예시: 적절한 전처리 코드를 추가합니다.
    return processed_data

def predict_menu(input_data):
    prediction = model.predict(input_data)
    predicted_label = label_encoder.inverse_transform(prediction)
    return predicted_label[0]

if __name__ == '__main__':
    app.run(debug=True)
