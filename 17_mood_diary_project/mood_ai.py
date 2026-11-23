# AI 모듈
# - 파인 튜닝한 모델로 일기를 기쁨 / 슬픔 / 분노 / 불안으로 분류

import os
import streamlit as st
from transformers import pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# 감정별 마음 날씨
WEATHER = {
    "기쁨": "☀️ 맑음",
    "슬픔": "🌧️ 비",
    "분노": "⛈️ 천둥번개",
    "불안": "🌫️ 안개",
}

# 감정별 응원 메시지
MESSAGES = {
    "기쁨": "좋은 일이 있었네요! 오늘 기분을 오래 기억해요.",
    "슬픔": "많이 속상했겠어요. 비가 그치면 맑은 날도 와요.",
    "분노": "화가 많이 났군요. 숨 크게 쉬고 잠깐 쉬어 봐요.",
    "불안": "걱정이 많았네요. 하나씩 천천히 하면 괜찮아요.",
}


# 모델은 처음 한 번만 불러오고 계속 재사용
@st.cache_resource(show_spinner="모델 불러오는 중...")
def get_classifier():
    return pipeline("text-classification", model=MODEL_DIR)


# 일기 분류
# - top_k=None : 4가지 감정의 확률을 모두 받음 (확률 높은 순서)
# - 결과 예) [{'label': '기쁨', 'score': 0.91}, {'label': '슬픔', 'score': 0.04}, ...]
def classify(text):
    return get_classifier()(text, top_k=None)
