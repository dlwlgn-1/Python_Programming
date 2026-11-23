# 감정 분류 모델 파인 튜닝
# - 일기 문장을 기쁨 / 슬픔 / 분노 / 불안 4가지로 분류하는 모델 만들기
# - 처음에 한 번만 실행하면 model 폴더에 모델이 저장됨

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "model")

LABELS = ["기쁨", "슬픔", "분노", "불안"]       # 0: 기쁨, 1: 슬픔, 2: 분노, 3: 불안


# 1. 데이터 준비
# - emotion.json 에 감정마다 일기 문장 50개씩 있음
# - 5번째 문장마다 평가용으로 빼고 나머지는 학습용으로 사용
with open(os.path.join(BASE_DIR, "data", "emotion.json"), encoding="utf-8") as f:
    emotion_data = json.load(f)

train_data = []
test_data = []

for label, name in enumerate(LABELS):
    for i, sentence in enumerate(emotion_data[name]):
        if i % 5 == 4:
            test_data.append((sentence, label))
        else:
            train_data.append((sentence, label))

print(f"학습용 {len(train_data)}개 / 평가용 {len(test_data)}개")


# 2. 기존 모델 불러오기 (분류 개수를 4개로 지정)
model_name = "monologg/koelectra-small-v3-discriminator"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    id2label={0: "기쁨", 1: "슬픔", 2: "분노", 3: "불안"},
)


# 3. 문장, 정답을 텐서로 바꾸기
def make_dataset(data):
    texts = [text for text, label in data]
    labels = torch.tensor([label for text, label in data])
    inputs = tokenizer(texts, padding=True, return_tensors="pt")
    return inputs, labels


train_inputs, train_labels = make_dataset(train_data)
test_inputs, test_labels = make_dataset(test_data)


# 4. 정확도 확인
def predict(inputs):
    model.eval()
    with torch.no_grad():
        logits = model(**inputs).logits
    return logits.argmax(dim=1)


def accuracy(inputs, labels):
    return (predict(inputs) == labels).float().mean().item()


print()
print("[학습 전]")
print(f"학습용 정확도 : {accuracy(train_inputs, train_labels):.0%}")
print(f"평가용 정확도 : {accuracy(test_inputs, test_labels):.0%}")


# 5. 학습하기 (예측 -> 오차 -> 수정 60번 반복)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

print()
print("[학습]")
for epoch in range(1, 61):
    model.train()
    outputs = model(**train_inputs, labels=train_labels)
    loss = outputs.loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"epoch {epoch:2d} | loss = {loss.item():.3f}")

print()
print("[학습 후]")
print(f"학습용 정확도 : {accuracy(train_inputs, train_labels):.0%}")
print(f"평가용 정확도 : {accuracy(test_inputs, test_labels):.0%}")


# 6. 모델 저장
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print("모델 저장 완료")
