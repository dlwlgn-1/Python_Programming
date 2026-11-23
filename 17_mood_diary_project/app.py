# AI + DB + Web 연동 프로젝트 : 마음날씨 AI 일기장
# - 일기 작성 -> AI가 감정 분류 -> 마음 날씨로 보여주기 -> DB 저장
# - 일기 조회 / 수정 / 삭제, 감정 통계
#
# [모듈 구성]
# - app.py : 메인 모듈 (웹 화면)
# - mood_ai.py : 감정 분류 AI
# - diary_db.py : 일기 DB 처리
# - train.py : 감정 분류 모델 학습 (처음에 한 번 실행)

import streamlit as st
import pandas as pd
from datetime import date

import mood_ai as ai
import diary_db as db


try:
    st.title("🌦️ 마음날씨 AI 일기장")

    # DB 초기화
    if "init_db" not in st.session_state:
        st.session_state.init_db = True
        db.init_db()


    # 1. 일기 쓰기
    # - 저장 버튼을 누르면 AI가 감정을 분류하고 DB에 저장
    st.header("1. 일기 쓰기")

    with st.form("diary_form", clear_on_submit=True):
        diary_date = st.date_input("날짜", value=date.today())
        content = st.text_area("오늘 하루는 어땠나요?", placeholder="예) 체육대회에서 우리 반이 우승했다")
        submitted = st.form_submit_button("저장")

    if submitted:
        content = content.strip()

        if not content:
            st.warning("일기 내용을 입력해 주세요.")
        else:
            with st.spinner("감정 분석 중..."):
                results = ai.classify(content)

            emotion = results[0]["label"]       # 확률이 가장 높은 감정
            score = results[0]["score"]

            db.save_diary(str(diary_date), content, emotion, score)
            st.success("저장했습니다.")

            st.subheader(f"오늘의 마음 날씨 : {ai.WEATHER[emotion]}")
            st.write(f"AI 분석 결과 : {emotion} ({score:.0%})")
            st.info(ai.MESSAGES[emotion])

            # 감정별 확률 그래프
            score_df = pd.DataFrame(results)
            score_df.columns = ["감정", "확률"]
            st.bar_chart(score_df, x="감정", y="확률", horizontal=True)

    st.divider()


    # 2. 일기장
    # - 감정을 골라서 검색
    # - 표에서 일기를 선택하면 수정, 삭제 가능
    st.header("2. 일기장")

    choice = st.radio("감정", ["전체", "기쁨", "슬픔", "분노", "불안"], horizontal=True)
    rows = db.load_diaries(emotion=None if choice == "전체" else choice)

    if not rows:
        st.info("저장된 일기가 없습니다.")
    else:
        df = pd.DataFrame(rows, columns=["번호", "날짜", "내용", "감정", "확률", "작성시각"])
        df["날씨"] = [ai.WEATHER[emotion] for emotion in df["감정"]]
        df["확률"] = df["확률"].round(2)

        event = st.dataframe(
            df[["번호", "날짜", "날씨", "내용", "확률"]],
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
        )

        selected = event.selection.rows         # 선택한 행 번호

        if selected and selected[0] < len(df):
            row = df.iloc[selected[0]]
            diary_id = int(row["번호"])

            st.subheader(f"{row['날짜']}  {row['날씨']}")
            st.info(ai.MESSAGES[row["감정"]])

            new_content = st.text_area("일기 내용", value=row["내용"])

            if st.button("수정하기"):
                # 바뀐 내용으로 감정을 다시 분석해서 저장
                results = ai.classify(new_content)
                db.update_diary(diary_id, new_content, results[0]["label"], results[0]["score"])
                st.rerun()      # 화면을 다시 실행해서 표 새로고침

            if st.button("삭제하기"):
                db.delete_diary(diary_id)
                st.rerun()

    st.divider()


    # 3. 마음 통계
    # - 감정별로 일기가 몇 개인지 세서 그래프로 보여주기
    st.header("3. 마음 통계")

    all_rows = db.load_diaries()

    if not all_rows:
        st.info("일기를 쓰면 통계를 볼 수 있습니다.")
    else:
        counts = {"기쁨": 0, "슬픔": 0, "분노": 0, "불안": 0}
        for row in all_rows:
            counts[row[3]] += 1                 # row[3] : 감정

        top = max(counts, key=counts.get)       # 개수가 가장 많은 감정

        st.write(f"전체 일기 : {len(all_rows)}개")
        st.write(f"가장 많은 마음 날씨 : {ai.WEATHER[top]}")

        count_df = pd.DataFrame({"감정": list(counts.keys()), "개수": list(counts.values())})
        st.bar_chart(count_df, x="감정", y="개수", horizontal=True)

        # 최근에 쓴 일기 7개의 마음 날씨
        st.subheader("최근 마음 날씨")
        recent = []
        for row in all_rows[-7:]:
            recent.append(f"{row[1][5:]} {ai.WEATHER[row[3]]}")     # 예) 11-23 ☀️ 맑음
        st.write("  →  ".join(recent))

except Exception as e:
    st.error("실행 중 오류가 발생했습니다.")
    print(f"에러 메시지: {e}")
