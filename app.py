import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ====================================================================
# [조건 3 & 조건 4] 웹 서비스 구현 및 실시간 예측 UI 구성 (app.py)
# ====================================================================

# 1. 저장된 pkl 파일 로드
payload = joblib.load("life_expectancy_pipeline.pkl")  # 디스크에 저장되어 있는 최적화 파이프라인 패키지 데이터 로드
models = payload["models"]  # 딕셔너리에서 추출: 학습이 완료된 3종의 파이프라인 모델 객체들
df_perf = payload["df_perf"]  # 딕셔너리에서 추출: 3가지 모델의 종합 성능 평가지표 판다스 데이터프레임
stats = payload["stats"]  # 딕셔너리에서 추출: 입력 슬라이더 제한 범위를 조절하기 위한 소문자 특성 통계치

# 2. 대시보드 레이아웃 설정
st.set_page_config(page_title="기대수명 예측 시스템", layout="wide")  # 웹 브라우저의 페이지 제목 및 화면 해상도 와이드 모드 설정
st.title("🧬 다중 특성 회귀 파이프라인 기반 기대수명 예측 서비스")  # 웹 대시보드 최상단 메인 타이틀 텍스트 출력
st.markdown("WHO 데이터를 활용하여 고차원 다항 회귀의 과대적합을 분석하고 규제 효과를 검증합니다.")  # 대시보드 하위 명세 설명 기술

# 3. 사이드바 UI
st.sidebar.header("📋 신체 및 경제 지표 입력")  # 왼쪽 사이드바 영역에 입력 인터페이스 헤더 텍스트 출력
u_am = st.sidebar.slider("성인 사망률 (Adult Mortality)", stats["am_min"], stats["am_max"], stats["am_mean"])  # 실제 adult mortality 분포 범위를 활용한 슬라이더 생성
u_bmi = st.sidebar.slider("체질량지수 (BMI)", stats["bmi_min"], stats["bmi_max"], stats["bmi_mean"])  # 실제 bmi 분포 범위를 활용한 슬라이더 생성
u_gdp = st.sidebar.slider("국내총생산 (GDP)", stats["gdp_min"], stats["gdp_max"], stats["gdp_mean"])  # 실제 gdp 분포 범위를 활용한 슬라이더 생성

# 모델 선택 인터페이스
selected_model_name = st.sidebar.selectbox("예측에 사용할 파이프라인 모델 선택:", ["Linear", "Poly", "Ridge"])  # 사용자가 [Linear, Poly, Ridge] 중 하나를 선택할 드롭다운 생성
current_pipeline = models[selected_model_name]  # 사용자가 선택한 모델명 키값에 매칭되는 학습 완료 파이프라인 모델 추출

# 4. 메인 화면 - 성능 비교 섹션 구현
st.subheader("📊 1단계: 모델별 성능 지표 비교 (Linear vs Poly vs Ridge)")  # 메인 본문 영역 성능 지표 비교 섹션 서브 헤더 출력

# (1) 테이블 출력
st.write("### 📈 모델 평가 지표 테이블 (R² 및 MSE)")  # 평가지표 테이블 타이틀 명시
st.dataframe(df_perf.set_index("Model"), use_container_width=True)  # 'Model' 컬럼을 인덱스로 잡고 웹 화면 너비에 맞게 전체 성능 비교 테이블 상시 출력

# (2) 막대그래프 시각화
col1, col2 = st.columns([2, 1])  # 시각화 그래프와 리포트 영역을 깔끔하게 분할하기 위한 가로 2:1 비율 레이아웃 컬럼 생성
with col1:
    st.write("### 📉 모델별 예측 성능 (Test R² Score) 비교")  # 1번 컬럼 내부에 막대그래프 타이틀 명시
    fig, ax = plt.subplots(figsize=(7, 3.5))  # matplotlib을 이용해 시각화 차트를 그리기 위한 도화지 및 축 객체 생성

    colors = ['dodgerblue' if x == 'Linear' else 'red' if x == 'Poly' else 'green' for x in df_perf['Model']]  # 모델명 조건에 따라 블루, 레드, 그린 순서로 다른 막대 색상 리스트 정의
    bars = ax.bar(df_perf['Model'], df_perf['Test R2'], color=colors, width=0.5, edgecolor='black')  # x축 모델명, y축 Test R2 스코어를 기반으로 한 막대그래프 렌더링

    ax.set_ylabel("Test R2 Score", fontsize=10)  # y축 세로 레이블 명칭 및 폰트 크기 지정
    ax.set_title("Model Comparison: Higher is Better", fontsize=11, fontweight='bold')  # 그래프 최상단 세부 타이틀 서식 조정

    # 그래프 y축 범위를 동적으로 잡아줌 (과대적합 시 마이너스로 떨어지는 것 방어)
    min_test_r2 = df_perf['Test R2'].min()  # 오버피팅으로 극단적으로 깎인 Test R2의 최소 점수를 탐색
    ax.set_ylim(min_test_r2 - 0.2 if min_test_r2 < 0 else 0, 1.1)  # 마이너스 값 존재 유무에 따라 상하한 스케일을 유연하게 보정
    ax.grid(True, linestyle=':', alpha=0.6)  # 시각적 가독성 확보를 위해 배경에 흐릿한 격자 점선 추가

    for bar in bars:
        height = bar.get_height()  # 개별 막대의 세로 높이(Test R2 점수 데이터값)를 추출
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -10),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top', fontsize=9, fontweight='bold')  # 각 막대의 상단 혹은 하단 경계면에 텍스트 점수 라벨 출력

    st.pyplot(fig)  # 생성 및 스타일 처리가 완료된 matplotlib 차트 피겨 객체를 streamlit 웹 대시보드에 렌더링

with col2:
    st.write("### 💡 규제 모델 분석 리포트")  # 2번 컬럼 내부에 리포트 섹션 타이틀 명시
    poly_complex = df_perf.loc[df_perf['Model'] == 'Poly', 'Complexity'].values[0]  # 성적표 데이터프레임 내부에서 Poly 모델이 갖는 특성 복잡도 개수 추출
    ridge_test_r2 = df_perf.loc[df_perf['Model'] == 'Ridge', 'Test R2'].values[0]  # 성적표 데이터프레임 내부에서 Ridge 모델의 최종 일반화 Test R2 점수 추출

    st.info(f"독립변수 3개가 3차 다항식으로 변환되며 **{poly_complex}개**의 복합 특성으로 확장되었습니다.\n\n"
            f"**Ridge 규제 모델**은 계수 크기를 제어하여 **Test R²: {ridge_test_r2:.4f}**의 일반화를 이뤄냅니다.")  # 동적으로 가공된 분석 메시지를 정보성 상자(st.info) 형태로 고정 출력

# 5. 실시간 예측 결과 출력
st.divider()  # 상단 평가지표 분석 영역과 하단 실시간 동적 예측 구역을 나누는 세련된 수평 구분선 배치
st.subheader("🎯 2단계: 선택된 모델의 실시간 기대수명 진단 결과")  # 2단계 동적 예측 영역 서브 헤더 출력

new_data = np.array([[u_am, u_bmi, u_gdp]])  # 사이드바 슬라이더를 통해 사용자가 지정한 3개 변수 입력 값을 scikit-learn 규격인 2차원 배열로 전환
prediction = current_pipeline.predict(new_data)  # 선택된 모델 파이프라인의 스케일러 및 다항변환 단계를 거쳐 실시간 최종 예측 결과 산출

st.metric(label=f"🏃‍♂️ 현재 {selected_model_name} 모델 기반 예측치", value=f"{prediction[0]:.2f} 세")  # 최종 수명 예측 결과 값을 대시보드 중앙에 실시간 크고 아름다운 글씨 지표로 출력
