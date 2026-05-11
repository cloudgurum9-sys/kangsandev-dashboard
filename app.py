import streamlit as st
import pandas as pd
import numpy as np

# ==========================================
# 페이지 기본 설정
# ==========================================
st.set_page_config(
    page_title="강산개발 통합 재무 대시보드",
    page_icon="🏢",
    layout="wide"
)

# ==========================================
# [모듈 1] 증빙 대사 자동화 핵심 로직 & 데이터
# ==========================================
def process_reconciliation(df_site, df_tax):
    merged_df = pd.merge(df_site, df_tax, on=['현장명', '거래처명'], how='outer', suffixes=('_현장', '_홈택스'), indicator=True)

    def check_status(row):
        if row['_merge'] == 'both':
            if row['청구금액'] == row['세금계산서금액']: return '정상 (금액 일치)'
            else: return '금액 불일치'
        elif row['_merge'] == 'left_only': return '증빙 누락 (계산서 미발행)'
        elif row['_merge'] == 'right_only': return '현장 지출보고 누락'

    merged_df['검증결과'] = merged_df.apply(check_status, axis=1)
    
    # [수정] 결측치를 0으로 채운 후, 소수점(.0) 제거를 위해 정수(int) 타입으로 변환
    merged_df['청구금액'] = merged_df['청구금액'].fillna(0).astype(int)
    merged_df['세금계산서금액'] = merged_df['세금계산서금액'].fillna(0).astype(int)
    merged_df['차액'] = (merged_df['청구금액'] - merged_df['세금계산서금액']).astype(int)
    
    return merged_df.drop(columns=['_merge'])

def load_reconcile_sample():
    site_data = pd.DataFrame({
        '일자': ['10-01', '10-05', '10-12', '10-15', '10-20'],
        '현장명': ['A아파트현장', 'A아파트현장', 'B토목현장', 'B토목현장', 'C도로공사현장'],
        '거래처명': ['(주)철근나라', '안전제일용역', '(주)레미콘스타', '동네식당(함바)', '건설장비렌탈'],
        '청구금액': [11000000, 3300000, 5500000, 1500000, 4400000]
    })
    tax_data = pd.DataFrame({
        '일자': ['10-01', '10-05', '10-12', '10-18', '10-25'],
        '현장명': ['A아파트현장', 'A아파트현장', 'B토목현장', 'C도로공사현장', '본사'],
        '거래처명': ['(주)철근나라', '안전제일용역', '(주)레미콘스타', '건설장비렌탈', '오피스문구'],
        '세금계산서금액': [11000000, 3000000, 5500000, 4400000, 200000] 
    })
    return site_data, tax_data

# ==========================================
# [모듈 2] 법인카드 리스크 탐지 로직 & 데이터
# ==========================================
def detect_card_anomalies(df):
    df = df.copy()
    df['승인일자'] = pd.to_datetime(df['승인일자'])
    df['승인시간_dt'] = pd.to_datetime(df['승인시간'], format='%H:%M').dt.time
    df['요일'] = df['승인일자'].dt.dayofweek
    df['이상치_사유'] = [[] for _ in range(len(df))]
    
    # 룰 검증
    mask_weekend = df['요일'].isin([5, 6])
    df.loc[mask_weekend, '이상치_사유'] = df.loc[mask_weekend, '이상치_사유'].apply(lambda x: x + ['주말결제'])
    
    mask_night = df['승인시간_dt'].apply(lambda t: t >= pd.to_datetime('23:00', format='%H:%M').time() or t <= pd.to_datetime('06:00', format='%H:%M').time())
    df.loc[mask_night, '이상치_사유'] = df.loc[mask_night, '이상치_사유'].apply(lambda x: x + ['심야결제'])

    mask_category = df['업종'].isin(['유흥주점', '단란주점', '노래방', '스크린골프', '피부미용'])
    df.loc[mask_category, '이상치_사유'] = df.loc[mask_category, '이상치_사유'].apply(lambda x: x + ['제한업종'])

    split_counts = df.groupby(['승인일자', '현장명', '가맹점명']).size().reset_index(name='결제횟수')
    split_suspects = split_counts[split_counts['결제횟수'] > 1]
    for _, row in split_suspects.iterrows():
        mask_split = (df['승인일자'] == row['승인일자']) & (df['현장명'] == row['현장명']) & (df['가맹점명'] == row['가맹점명'])
        df.loc[mask_split, '이상치_사유'] = df.loc[mask_split, '이상치_사유'].apply(lambda x: x + ['쪼개기결제의심'])

    df['이상치_사유'] = df['이상치_사유'].apply(lambda x: ', '.join(x) if len(x) > 0 else '정상')
    df['리스크여부'] = df['이상치_사유'] != '정상'
    
    # [수정] 불필요한 시간 단위 삭제 (YYYY-MM-DD 포맷 유지)
    df['승인일자'] = df['승인일자'].dt.strftime('%Y-%m-%d')
    # [수정] 금액 소수점 제거
    df['결제금액'] = df['결제금액'].fillna(0).astype(int)
    
    return df.drop(columns=['승인시간_dt', '요일'])

def load_card_sample():
    return pd.DataFrame({
        '승인일자': ['2023-10-11', '2023-10-13', '2023-10-14', '2023-10-18', '2023-10-18', '2023-10-20'],
        '승인시간': ['12:30', '23:45', '14:00', '12:00', '12:05', '02:30'],
        '현장명': ['A아파트현장', 'A아파트현장', 'B토목현장', '본사_영업팀', '본사_영업팀', 'B토목현장'],
        '가맹점명': ['함바식당', '별밤단란주점', '이마트', '고급한우전문점', '고급한우전문점', '24시해장국'],
        '업종': ['일반음식점', '단란주점', '마트/편의점', '일반음식점', '일반음식점', '일반음식점'],
        '결제금액': [150000, 850000, 45000, 400000, 350000, 45000]
    })

# ==========================================
# 사이드바 네비게이션
# ==========================================
st.sidebar.title("🏢 강산개발 재무 시스템")
menu = st.sidebar.radio(
    "메뉴 이동",
    ["🏠 대시보드 홈", "📊 1. 공사 원가/증빙 대사", "💳 2. 법인카드 리스크 모니터링"]
)

# ==========================================
# 화면 렌더링
# ==========================================
if menu == "🏠 대시보드 홈":
    st.title("강산개발 맞춤형 재무통제 대시보드")
    st.markdown("""
    좌측 메뉴를 클릭하여 건설업 특성에 맞춘 재무 자동화 로직을 테스트해 보세요.
    
    *   **프로젝트 1:** 수많은 현장 지출 내역과 국세청 세금계산서를 자동 대사하여 부가세 공제 누락 방지
    *   **프로젝트 2:** 본사와 떨어진 현장의 법인카드 남용(유흥, 심야, 쪼개기 결제)을 알고리즘으로 자동 탐지
    """)
    st.info("👈 왼쪽 사이드바에서 메뉴를 선택해주세요.")

elif menu == "📊 1. 공사 원가/증빙 대사":
    st.title("📊 건설 현장 공사비 증빙 자동 대사")
    use_sample = st.sidebar.checkbox("샘플 데이터로 데모 실행", value=True, key='sample1')
    
    if use_sample:
        df_site, df_tax = load_reconcile_sample()
    else:
        file1 = st.sidebar.file_uploader("현장 청구(Excel)", type=['xlsx'])
        file2 = st.sidebar.file_uploader("홈택스 내역(Excel)", type=['xlsx'])
        df_site = pd.read_excel(file1) if file1 else None
        df_tax = pd.read_excel(file2) if file2 else None

    if df_site is not None and df_tax is not None:
        result_df = process_reconciliation(df_site, df_tax)
        st.subheader("대사 요약 결과")
        col1, col2, col3 = st.columns(3)
        col1.metric("총 거래 건수", f"{len(result_df)}건")
        col2.metric("정상 일치", f"{len(result_df[result_df['검증결과']=='정상 (금액 일치)'])}건")
        col3.metric("⚠️ 확인 요망", f"{len(result_df[result_df['검증결과']!='정상 (금액 일치)'])}건")
        
        # [수정] 화면 출력용(display_df)을 따로 만들어 천 단위 콤마 추가
        display_df = result_df.copy()
        for col in ['청구금액', '세금계산서금액', '차액']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:,}")
            
        # 오류 건만 빨간색 글씨로 하이라이트
        def color_error(val):
            return 'color: red; font-weight: bold' if val != '정상 (금액 일치)' else ''
            
        st.dataframe(display_df.style.map(color_error, subset=['검증결과']), use_container_width=True)

elif menu == "💳 2. 법인카드 리스크 모니터링":
    st.title("💳 현장 법인카드 리스크 모니터링")
    use_sample = st.sidebar.checkbox("샘플 데이터로 데모 실행", value=True, key='sample2')
    
    if use_sample:
        df_card = load_card_sample()
    else:
        file3 = st.sidebar.file_uploader("카드 승인내역(Excel)", type=['xlsx'])
        df_card = pd.read_excel(file3) if file3 else None

    if df_card is not None:
        result_df = detect_card_anomalies(df_card)
        st.subheader("리스크 탐지 요약")
        
        # 합계 계산은 콤마가 없는 원본 데이터(result_df)로 수행
        risk_df = result_df[result_df['리스크여부'] == True]
        
        col1, col2 = st.columns(2)
        col1.metric("총 결제 금액", f"{result_df['결제금액'].sum():,.0f}원")
        col2.metric("🚨 이상치 의심 금액", f"{risk_df['결제금액'].sum():,.0f}원", delta_color="inverse")
        
        st.write("**(주말/심야 결제, 제한업종, 쪼개기 결제 의심 건)**")
        
        # [수정] 화면 출력용(display_df2) 복사 및 천 단위 콤마 추가
        display_df2 = risk_df.copy()
        display_df2['결제금액'] = display_df2['결제금액'].apply(lambda x: f"{x:,}")
        
        # 이상치 발생 행에 배경색(옅은 빨강) 부여
        def highlight_row(row):
            return ['background-color: #ffe6e6']*len(row)
            
        st.dataframe(display_df2.drop(columns=['리스크여부']).style.apply(highlight_row, axis=1), use_container_width=True)
