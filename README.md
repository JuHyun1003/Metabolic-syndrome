# Comorbidity Network Analysis Project

**기간**: 2025년 6월 23일 ~ 2025년 9월 30일  
**목표**: 환자-질병 진단 정보를 기반으로 코모르비디티 네트워크를 구축하고, 생물학적 지식 기반 해석까지 연결하는 포트폴리오 프로젝트.

# 📅 Week 1
---
## Week 1 전체 요약
Pandas 입문을 시작하고, 데이터 전처리를 익힌 주차.
기초 문법, 파생 변수 생성, 간단한 시각화, 결측치 처리 등을 통해 데이터 처리를 위한 기초 소양을 익힘.

##  사용한 주요 문법 & 함수
- DataFrame 생성 / 인덱싱: `df[]`, `loc`, `iloc`
- 조건 처리: `np.where()`, `apply()`, `map()`
- 그룹 통계: `groupby()`, `mean()`, `value_counts()`, `unstack()`
- 파생 변수 생성: `apply(axis=1)`, 사용자 함수 적용
- 구간 분류: `pd.cut()`
- 결측치 처리: `isnull()`, `fillna()`, `dropna()`, `interpolate()`
- 시각화: `plot(kind='bar'/'hist')`, `matplotlib`, `한글 폰트 설정`
- 요약정보: `describe()`, `info()`
  
## Week 1 날짜 별 정리
- [week1 요약](week1/week1_summary.md) - 1주차 요약


## 📚 Dependencies
- Python 3.10+
- pandas, numpy
- networkx
- matplotlib, seaborn
