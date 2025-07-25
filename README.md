# Comorbidity Network Analysis Project

**기간**: 2025년 6월 23일 ~ 2025년 9월 30일  
**목표**: 환자-질병 진단 정보를 기반으로 코모르비디티 네트워크를 구축하고, 생물학적 지식 기반 해석까지 연결하는 포트폴리오 프로젝트.

# 📅 Week 1

## Week 1 전체 요약
Pandas 입문을 시작하고, 데이터 전처리를 익힌 주차.
기초 문법, 파생 변수 생성, 간단한 시각화, 결측치 처리 등을 통해 데이터 처리를 위한 기초 소양을 익힘.

## 발생한 문제점
1. 작업 환경 구축 문제
   - 각종 라이브러리(pandas, numpy 등)을 다운롤드 하는 과정에서 파이선 경로와 vscode 경로가 일치하지 않아서 설치가 안되는 문제 발생
  
2. `np.where()` 문제
   - 새로운 열을 생성하는 과정에서 조건을 입력할 때, `np.where()` 안에 다중 조건을 넣는 과정에서 어려움 발생 (Day 4 마크다운에 상세 내용 기록)

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

---
# 📅 Week 2

## Week 2 전체 요약
실전 데이터 분석을 위한 EDA 학습 및 시각화

##  사용한 주요 문법 & 함수
- `.str.split`, `.str.extract`, `.contains()` 등 문자열 처리
- `pd.to_datetime()`, `.dt.day_name()`, `.dt.weekday` 등 날짜 데이터 처리
- `MinMaxScaler`, `StandardScaler`로 정규화
- `groupby()` + `.mean()`, `.std()`, `.agg()`
- `sns.barplot`, `boxplot`, `histplot`, `kdeplot` 시각화
- `.corr()` + `sns.heatmap()` 상관분석

## Week 2날짜 별 정리
- [week2 요약](week2/week2_summary.md) - 2주차 요약
- [EDA report](week2/day7/20250707day14.md)

---
# 📅 Week 3

## Week 3 전체 요약
주제: 회귀모델을 통한 변수의 영향력 분석, 모델 성능 비교 및 평가

목표: 실제 데이터 기반으로 다중 선형회귀 분석 수행 → 모델 단순화 및 해석 가능성 확보

## 사용한 주요 문법 & 함수
```python
# 회귀 모델
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# 지표 평가
from sklearn.metrics import mean_squared_error, r2_score

# 데이터 전처리
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 시각화
import matplotlib.pyplot as plt
import seaborn as sns
```

## Week 3날짜 별 정리
- [week3 요약](week3/week3_summary.md) - 3주차 요약
- [모델학습 레포트](week3/day7/20250725day21.ipynb)


## 📚 Dependencies
- Python 3.10+
- pandas, numpy
- networkx
- matplotlib, seaborn
