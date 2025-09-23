# 생활 습관에 따른 Metabolic Syndrom 예측모델 제작 프로젝트

**기간**: 2025년 6월 23일 ~ 2025년 9월 30일  
**목표**: 사람들의 생활 습관 (흡연, 음주, 운동 등)에 따른 대사 증후군 발현 정도를 예측하는 모델 제작 및 인사이트 도출

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
- [모델학습 레포트](week4/day1/20250726day22.md)
- [모델학습 레포트 코드 실습 파일](week4/day1/20250726day22.ipynb)

---
# 📅 Week 4

## Week 4 전체 요약
| 구분       | 내용                                                                   |
| -------- | -------------------------------------------------------------------- |
| 🎯 목표    | 로지스틱 회귀로 분류 문제 접근 + 성능 평가 지표의 본질 이해                                  |
| 🧪 실습    | `LogisticRegression` 사용해서 이진 분류 문제 풀고 평가함                            |
| 📈 평가 지표 | 정확도(accuracy), 정밀도(precision), 재현율(recall), F1-score, ROC Curve, AUC |



## 사용한 주요 문법 & 함수
- `LogisticRegression`
- `confusion_matrix` 
- `precision, recall` 
- `F1-score` 
- `ROC Curve`, `AUC`
- `threshold`


## Week 4날짜 별 정리
- [week4 요약](week4/week4_summary.md) - 4주차 요약
- [로지스틱 회귀모델 최종 보고서](week5/day1/20250802day28.ipynb)

---
# 📅 Week 5

## Week 5 전체 요약
| 구분       | 내용                                                                |
| -------- | ----------------------------------------------------------------- |
| 🎯 목표    | 여러 모델을 비교하고, 각 모델이 얼마나 잘 예측하는지 성능 평가 + 해석 가능성 분석                  |
| 🧪 실습    | 로지스틱 회귀, 결정트리, 랜덤포레스트 등 다양한 모델 적용 + 성능 비교                         |
| 📈 평가 지표 | 정확도, 정밀도, 재현율, F1, AUC, ROC, confusion matrix, feature importance |
| 🧼 실전 팁  | 어떤 상황에서 어떤 모델을 쓰는 게 나은지, 해석이 가능한지, 데이터 크기/복잡도에 따라 달라지는 성능 고려      |

## 사용한 주요 문법 & 함수
- `DecisionTreeClassifier`
- `RandomForestClassifier`
- `XGBClassifier`

## Week 5 날짜 별 정리
- [week5 요약](week5/week5_summary.md)
- [분류모델 비교 레포트](week5/day7/20250817day35.md)

---
# 📅 Week 6(8/15 - 8/27)
- KHUDA 프로젝트 진행(경희대 최적 사업 아이템 제시)

## Week 6 전체 요약
- 데이터 수집, 전처리, 모델링 등 방학 기간동안 학습한 내용을 실제로 적용하는 프로젝트

## 사용한 주요 문법 & 함수
- `KMeans`
- `PCA`
- `RandomForestClassifier` 등

## Week6 정리
- `KHUDA` Repository의 `토이 프로젝트` 참고


## 📚 Dependencies
- Python 3.10+
- pandas, numpy
- scikit learn
- matplotlib, seaborn

- 학습 방법 : ChatGPT
