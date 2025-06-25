icd_list = ['E11', 'I10', 'J18', 'E11', 'I10', 'I10', 'A41']
import pandas as pd
data=[10,20,30,40,50]
series=pd.Series(data)
print(series)

data = {
    '이름': ['길주현', '김성훈', '박기웅'],
    '나이': [23, 27, 31],
    '키(cm)': [177, 180, 168]
}
df=pd.DataFrame(data)
print(df)
