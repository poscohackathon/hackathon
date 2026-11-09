# 열간압연 압하력 데이터 (Hot Rolling Force Database)

열간압연 사상압연기에서 측정한 압하력(롤 분리력) 데이터입니다. 측정값은 38,980건입니다.

## 데이터 정보

| 항목 | 내용 |
|---|---|
| 출처 | Nucor Steel Decatur 사상압연 공장(Finishing Mill) |
| 기간 | 2023~2024년 |
| 강종 | 10종 (Grade01~Grade10으로 이름을 바꿔 공개) |
| 단위 | 모두 미터법(SI) |

## 파일

| 파일 | 설명 |
|---|---|
| `metals.xlsx` | 원본 엑셀 |
| `metals.csv` | 엑셀을 CSV로 변환한 파일 |
| `metals-15-01346-s001.zip` | 논문 부록 원본 압축 파일 (`metals-3957876-supplementary.xlsx` 포함) |

## 컬럼 설명

CSV에 들어 있는 순서대로 적었습니다.

| 컬럼 | 설명 | 단위 |
|---|---|---|
| `Grade` | 강종 구분 (Grade01~Grade10) | - |
| `Coil ID` | 코일 번호 (1~7797) | - |
| `Caster ID` | 연속주조기 구분 (S 또는 N) | - |
| `Stand ID` | 사상압연 스탠드 (F1~F5) | - |
| `Entry gauge (mm)` | 스탠드 입측 두께 | mm |
| `Exit gauge (mm)` | 스탠드 출측 두께 | mm |
| `Radius (mm)` | 작업롤 반경 | mm |
| `Width (mm)` | 판 폭 | mm |
| `Temperature (K)` | 판 온도 | K (켈빈) |
| `Force (N)` | 압하력(롤 분리력) | N (뉴턴) |

## 참고 사항

- 데이터 한 행은 코일 하나가 스탠드 하나를 지날 때의 측정값입니다. 그래서 행 수는 코일 수의 5배입니다(스탠드 F1~F5).
- 예외로 Grade02의 코일 2개는 값이 0이 아닌 스탠드가 각각 2개, 3개뿐이었습니다.
- 0 이하의 값과 결측값은 정리해서 뺐습니다. 이렇게 빠진 측정값은 위 Grade02 코일 2개의 5건입니다.
- `metals.csv`에는 `Force (N)` 뒤에 내용이 없는 빈 컬럼 18개가 붙어 있습니다. 읽을 때 앞의 10개 컬럼만 쓰면 됩니다.

```python
import pandas as pd
df = pd.read_csv("main/raw_data/metals.csv", usecols=range(10))
```
