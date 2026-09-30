# Symbolic Regression with Genetic Programming

소재(material) descriptor로 **Heat of formation**을 설명하는 수식을 Genetic Programming(GP) 기반 symbolic regression으로 탐색하는 석사 연구 코드다. 비교용 MLP baseline을 함께 둔다.

## 구성

| 경로 | 내용 |
|---|---|
| `gp/` | GP 엔진. [gplearn](https://github.com/trevorstephens/gplearn)을 수정한 사본 (`SymbolicRegressor`, `SymbolicClassifier`, `SymbolicTransformer`) |
| `data_utils.py` | dataset 로드 → train/test split → 표준화. Scaler는 train에만 fit한다 |
| `GP_example.ipynb` | `high` dataset 예제: 1차 GP 모델 → 잔차(residual)에 2차 GP 모델 → 합성 성능 |
| `MLP_regression.py` | MLP baseline (`high`, `mid`, `low`) |
| `dataset/` | CSV 데이터 |
| `tests/` | pytest |

## gplearn 대비 주요 변경

- `pow2`(제곱), `pow3`(세제곱) 함수 추가. `SymbolicRegressor`의 기본 `function_set`을 14개로 확장
- Overflow·NaN이 난 프로그램의 fitness를 최악값으로 처리한다. NaN 프로그램이 `np.argmin`/`np.argmax`에서 선택되던 문제를 막고, `parsimony_coefficient='auto'`도 유한값만으로 계산한다
- NumPy 2.x / scikit-learn 1.6+ 호환: `np.int`·`numpy.lib.arraysetops` 사용 제거, mixin을 `BaseEstimator`보다 앞에 두도록 상속 순서 수정, classifier tag를 `__sklearn_tags__`로 제공
- `from gp import SymbolicRegressor`로 import할 수 있고(`gp/__init__.py`), pip로 설치할 수 있다(`pyproject.toml`)

## 설치

다른 프로젝트 환경에 설치해서 쓰는 경우 (numpy, scipy, scikit-learn, joblib 필요):

```bash
pip install -e /path/to/Symbolic_regression
```

이 repo 전용 환경을 만드는 경우 (검증된 버전 고정):

```bash
conda env create -f environment.yml
conda activate symreg
```

## 사용법

```python
import numpy as np
from gp import SymbolicRegressor

rng = np.random.RandomState(42)
X = rng.uniform(1, 5, (500, 3))
y = X[:, 0] * X[:, 1] / X[:, 2]

est = SymbolicRegressor(population_size=3000, generations=20,
                        function_set=('add', 'sub', 'mul', 'div'),
                        parsimony_coefficient=0.001,
                        feature_names=['P', 'V', 'T'],
                        random_state=0, n_jobs=4)
est.fit(X, y)
print(est)  # div(mul(P, V), T)
```

- 사용자 정의 함수는 `gp.make_function`, 사용자 정의 fitness는 `gp.make_fitness`로 만든다
- scikit-learn estimator이므로 `Pipeline`, `cross_val_score`, `clone`, pickle 저장을 그대로 쓸 수 있다
- `random_state`가 같으면 `n_jobs` 값과 관계없이 같은 결과가 나온다

## 데이터

`data_utils.load_split(name, target='Heat of formation')`은 `x_tr, x_ts, y_tr, y_ts, feature_names`를 돌려준다. `index_col`(기본 `Material`)과 `target`을 뺀 나머지 열이 모두 feature가 된다. 따라서 기본값에서는 `Energy above convex hull`도 feature로 쓰인다(feature 85개). 데이터 폴더는 다음 순서로 찾는다.

1. `data_dir` 인자
2. `$DATA_ROOT/symbolic_regression` (폴더가 있을 때)
3. repo의 `dataset/`

| 파일 | 행 | 열 | 구성 |
|---|---|---|---|
| `high.csv` | 1,366 | 87 | `Material` + descriptor 84개(`Z_Average`, `Z_Weighted_Average`, `Z_Max`, …) + `Heat of formation` + `Energy above convex hull` |
| `mid.csv` | 590 | 87 | `high.csv`와 같은 열 구성 |
| `low.csv` | 469 | 87 | `high.csv`와 같은 열 구성 |
| `crawl.csv` | 461 | 87 | `high.csv`와 같은 열 구성 |
| `Schleder2019_AtomicTable.csv` | 118 | 18 | 원소별 물성표 (`Element`, `Z`, `Electronegativity`, …) |

## 실행

repo 루트에서 실행한다.

```bash
python -m pytest
python MLP_regression.py
jupyter nbconvert --to notebook --execute --inplace GP_example.ipynb
```

## 변경 이력

- 2026-09: 아래 버그를 고쳤다. 이 때문에 노트북·MLP 수치가 2021년 결과와 다를 수 있다
  - test set에 scaler를 다시 fit하던 문제 (`fit_transform` → `transform`)
  - `MLP_regression.py`의 `r2_score` 인자 순서
  - 노트북 첫 번째 GP 실행에 `init_depth`가 전달되지 않던 문제

## 검증 환경

2026-09, WSL2 Ubuntu: Python 3.11.15, numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.0, pandas 3.0.3, joblib 1.5.3.

## License

- `gp/`는 gplearn(BSD 3-Clause, Copyright (c) 2015, Trevor Stephens)을 기반으로 한다. 배포할 때 `gp/LICENSE`를 함께 포함해야 한다
- 그 외 파일의 라이선스는 아직 지정하지 않았다
