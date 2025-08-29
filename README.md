# MLOps HW1 — Wine Quality (Classification)
---

## 3.3 Документация (по ТЗ)

### 1) Датасет и задача

* **Источник:** Kaggle → Wine Quality (*WineQT.csv*).
* **Признаки:** 11 числовых колонок (кислотность, сахар, pH, спирт и т.п.).
* **Цель:** `quality`.
* **Формулировка:** **бинарная классификация** — `label = 1`, если `quality >= 6`, иначе `0`.

### 2) Алгоритмы (минимум два — выполнено)

* `LogisticRegression` + `StandardScaler` (Pipeline).
* `RandomForestClassifier` — **две** конфигурации: `rf_base`, `rf_tuned`.

### 3) Что логируем в MLflow (выполнено)

* **Параметры модели** (гиперпараметры + размер датасета + список фич).
* **Метрики качества**: `accuracy`, `f1_macro`, `train_time_sec`.
* **Артефакты**: сохранённая модель (`model/` в MLflow) + локально `model_weights/*.pkl`, лучшая — `best_model.pkl`.
* **Доп. инфо**: теги `algo`, `run_ts`; имя запуска с датой (например, `rf_tuned_20250829_221234`).

---

## Как запустить (очень коротко)

### Вариант A — через DVC (рекомендуется)

```bash
pip install -U pandas scikit-learn mlflow dvc
# если данные под DVC уже в репо:
dvc pull             # подтянуть WineQT.csv и/или processed сплиты

dvc repro            # запустить конвейер: data_load → train
mlflow ui --backend-store-uri mlruns --port 5000  # UI: http://127.0.0.1:5000
```

### Вариант B — по шагам (без DVC)

```bash
python src/data_loader.py --in data/WineQT.csv --out-dir data/processed --task binary --threshold 6 --seed 42
python src/train_model.py --train data/processed/train.csv --test data/processed/test.csv --seed 42
mlflow ui --backend-store-uri mlruns --port 5000
```

В MLflow увидите 3 отдельных запуска: `logreg_*`, `rf_base_*`, `rf_tuned_*`.

---

## 3.2.3 DVC конфигурация (по ТЗ)

1. **Инициализация** (один раз):

```bash
dvc init && git add . && git commit -m "init dvc"
```

2. **Трек данных** (один раз, если ещё не сделано):

```bash
dvc add WineQT.csv
git add WineQT.csv.dvc .gitignore && git commit -m "track raw data"
```

3. **dvc.yaml** (две стадии — *уже в проекте*):

```yaml
stages:
  data_load:
    cmd: python src/data_loader.py --in data/WineQT.csv --out-dir data/processed --task binary --threshold 6 --seed 42
    deps:
      - data_loader.py
      - WineQT.csv
    outs:
      - train.csv
      - test.csv

  train:
    cmd: python src/train_model.py --train data/processed/train.csv --test data/processed/test.csv --seed 42
    deps:
      - train_model.py
      - train.csv
      - test.csv
    outs:
      - model_weights/best_model.pkl
```

4. **Remote (локальный достаточно):**

```bash
dvc remote add -d local_store .dvc/storage
git add .dvc/config && git commit -m "add local remote"
dvc push   # отправит данные/модели в локальный remote
```

---

## Краткие результаты и выводы

| Модель                                                                                                  |        Accuracy |        F1-macro |
| ------------------------------------------------------------------------------------------------------- | --------------: | --------------: |
| logreg                                                                                                  |     \~0.77–0.79 |     \~0.76–0.78 |
| rf\_base                                                                                                |     \~0.80–0.82 |     \~0.79–0.81 |
| **rf\_tuned**                                                                                           | **\~0.81–0.83** | **\~0.80–0.82** |
| **Вывод:** ансамбль деревьев даёт лучшую точность на этом датасете; масштабирование важно для `LogReg`. |                 |                 |

---

## Структура (минимум для запуска)

```
src/
  data_loader.py       # 3.2.1
  train_model.py       # 3.2.2
data/
  WineQT.csv           # под DVC (.dvc в репо)
  train.csv  # генерируется
  test.csv   # генерируется
model_weights/
  best_model.pkl       # аутпут стадии train
```

---

## Частые проблемы (коротко)

* **MLflow experiment deleted** → скрипт сам восстановит (функция `ensure_experiment`).
* **Нет метрик в таблице** → уберите фильтры в UI и смотрите top‑level руны (`logreg_*`, `rf_*`).
* **Не совпали колонки train/test** → перезапустите стадию `data_load`: `dvc repro data_load`.

---

## Команды для проверки (копипаст)

```bash
# полная репродукция
pip install -r requirements.txt
 dvc pull && dvc repro
 mlflow ui --backend-store-uri mlruns --port 5000
```

---

