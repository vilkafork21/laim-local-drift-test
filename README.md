# laim-local-drift-test

Тестовая нода мониторингового контура LAIM. Принимает **эталонную корзину**
(`reference_umr`), **мониторинговые запросы** (`monitoring_umr`) и
**валидированный контракт метрики** (`monitoring_metric`) и отдаёт в агрегатор
светофор теста на локальный дрифт запросов: оценку ключевой метрики на
мониторинге, посчитанную по семантически ближайшим эталонным запросам, и
надёжность этой оценки.

## Зачем нода нужна

На мониторинге нет разметки: неизвестно, хорошо ли агент ответил на запрос из
трафика. Нода закрывает пробел без асессора: для каждого мониторингового
запроса находит ближайшие по смыслу запросы корзины с известной оценкой и
переносит их оценку с весом, равным близости. Если близких эталонов нет
(запросы «уехали» от корзины), нода отвечает «серый», а не выдаёт число.

Ключевые решения: LLM используется **только как эмбеддер** (GigaChat
Embeddings), арифметика детерминирована; форма данных (QA, реплика с историей,
диалог) нормализуется контрактом `laim_monitoring/core.py`; нарушение
контракта — падение, недостаток данных — серый светофор.

## Место в контуре

```text
laim-baskets-adapter.reference_umr ─────────────────┐
laim-traces-dataset-converter.monitoring_umr ───────┼─► laim-local-drift-test
laim-kriteria-selector.validated_monitoring_metric ─┘         │
                                                              ├─► all_results      ─► laim-agg.in
                                                              └─► test_description ─► HTML в UI ноды
                                                                  (в port_wiring.json не подключён)
```

## Порты и настройки

### Входы (все обязательные)

| Порт | Тип | Что приходит с платформы |
|---|---|---|
| `reference_umr` | dataframe | Корзина в формате тестового датасета (`laim-umr.v2`): flat-лист (`query_id`, `input_query`, `output_answer`) либо packed-диалог (`session_id`, `dialogue`); обязательна колонка `main_metric` |
| `monitoring_umr` | dataframe | Выход TDC той же формы без `main_metric`; принимается DataFrame, parquet-bytes или путь к parquet |
| `monitoring_metric` | default | Контракт `laim-monitoring-metric.v2` (`v1` поднимается автоматически); обязателен `assessment_mode` |

### Выходы

| Порт | Тип | Что отдаёт |
|---|---|---|
| `all_results` | default | Словарь результата теста (см. «Форматы выхода») — вход `laim-agg.in` |
| `test_description` | hidden | HTML-отчёт теста; рендерится в результатах ноды через `$.test_description` |

### Настройки ноды

| Настройка | По умолчанию | Зачем |
|---|---|---|
| `ann_config` | `{'create_index': {'exact': True}, 'search_query': {}}` | Строка-литерал Python; `create_index` принимает `exact`, `n_partitions` (10), `nprobe` (2). `exact: False` включает IVF, но при OOS меньше `40 * n_partitions` строк индекс всё равно точный |
| `n_closest` | `5` | Верхняя граница числа соседей; фактическое `N = min(n_closest, max(3, min(10, n_oos // 20)))`, при `n_closest <= 0` берётся сама граница |
| `metric_agg` | `single_mean` | Способ агрегации `target`; единственное поддерживаемое значение — в UI других нет |
| `red_threshold` | `0.25` | Абсолютное снижение метрики, с которого светофор красный |
| `green_threshold` | `0.15` | Абсолютное снижение метрики, до которого светофор зелёный. Код сортирует пару порогов: меньший — граница зелёного, больший — красного |
| `reliability_threshold` | `0.2` | Порог средней близости соседей; ниже — тест неинформативен |
| `greater_is_better` | `true` | Направление метрики: при `false` красным считается рост оценки |
| `is_info` | `false` | Принудительно серый светофор (информативный тест) |

## Как проходит прогон

```text
1. Контракт      validate_monitoring_metric; нормализация обоих UMR в drift-фреймы
2. Семплер       корзина -> OOS (train), мониторинг -> OOT (test); target -> float
3. Гейт размера  OOS < 30 строк -> серый без обращения к GigaChat
4. Метрика OOS   среднее main_metric корзины и его светофор по порогам (0.4, 0.6)
5. Эмбеддинги    GigaChat Embeddings для вопросов OOS и OOT, FAISS-индекс по OOS
6. Оценка        для каждого OOT-запроса top-N соседей, score и reliability
7. Отчёт         цвет = худший из светофора OOS и светофора по снижению; HTML
```

**1. Контракт.** `prepare_drift_frames` валидирует `monitoring_metric`
(`require_computed=False`, но без `assessment_mode` — отказ) и приводит оба
UMR к единицам наблюдения по `assessment_mode`: `qa` — строка,
`turn_with_history` — реплика с историей, `dialogue` — сессия. В drift-фрейм
уходят `question` (для контекстных режимов — JSON-список `input_query`
реплик), пустой `answer` и `target` = `main_metric`; в корзине `target`
обязателен, строки с пустым `target` при `missing_policy != fail` отбрасываются.

**2–3. Семплер и гейт.** `AutoAsessorSampler` кладёт корзину в `train`,
мониторинг в `test`. Меньше 30 строк OOS (`MIN_OOS_SAMPLES`) — серый до
обращения к эмбеддеру.

**4. Метрика OOS.** `valtest_metric` считает среднее `target` по корзине и
ставит ему светофор `tricky_semaphore` с зашитыми порогами `(0.4, 0.6)`:
ниже 0.4 — красный, от 0.4 до 0.6 — жёлтый, от 0.6 — зелёный. Пороги в
настройки не вынесены и участвуют в итоговом цвете (шаг 7).

**5. Эмбеддинги и индекс.** Пустые вопросы — заглушка `<empty>`, `NaN` в
векторах — нули. Векторы нормализуются L2, индекс `IndexFlatIP` (косинусная
близость); IVF только при `exact: False` и достаточном OOS.

**6. Оценка.** Метки корзины переводятся в знаковые: бинарные `{0, 1}` — в
`{-1, +1}`, иные — центрируются `2 * (y - min) / (max - min) - 1`. Для
каждого OOT-запроса `score = 0.5 + sum(sim_i * target_i) / (2 * N)`,
обрезанный в `[0, 1]`; `reliability` запроса — среднее `sim_i`. Оценка
метрики — среднее `score` по OOT; надёжность — `mean`, `median`, `q05` и
`share_below_threshold` (доля запросов с `reliability < reliability_threshold`).

**7. Отчёт.** Снижение `drop = metric_value - estimate` (при
`greater_is_better = false` — с обратным знаком): `drop < 0.15` — зелёный,
до 0.25 — жёлтый, от 0.25 — красный; итог — `worst_semaphore` этого цвета
и цвета OOS-метрики (шаг 4). Серый перекрывает всё: `is_info`,
`reliability.mean < reliability_threshold`, `share_below_threshold > 0.3`, `NaN`-оценка.

### Пример лога прогона

Формат строк — из кода; значения условные. Все модули пишут в корневой
логгер (`logging.info`), поэтому имя логгера — `root`.

```text
INFO root: Размер OOS (real): 266
INFO root: Размер OOT (agent): 412
INFO root: Тест на локальный дрифт запущен
INFO root: Используется n_closest=5 для OOS размера 266
INFO root: Расчёт основной метрики на OOS
INFO root: Подготовка ANN и эмбеддингов
INFO root: Начало формирования отчёта
INFO root: {'report': {'semaphore': 'green', 'result_plots': [], 'result_dataframes': [...]}, 'precomputed': {'metric_value': 0.9736, 'metric_value_estimate': 0.9012, 'reliability': {'mean': 0.7143, 'median': 0.7311, 'q05': 0.4187, 'share_below_threshold': 0.0}}}
```

Предупреждения, которые нода пишет при деградации:

```text
WARNING root: OOS слишком мал (24 < 30); тест неинформативен
WARNING root: Некоторые OOT-вопросы пусты; заменяются заглушкой
WARNING root: GigaEmbed: 3 из 412 текстов длиннее 1000 символов — разбиты на части
WARNING root: GigaEmbed batch failed (attempt 1/3): <текст ошибки>; повтор через 1.0s
```

## Форматы выхода и контракты

Порт `all_results` — словарь с фиксированным набором ключей:

| Ключ | Значение |
|---|---|
| `test_name` | всегда `local_drift` (имя, по которому `laim-agg` находит тест) |
| `color` | `red` / `yellow` / `green` / `gray` |
| `status` | `not_computable` при `gray`, иначе `computed` |
| `calculated_traffic_lights` | `{"test_light": <цвет>, "semaphore_title": <текст вердикта>}` |
| `reason` | всегда `null`: код ключ публикует, но не заполняет |
| `metric_value` | среднее `main_metric` по корзине (OOS), не `baseline.value` контракта |
| `metric_value_estimate` | оценка метрики на мониторинге (OOT) |
| `drop_estimate` | `abs(metric_value - metric_value_estimate)`, знак не сохраняется |
| `reliability_mean` | средняя близость соседей по OOT |
| `share_uncovered` | доля OOT-запросов с близостью ниже `reliability_threshold` |

Словарь цветов ноды — `red`, `yellow`, `green`, `gray`: нода отдаёт именно
`yellow`, а не `amber`; `laim-agg` нормализует `yellow` в `amber` на входе.
При сером прогоне по малому OOS `metric_value` и `metric_value_estimate` —
`NaN`, `drop_estimate` — `null`, `share_uncovered` — `1.0`. Заголовок
светофора в UI ноды читает `$.all_results.issue` — такого ключа нода не
публикует; текст вердикта лежит в `calculated_traffic_lights.semaphore_title`.

Порт `test_description` — HTML: цель, условия, формула, таблица критериев
светофора и таблица результатов. Критерии в HTML сформулированы через
«светофор OOT»; в коде это светофор среднего `main_metric` корзины (OOS).

## Падение против деградации

Нода не публикует `reason_code`: падение — исключение, которое платформа
показывает как ошибку ноды.

| Причина | Исключение |
|---|---|
| Контракт не `v2`/`v1`, нет `assessment_mode`, `umr_version` не `laim-umr.v2`, `score_column` не `main_metric` | `MonitoringContractError` |
| UMR пуст, не DataFrame, смешан flat и `dialogue`, пустой `query_id`, контекстный режим без `session_id`, turn диалога не тройка | `MonitoringContractError` |
| В корзине нет `main_metric`; пустой `main_metric` при `missing_policy = fail`; `main_metric` не константен внутри диалога | `MonitoringContractError` |
| `monitoring_umr` — нечитаемый parquet или несуществующий путь | `MonitoringContractError` |
| GigaChat недоступен после 3 попыток батча; число векторов не равно числу частей | `RuntimeError` |
| `ann_config` — не литерал Python; эмбеддинги не двумерны | `ValueError` / `SyntaxError` |

Деградация — всегда в серый (`status = not_computable`), без падения:

| Событие | Реакция |
|---|---|
| OOS меньше 30 строк | `gray`, WARNING; GigaChat не вызывается; числа `NaN` |
| `reliability.mean < reliability_threshold` | `gray`; числа публикуются |
| Доля запросов с низкой близостью выше 0.3 | `gray`; числа публикуются |
| `is_info = true` | `gray` |
| Пустой `main_metric` в корзине при `missing_policy` `exclude_unit`/`exclude_value`/`zero` | строка отброшена из OOS без записи в лог |
| Пустой вопрос в OOS/OOT; `NaN` в векторе эмбеддинга | заглушка `<empty>` с WARNING; нули |
| Текст длиннее 1000 символов; ответ 413 от эмбеддера | части по 1000 символов (при 413 — пополам до 200), векторы усредняются, WARNING |
| `exact: False` при OOS меньше `40 * n_partitions` | точный индекс, INFO |

## Внешние сервисы

Единственный сервис — эмбеддер GigaChat через SDK `gigachat`
(`GigaChat.embeddings`), обёрнутый в `GigaEmbed`. Конфигурация — `Config`
(`config.py`), переменные окружения читаются через `python-dotenv`:

| Переменная | Роль |
|---|---|
| `AI_GATEWAY_URL` | Если задана — контур `sds`: `base_url = AI_GATEWAY_URL + "/api/v1"`, учётные данные SDK не передаются |
| `BASE_URL`, `CREDENTIALS`, `AUTH_URL`, `SCOPE`, `VERIFY_SSL_CERTS` | Иначе контур `sigma`: полный набор параметров SDK; `VERIFY_SSL_CERTS` истинен только при строке `True` |

Адреса по умолчанию в коде нет: без переменных окружения SDK получает
`base_url = None`. Батч — 100 текстов, часть — 1000 символов (нижняя граница
дробления — 200), части одного текста усредняются в один вектор. Сетевые
ошибки батча — 3 попытки с паузами 1 и 2 секунды, затем `RuntimeError`
(серого в этом случае нет). При неизменных ответах эмбеддера прогон
детерминирован: точный поиск FAISS и арифметика без случайности; IVF
(`exact: False`) обучает k-means и детерминированность не гарантирует.

## Наблюдаемость

Отдельного порта журнала нет. В лог платформы уходят строки из «Примера
лога» и полный словарь результата (`logging.info(res)`) вместе с
`result_dataframes`. Машинная запись прогона — сам `all_results`: триаж на
сотне прогонов делается по `status`, `color`, `drop_estimate`,
`reliability_mean`, `share_uncovered`. Серый с `share_uncovered = 1.0` и
`NaN` — малый OOS; серый с числами — низкая близость соседей.

## Карта кода

```text
main.py                              порты платформы, настройки, all_results и HTML
config.py                            Config: контур sigma/sds из переменных окружения
giga_wraper.py                       GigaEmbed: батчи, чанкинг, ретраи, mean pooling
html_report_helper.py                светофоры и таблица критериев для HTML (IPython)
llm_val/valtest_local_drift_stability.py  тест: гейт OOS, адаптивный N, score, reliability, цвет
llm_val/valtest_metric.py            метрика OOS и её светофор (пороги 0.4/0.6)
llm_val/report_helper.py             semaphore_by_threshold, worst_semaphore, tricky_semaphore
llm_val/ann.py                       ANN: FAISS IndexFlatIP / IndexIVFFlat, fallback на exact
llm_val/sampler.py                   AutoAsessorSampler: OOS = train, OOT = test
llm_val/scorer.py                    AutoAsessorScorer: среднее target
llm_val/utils.py                     METRICS (single_mean, multicol_mean), string_to_float
laim_monitoring/core.py              контракт v2, normalize_umr, unitize, prepare_drift_frames
tests/                               контракт all_assessors, чанкинг GigaEmbed, формы UMR, smoke main
```

## Что делать, если

- **Серый, `share_uncovered = 1.0`, значения `NaN`** — корзина меньше 30
  единиц наблюдения (в режиме `dialogue` единица — сессия, а не реплика).
  Проверьте `assessment_mode` контракта и размер корзины.
- **Серый с числами** — мониторинговые запросы далеки от корзины
  (`reliability_mean` ниже 0.2 или больше 30% запросов ниже порога). Это
  результат теста, а не дефект; снижать `reliability_threshold` — решение
  методолога.
- **Жёлтый или красный при малом `drop_estimate`** — сработал светофор
  OOS-метрики: среднее `main_metric` корзины ниже 0.6 (жёлтый) или 0.4
  (красный). Проверьте `metric_value` в `all_results`.
- **`RuntimeError: GigaEmbed: исчерпаны попытки (3)`** — эмбеддер недоступен;
  проверьте `AI_GATEWAY_URL` (sds) или `BASE_URL`/`CREDENTIALS` (sigma).
  Нода в этом случае падает, а не деградирует.

## Деплой

База — `py312-simple`; синтаксис и stdlib новее Python 3.12 не используются.
Точка входа — функция `main` в `main.py`. `descriptor.json` перечисляет в
`script.runConfiguration.sourceFiles` 13 файлов: `main.py`,
`html_report_helper.py`, `config.py`, `giga_wraper.py`, семь модулей
`llm_val/` (`ann`, `report_helper`, `sampler`, `scorer`, `utils`,
`valtest_local_drift_stability`, `valtest_metric`) и два
`laim_monitoring/` (`__init__`, `core`). Теста соответствия `sourceFiles`
диску в `tests/` нет; `test_descriptor_defaults_match_runtime_contract`
закрепляет пороги 0.25 / 0.15 / 0.2 и описание порта `monitoring_umr`.

Зависимости (`requirements.txt`, объявлен в `libraryDependencies`):
`pandas`, `numpy`, `faiss-cpu`, `gigachat`, `python-dotenv`, `ipython`
(`IPython.display` в `html_report_helper.py`), `jinja2` (`Styler.to_html`),
а также `scikit-learn` и `scipy`, которые кодом ноды не импортируются.
ZIP ноды — `descriptor.json`, `requirements.txt` и файлы `sourceFiles` из
ветки `dev`; готовая версия переносится в
Проверка перед сборкой: `python -m pytest -q` и `ruff check .` (CI, Python 3.12).

## Глоссарий

- **OOS** — выборка корзины (эталонные запросы с известной оценкой); в
  семплере — `train`.
- **OOT** — мониторинговые запросы без оценки; в семплере — `test`.
- **Локальный дрифт** — смещение запросов трафика относительно корзины,
  измеряемое косинусной близостью эмбеддингов к N ближайшим эталонам (ANN).
- **Reliability** — средняя близость OOT-запроса к его N соседям; мера
  доверия к перенесённой оценке.
- **Единица наблюдения** — строка (`qa`), реплика с историей
  (`turn_with_history`) или сессия (`dialogue`) по `assessment_mode` контракта.
- **Абсолютное снижение** — разница среднего `main_metric` корзины и оценки
  на мониторинге в долях (0.15 = 15 п.п.).
