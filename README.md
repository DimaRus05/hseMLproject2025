# Проект: Оценка токсичности сообщений

Учебный проект: классификация токсичности русскоязычных текстов.

Автор: Русанов Дмитрий Сергеевич

## Запуск

1) Установка зависимостей:

```powershell
pip install -r requirements.txt
```

2) Подготовка данных (скачивание HF-датасетов и объединение):

```powershell
python scripts\download_hf_datasets.py --outdir data\hf_raw
python scripts\prepare_combined.py --input data\hf_raw --out data\ru_toxic\combined.csv
```

3) Обучение baseline и получение OOF-предсказаний:

```powershell
python scripts\train_baseline.py --input data\ru_toxic\combined.csv \
	--oof_out data\ru_toxic\combined_oof_full.csv \
	--model_out models\calibrated_model_full.joblib
```

4) Оценка модели на отдельном CSV (пример):

```powershell
python scripts\evaluate_model.py models\calibrated_model_full.joblib data\ru_toxic\sample_small.csv
```

5) Запуск Telegram-бота (локально, polling):

```powershell
python bot\telegram_bot.py --token "<TG-TOKEN>" --model models\calibrated_model_full.joblib
```

## Ноутбук с анализом

Файл `notebooks/analysis.ipynb` строит диагностические графики (ROC, Precision–Recall, reliability diagram, гистограмма вероятностей, матрица ошибок). Нотебук теперь показывает графики в интерактивной среде и не сохраняет картинки в файлы — всё отображается inline.

Если вам нужно сохранить отдельные графики на диск, открывайте ноутбук и вручную сохраняйте через меню или добавляйте `plt.savefig(...)` в нужных ячейках.

## Структура и важные пути

- Данные: `data/ru_toxic/combined.csv`, `data/ru_toxic/combined_oof_full.csv`
- Модели: `models/calibrated_model_full.joblib`
- Скрипты: `scripts/` (download, prepare, train, evaluate, run_full_pipeline)
- Телеграм-бот: `bot/telegram_bot.py`
- Анализ: `notebooks/analysis.ipynb`

## Ручная проверка и быстрая инструкция

Подробные пошаговые инструкции по запуску (виртуальное окружение, скачивание данных, обучение, оценка, консольное приложение, Telegram-бот) находятся в `docs/manual_testing.md`.
