# Команды для визуализации результатов

## Базовая команда

```bash
python visualize.py \
    --data_dir 3011-20251217T195928Z-1-001 \
    --area 3011 \
    --checkpoint ./checkpoints/best_model.pth \
    --num_classes 13 \
    --num_samples 5 \
    --output_dir ./visualizations
```

## Параметры

- `--data_dir`: путь к данным (3011-20251217T195928Z-1-001)
- `--area`: область данных (3011)
- `--checkpoint`: путь к обученной модели (обязательно!)
- `--num_classes`: количество классов (13 или проверьте в данных)
- `--num_samples`: количество примеров для визуализации (по умолчанию 5)
- `--output_dir`: директория для сохранения результатов
- `--use_open3d`: использовать Open3D вместо matplotlib (опционально)

## Примеры

### Локально (после обучения):

```bash
python visualize.py \
    --data_dir 3011-20251217T195928Z-1-001 \
    --area 3011 \
    --checkpoint ./checkpoints/best_model.pth \
    --num_classes 13 \
    --num_samples 10
```

### В Colab:

```python
!python visualize.py \
    --data_dir 3011-20251217T195928Z-1-001 \
    --area 3011 \
    --checkpoint ./checkpoints/best_model.pth \
    --num_classes 13 \
    --num_samples 5 \
    --output_dir ./visualizations
```

### С Open3D (если установлен):

```bash
python visualize.py \
    --data_dir 3011-20251217T195928Z-1-001 \
    --area 3011 \
    --checkpoint ./checkpoints/best_model.pth \
    --num_classes 13 \
    --use_open3d \
    --output_dir ./visualizations
```

## Результаты

Визуализации сохраняются в формате PNG (или PLY для Open3D) в указанную директорию:
- `sample_0.png` - первый пример
- `sample_1.png` - второй пример
- и т.д.

Каждый файл содержит 3 графика:
1. Ground Truth (истинные метки)
2. Predictions (предсказания модели)
3. Errors (ошибки - красным цветом)

