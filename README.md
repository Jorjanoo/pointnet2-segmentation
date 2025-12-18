# PointNet++ для семантической сегментации облака точек

Проект для обучения нейронной сети PointNet++ для семантической сегментации 3D-облаков точек.

## Описание

Реализация архитектуры PointNet++ для задачи семантической сегментации, где каждой точке 3D-облака присваивается метка класса.

### Особенности

- Полная реализация архитектуры PointNet++ с Set Abstraction и Feature Propagation слоями
- Поддержка PLY файлов
- Обучение с Adam оптимизатором и StepLR scheduler
- Логирование в TensorBoard
- Метрики: Accuracy, IoU (Intersection over Union)
- Визуализация результатов

## Требования

- Python 3.8+
- PyTorch 1.9.0+
- CUDA (опционально, для GPU)

## Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/Jorjanoo/pointnet2-segmentation.git
cd pointnet2-segmentation
```

2. Создайте виртуальное окружение (рекомендуется):
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows
```

3. Установите зависимости:
```bash
pip install -r requirements.txt
```

**Примечание:** Open3D опциональна (не поддерживает Python 3.13), но код работает без неё.

## Структура проекта

```
├── dataset.py          # Класс Dataset для загрузки данных
├── pointnet2.py        # Архитектура PointNet++
├── train.py            # Скрипт обучения
├── test.py             # Скрипт тестирования
├── visualize.py        # Скрипт визуализации
├── requirements.txt    # Зависимости проекта
├── colab_setup.ipynb   # Ноутбук для Google Colab
└── README.md           # Документация
```

## Подготовка данных

Данные должны быть в формате PLY файлов со структурой:
```
data_dir/
└── area/
    ├── file1.ply
    ├── file2.ply
    └── ...
```

Каждый PLY файл должен содержать точки в формате:
- `x, y, z, label` (4 колонки)
- или `x, y, z, r, g, b, label` (7 колонок)

## Использование

### Обучение модели

```bash
python train.py \
    --data_dir 3011-20251217T195928Z-1-001 \
    --area 3011 \
    --num_points 2048 \
    --batch_size 8 \
    --epochs 50 \
    --lr 0.001 \
    --num_classes 13 \
    --device cuda
```

Параметры:
- `--data_dir`: путь к директории с данными
- `--area`: область для использования
- `--num_points`: количество точек в каждом облаке (по умолчанию 4096)
- `--batch_size`: размер батча (8-16)
- `--epochs`: количество эпох (50-100)
- `--lr`: начальный learning rate (по умолчанию 0.001)
- `--num_classes`: количество классов
- `--device`: устройство (cuda или cpu)
- `--save_dir`: директория для сохранения чекпоинтов (по умолчанию ./checkpoints)
- `--log_dir`: директория для логов TensorBoard (по умолчанию ./logs)

### Просмотр обучения в TensorBoard

```bash
tensorboard --logdir ./logs
```

Откройте браузер и перейдите по адресу `http://localhost:6006`

### Тестирование модели

```bash
python test.py \
    --data_dir 3011-20251217T195928Z-1-001 \
    --area 3011 \
    --checkpoint ./checkpoints/best_model.pth \
    --num_classes 13
```

### Визуализация результатов

```bash
python visualize.py \
    --data_dir 3011-20251217T195928Z-1-001 \
    --area 3011 \
    --checkpoint ./checkpoints/best_model.pth \
    --num_classes 13 \
    --num_samples 5 \
    --output_dir ./visualizations
```

Подробнее: [VISUALIZE.md](VISUALIZE.md)

## Архитектура модели

PointNet++ состоит из двух основных компонентов:

1. **Encoder (Set Abstraction Layers)**: 
   - Иерархическая выборка точек (Farthest Point Sampling)
   - Группировка точек (Ball Query)
   - Извлечение локальных признаков

2. **Decoder (Feature Propagation Layers)**:
   - Интерполяция признаков обратно на исходные точки
   - Комбинирование локальных и глобальных признаков
   - Финальная классификация

## Запуск в Google Colab

**Данные уже включены в репозиторий!**

Откройте ноутбук `colab_setup.ipynb` в Google Colab и выполните ячейки - все настроено автоматически.

Или клонируйте репозиторий:
```python
!git clone https://github.com/Jorjanoo/pointnet2-segmentation.git
%cd pointnet2-segmentation
%pip install torch torchvision numpy scikit-learn tqdm matplotlib tensorboard -q
```

## Устранение неполадок

### Out of Memory
- Уменьшите `batch_size` (например, до 4)
- Уменьшите `num_points` (например, до 2048)

### Данные не загружаются
- Проверьте путь к данным
- Убедитесь, что файлы имеют правильный формат PLY

### Медленное обучение
- Используйте GPU (CUDA)
- Уменьшите `num_points` или `batch_size`

