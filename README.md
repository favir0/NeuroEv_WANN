# NeuroE_WANN
Проект реализован в рамках учебной программы по предмету "Нейроэволюционные вычисления" ТПУ

## Установка проекта
1. Установите poetry или используйте альтернативные менеджеры виртуальных окружений
2. Установите зависимости при помощи команды "poetry install" и активируйте его командой "poetry shell", все зависимости перечислены в файле pyproject.toml

## Запуск обучения WANN
1. Настройтe нужные параметры в файле config.py
2. Выберите один из предложенных или имплеметируйте свой task в trainer.py (поменяйте значение переменной task на нужную задачу в файле train.py)
3. Выполните команду "python train.py"

## Тестирование обученной WANN
1. Поменяйте значение переменной task на нужную задачу в файле test.py 
2. Выполните команду "python test.py"
