from dataclasses import dataclass


@dataclass
class Configuration:
    # Генерация сети
    add_bias_node: bool = True                              # Добавление сдвига
    bias_value: float = 1                                   # Значение смещения
    max_depth: int = 10                                     # Максимальная глубина сети
    initial_connection_prob: float = 1                      # создания начальной связи для ноды инпута

    # Диапазоны значений
    weight_range: tuple[float, float] = (-2, 2)                 # Диапазон весов связей
    activation_response_range: tuple[float, float] = (0, 1)     # Диапазон отклика функции активации
    max_activation_response_delta: float = 0.3                  # Максимальное изменение отклика активации при мутации

    # Вероятности мутаций
    mutation_add_connection_prob: float = 0.1              # Вероятность добавления связи
    mutation_split_connection_prob: float = 0.1             # Вероятность разделения связи (добавления ноды)
    mutation_disable_connection_prob: float = 0.15          # Вероятность отключения связи
    mutation_enable_connection_prob: float = 0.15           # Вероятность включения связи
    mutation_change_activation_f_prob: float = 0.30         # Вероятность изменения функции активации
    mutation_change_activation_response_prob: float = 0.3   # Вероятность изменения отклика активации
    mutation_reenable_connection: float = 0.15              # Вероятность активации выключенной связи

    # Параметры популяции
    population_size: int = 100                              # Размер популяции
    single_structure_mutation: bool = False                 # Единичная структурная мутация
    reset_innovations: bool = True                          # Сброс списка инноваций
    target_species: int = 15                                # Целевое количество видов
    max_tournament_champions: int = 3                       # Максимальное количество чемпионов турнира

    # Совместимость
    distance_excess: float = 1.0                            # Расстояние для лишних генов
    distance_disjoint: float = 1.0                          # Расстояние для разобщенных генов
    distance_activation: float = 0.2    
    compatibility_threshold: float = 5                      # Порог совместимости
    compatibility_threshold_delta: float = 1                # Изменение порога совместимости
    min_compatibility_threshold: float = 0.1                # Минимальный порог совместимости

    # Элитизм
    elitism_enabled: bool = True                            # Включение элитизма
    specie_survival_rate: float = 0.3                       # Доля выживаемости для вида

    # Возрастная настройка
    allow_age_fitness_ajustment: bool = True                # Разрешить настройку фитнеса по возрасту
    young_age: int = 5                                      # Молодой возраст
    young_multiplier: float = 1.2                           # Множитель для молодого возраста
    old_age: int = 15                                       # Старый возраст
    old_multiplier: float = 0.5                             # Множитель для старого возраста

    # Стагнация
    stagnation_age: int = 15                                # Возраст стагнации

    # Параметры WANN (Weight Agnostic Neural Networks)
    wann_random_activation_on_init: bool = True             # Генерировать случайную функцию активации при инициализации новой ноды
    wann_initial_weight: float = 1                          # Начальный вес
    wann_use_weights_pool: bool = True                      # Использование пула весов
    wann_weights_pool = [-2, -1, 1, 2]                      # Пул весов
    wann_best_eval_multiplier: float = 0.7                  # Множитель для лучшего результата из пула
    wann_avg_eval_multiplier: float = 0.3                   # Множитель для среднего результата из пула
    wann_get_node_from_more_fit: bool = False               # Получение ноды из более приспособленного родителя
    wann_use_custom_fitness: bool = False                   # Использование пользовательской функции фитнеса
    

config = Configuration()
