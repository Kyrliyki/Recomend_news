#!/usr/bin/env python3
"""
Главное меню для управления пайплайном рекомендательной системы
Запуск: python main_menu.py
"""

import sys
import os

# Добавляем Colab в путь для импорта модулей
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Colab'))

from Colab.config import Config
from Colab.utils import check_model_exists, get_model_info


def print_header():
    """Печать заголовка меню"""
    print("\n" + "=" * 70)
    print("    СИСТЕМА РЕКОМЕНДАЦИЙ НОВОСТЕЙ - ПАЙПЛАЙН")
    print("=" * 70)
    print(f"   Датасет: {Config.DATA_PATH}")
    print(f"   Размер эмбеддингов: {Config.EMBEDDING_DIM}")
    print(f"   Модель сохранена: {' Да' if check_model_exists() else ' Нет'}")
    print("=" * 70)


def print_menu():
    """Печать пунктов меню"""
    print("\n ДОСТУПНЫЕ ОПЕРАЦИИ:")
    print("-" * 70)
    print("   1.  Шаг 1: Загрузка и EDA данных")
    print("   2.  Шаг 2: Предобработка данных")
    print("   3.  Шаг 3: Создание индексов и сплит данных")
    print("   4.  Шаг 4: Обучение модели")
    print("   5.  Шаг 5: Сохранение модели")
    print("   6.  Шаг 6: Оценка и анализ модели")
    print("   7.  Шаг 7: Запуск рекомендательной системы (интерактив)")
    print("-" * 70)
    print("   8.  Запуск ВСЕГО пайплайна (шаги 1-6)")
    print("   9.  Информация о сохраненной модели")
    print("   0.  Выход")
    print("-" * 70)



def run_step(step_name, module_name, function_name, *args):
    """Универсальный запуск шага пайплайна"""
    print("\n" + "=" * 70)
    print(f"  ЗАПУСК: {step_name}")
    print("=" * 70)

    try:
        # Динамический импорт модуля
        module = __import__(f"Colab.{module_name}", fromlist=[function_name])
        func = getattr(module, function_name)

        # Запуск функции
        result = func(*args) if args else func()

        print(f"\n {step_name} - УСПЕШНО ЗАВЕРШЕН!")
        return result

    except Exception as e:
        print(f"\n ОШИБКА в {step_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Главная функция меню"""
    config = Config()

    while True:
        print_header()
        print_menu()

        choice = input("\n Введите номер операции: ").strip()

        # 1. Загрузка и EDA
        if choice == '1':
            run_step("Загрузка и EDA данных", "data_loader", "load_and_explore", config.DATA_PATH)

        # 2. Предобработка
        elif choice == '2':
            result = run_step("Предобработка данных", "preprocessing", "run_preprocessing",
                              config.DATA_PATH, config.MIN_CLICK_CUTOFF)
            if result:
                print("\n Результат предобработки:")
                print(f"   - Сохранено в: {result}")

        # 3. Создание индексов и сплит
        elif choice == '3':
            run_step("Создание индексов и разделение данных",
                     "preprocessing", "create_indices_and_split",
                     config.DATA_PATH, config.TEST_RATIO, config.MIN_CLICK_CUTOFF)

        # 4. Обучение модели
        elif choice == '4':
            run_step("Обучение модели", "train", "train_model",
                     config.DATA_PATH, config.BATCH_SIZE,
                     config.EMBEDDING_DIM, config.EPOCHS,
                     config.TEST_RATIO, config.MODEL_PATH,
                     config.INDICES_PATH)

        # 5. Сохранение модели (обычно уже сохраняется в шаге 4)
        elif choice == '5':
            run_step("Сохранение модели", "utils", "save_current_model",
                     config.MODEL_PATH, config.INDICES_PATH)

        # 6. Оценка и анализ
        elif choice == '6':
            run_step("Оценка и анализ модели", "train", "evaluate_model",
                     config.MODEL_PATH, config.INDICES_PATH, config.DATA_PATH)

        # 7. Интерактивная рекомендательная система
        elif choice == '7':
            run_step("Запуск рекомендательной системы", "recommend", "run_recommendation_app",
                     config.MODEL_PATH, config.INDICES_PATH, config.DATA_PATH)

        # 8. Весь пайплайн целиком
        elif choice == '8':
            print("\n" + "-" * 35)
            print("   ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА (обучение + анализ)")
            print("-" * 35)

            steps = [
                ("Загрузка данных", "data_loader", "load_data", config.DATA_PATH),
                ("Предобработка", "preprocessing", "run_preprocessing", config.DATA_PATH, config.MIN_CLICK_CUTOFF),
                ("Обучение модели", "train", "train_model",
                 config.DATA_PATH, config.BATCH_SIZE,
                 config.EMBEDDING_DIM, config.EPOCHS,
                 config.TEST_RATIO, config.MODEL_PATH,
                 config.INDICES_PATH),
                ("Анализ модели", "train", "evaluate_model",
                 config.MODEL_PATH, config.INDICES_PATH, config.DATA_PATH)
            ]

            success = True
            for step_name, module_name, func_name, *args in steps:
                result = run_step(step_name, module_name, func_name, *args)
                if result is None:
                    success = False
                    print(f"\n️  Пайплайн прерван на шаге: {step_name}")
                    break

            if success:
                print("\n" + "-" * 35)
                print("   ВЕСЬ ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЕН!")
                print("   Теперь можно запустить рекомендательную систему (пункт 7)")
                print("-" * 35)

        # 9. Информация о модели
        elif choice == '9':
            info = get_model_info(config.MODEL_PATH, config.INDICES_PATH)
            if info:
                print("\n" + "=" * 70)
                print(" ИНФОРМАЦИЯ О СОХРАНЕННОЙ МОДЕЛИ")
                print("=" * 70)
                for key, val in info.items():
                    print(f"   {key}: {val}")
            else:
                print("\n Сохраненная модель не найдена. Сначала обучите модель (пункт 4)")

        # 0. Выход
        elif choice == '0':
            print("\n До свидания! Хорошего дня!")
            break

        else:
            print("\n Неверный ввод. Пожалуйста, выберите пункт от 0 до 9.")

        input("\n\n Нажмите Enter, чтобы продолжить...")


if __name__ == "__main__":
    main()