# check_directml.py
"""
Проверка поддержки DirectML на Windows
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

def check_directml():
    print("=" * 60)
    print("ПРОВЕРКА DIRECTML (WINDOWS + AMD)")
    print("=" * 60)

    try:
        import torch_directml
        print(f"✓ Модуль 'torch_directml' импортирован успешно.")

        # Безопасная проверка доступности
        if torch_directml.is_available():
            print(f"✓ DirectML доступен!")
            device_count = torch_directml.device_count()
            print(f"Количество устройств: {device_count}")

            for i in range(device_count):
                print(f"\nУстройство {i}:")
                print(f"  Имя: {torch_directml.device_name(i)}")
        else:
            print(f"\n✗ DirectML не доступен (is_available() вернул False).")
            print("Возможные причины:")
            print("1. Нет AMD/NVIDIA/Intel GPU с поддержкой DirectX 12.")
            print("2. Устарели драйверы GPU.")
            print("3. Версия Windows старше Windows 10 (требуется WDDM 2.0+).")

    except ImportError as e:
        print(f"\n✗ Не удалось импортировать 'torch_directml'.")
        print(f"Ошибка: {e}")
        print("\nУстановите пакет командой: pip install torch-directml")
    except Exception as e:
        # Ловим любые другие ошибки (например, проблемы с атрибутами)
        print(f"\n⚠️  Произошла неожиданная ошибка при проверке DirectML: {e}")
        print("Попробуйте переустановить пакет: pip install --upgrade torch-directml")

    print("\n" + "=" * 60)
    print("ПРОВЕРКА ВАШЕЙ КОНФИГУРАЦИИ:")

    try:
        from config import CONFIG
        print(f"Модель в config.py: {CONFIG.llm.model_name}")
        print(f"Устройство в config.py: {CONFIG.llm.device}")
    except Exception as e:
        print(f"Ошибка загрузки конфигурации: {e}")

    print("\n" + "=" * 60)
    print("РЕКОМЕНДАЦИИ:")
    print("1. Для установки: pip install torch-directml")
    print("2. В config.py используйте: device='auto'")
    print("3. Для проверки: python quick_test.py")
    print("=" * 60)

if __name__ == "__main__":
    check_directml()