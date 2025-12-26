from PoetryGeneration import generate_poem
from Train import train_lora

def no_lora_generation():
    while True:
        print("\n=== Выбор автора. Модель без дообучения ===")
        print("1. Пушкин")
        print("2. Лермонтов")
        print("3. Бродский")
        print("4. Выход")

        choice = input("Выберите вариант (1-4): ").strip()

        if choice == "1":
            generate_poem(author="Алексанр Сергеевич Пушкин")
        elif choice == "2":
            generate_poem(author="Михаил Юрьевич Лермонтов")
        elif choice == "3":
            generate_poem(author="Иосиф Александрович Бродсикй")
        elif choice == "4":
            print("Выход в главное меню...")
            break
        else:
            print("Неверный выбор. Попробуйте снова.")

def lora_generation():
    while True:
        print("\n=== Выбор автора. Модель с дообучением ===")
        print("1. Пушкин")
        print("2. Лермонтов")
        print("3. Бродский")
        print("4. Выход")

        choice = input("Выберите вариант (1-4): ").strip()

        if choice == "1":
            generate_poem(author="Алексанр Сергеевич Пушкин", lora_path="D:\PythonProjects\LoRA poetry\models\lora_Pushkin")
        elif choice == "2":
            generate_poem(author="Михаил Юрьевич Лермонтов", lora_path="D:\PythonProjects\LoRA poetry\models\lora_Lermontov")
        elif choice == "3":
            generate_poem(author="Иосиф Александрович Бродсикй", lora_path="D:\PythonProjects\LoRA poetry\models\lora_Brodskii")
        elif choice == "4":
            print("Выход в главное меню...")
            break
        else:
            print("Неверный выбор. Попробуйте снова.")

def compilation():
    while True:
        print("\n=== Выбор автора для сравнения моделей===")
        print("1. Пушкин")
        print("2. Лермонтов")
        print("3. Бродский")
        print("4. Выход")

        choice = input("Выберите вариант (1-4): ").strip()

        if choice == "1":
            generate_poem(author="Алексанр Сергеевич Пушкин")
            generate_poem(author="Алексанр Сергеевич Пушкин", lora_path="D:\PythonProjects\LoRA poetry\models\lora_Pushkin")
        elif choice == "2":
            generate_poem(author="Михаил Юрьевич Лермонтов")
            generate_poem(author="Михаил Юрьевич Лермонтов", lora_path="D:\PythonProjects\LoRA poetry\models\lora_Lermontov")
        elif choice == "3":
            generate_poem(author="Иосиф Александрович Бродсикй")
            generate_poem(author="Иосиф Александрович Бродсикй",
                          lora_path="D:\PythonProjects\LoRA poetry\models\lora_Brodskii")
        elif choice == "4":
            print("Выход в главное меню...")
            break
        else:
            print("Неверный выбор. Попробуйте снова.")

def main_menu():
    while True:
        print("\n=== Главное меню ===")
        print("1. Генерация стихов без дообучения")
        print("2. Генерация стихов с дообучением")
        print("3. Сравнение генерации по одному запросу")
        print("4. Дообучение LoRA")
        print("5. Выход")

        choice = input("Выберите вариант (1-4): ").strip()

        if choice == "1":
            no_lora_generation()
        elif choice == "2":
            lora_generation()
        elif choice == "3":
            compilation()
        elif choice == "4":
            train_lora(author="Александр Пушкин")
            train_lora(author="Михаил Лермонтов")
            train_lora(author="Иосиф Бродский")
        elif choice == "4":
            print("Выход из программы")
            break
        else:
            print("Неверный выбор. Попробуйте снова.")

if __name__ == "__main__":
    main_menu()