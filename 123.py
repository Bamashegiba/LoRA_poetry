import json
import re


def print_jsonl_file_readable(file_path, limit=None):
    """
    Выводит содержимое JSONL файла в консоль в читаемом виде.

    :param file_path: путь к JSONL файлу
    :param limit: максимальное количество записей для вывода (None = все)
    """
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            print("=" * 80)
            print(f"Автор: {data.get('author', 'Unknown')}\n")
            print(data.get('text', ''))
            print("=" * 80 + "\n")

            count += 1
            if limit and count >= limit:
                break


# Пример использования:
file_path = r"D:\PythonProjects\LoRA poetry\poetry\Pushkin.jsonl"
# print_jsonl_file_readable(file_path, limit=40)  # выводим первые 5 записей

def parse_poems_to_jsonl(input_file, output_file, author_name):
    """
    Парсит файл с текстами стихотворений и сохраняет их в формате JSONL для LoRA.

    Правила:
    - Начало стиха: '\t\t'
    - Конец стиха:
        1) встреча с символом, не являющимся текстом или знаками препинания (*, [, ], {, }, <, >)
        2) блок: перенос строки, сразу цифра, снова перенос строки
    - Сохраняется форматирование (переносы строк), убирается только '\t\t'
    - Удаляется всё, что находится внутри квадратных скобок [ ]
    """
    # Регулярка для проверки конца стихотворения по символам
    end_pattern = re.compile(r'[^A-Za-zА-Яа-яЁё0-9\s.,!?;:()\-–—\'"]')
    # Регулярка для удаления текста в квадратных скобках
    brackets_pattern = re.compile(r'\[.*?\]', re.DOTALL)
    # Регулярка для блока "\n<цифра>\n"
    block_digit_pattern = re.compile(r'^\d+$')

    poems = []
    current_poem = []
    inside_poem = False

    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            # Если начало нового стиха
            if line.startswith('\t\t'):
                inside_poem = True
                clean_line = line.replace('\t\t', '', 1).rstrip()
                clean_line = brackets_pattern.sub('', clean_line)
                current_poem.append(clean_line)
                i += 1
                continue

            if inside_poem:
                stripped = line.strip()
                # Проверяем на конец стихотворения по символам
                end_by_symbol = end_pattern.search(stripped)
                # Проверяем на блок "\n<цифра>\n"
                end_by_digit_block = False
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if stripped == '' and block_digit_pattern.match(next_line):
                        end_by_digit_block = True

                if end_by_symbol or end_by_digit_block:
                    # Завершение текущего стиха
                    if current_poem:
                        poem_text = '\n'.join(current_poem).strip()
                        if poem_text:
                            poems.append({"author": author_name, "text": poem_text})
                    current_poem = []
                    inside_poem = False
                    # Если конец по блоку, пропускаем цифру
                    if end_by_digit_block:
                        i += 2
                        continue
                else:
                    clean_line = brackets_pattern.sub('', line.rstrip())
                    current_poem.append(clean_line)
            i += 1

    # На случай, если файл закончился внутри стиха
    if inside_poem and current_poem:
        poem_text = '\n'.join(current_poem).strip()
        poems.append({"author": author_name, "text": poem_text})

    # Запись в jsonl
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for poem in poems:
            f_out.write(json.dumps(poem, ensure_ascii=False) + '\n')

    print(f"Сохранено {len(poems)} стихотворений в {output_file}")


# Пример использования
input_path = r"D:\PythonProjects\LoRA poetry\poetry\Lermontov.txt"
output_path = r"D:\PythonProjects\LoRA poetry\poetry\Lermontov.jsonl"
# parse_poems_to_jsonl(input_path, output_path, author_name="Михаил Лермонтов")




def print_poem_with_visible_whitespace(file_path: str, max_lines: int = 10000):
    """
    Выводит первые max_lines строк файла, заменяя все спецсимволы на видимые escape-последовательности,
    но оставляя читаемый текст.
    """
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            # Заменяем специальные символы на видимые
            visible_line = line.replace("\t", "\\t").replace("\n", "\\n")
            print(visible_line)

# Пример использования:
file_path = r"D:\PythonProjects\LoRA poetry\poetry\Pushkin.txt"
# print_poem_with_visible_whitespace(file_path)


def parse_lermontov_to_jsonl1(input_file, output_file, author_name):
    """
    Парсит файл с текстами стихотворений Лермонтова и сохраняет их в формате JSONL для LoRA.

    Правила:
    - Начало стиха: '\t\t'
    - Конец стиха:
        1) встреча с символом, не являющимся текстом или знаками препинания (*, [, ], {, }, <, >)
        2) конструкция "***"
        3) три подряд переноса строки "\n\n\n"
    - Сохраняется форматирование (переносы строк), убирается только '\t\t'
    """
    # Регулярка для проверки конца стихотворения по символам
    end_pattern = re.compile(r'[^A-Za-zА-Яа-яЁё0-9\s.,!?;:()\-–—\'"]')

    poems = []
    current_poem = []
    inside_poem = False
    consecutive_empty = 0  # счетчик подряд пустых строк

    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i]

            # Начало нового стиха
            if line.startswith('\t\t'):
                inside_poem = True
                clean_line = line.replace('\t\t', '', 1).rstrip()
                current_poem.append(clean_line)
                consecutive_empty = 0
                i += 1
                continue

            if inside_poem:
                stripped = line.strip()

                # Проверка на конец по символам
                end_by_symbol = end_pattern.search(stripped)
                # Проверка на конструкцию "***"
                end_by_stars = stripped == "***"
                # Проверка на три подряд пустые строки
                if stripped == "":
                    consecutive_empty += 1
                else:
                    consecutive_empty = 0
                end_by_triple_newline = consecutive_empty >= 3

                if end_by_symbol or end_by_stars or end_by_triple_newline:
                    # Завершение текущего стиха
                    if current_poem:
                        poem_text = '\n'.join(current_poem).strip()
                        if poem_text:
                            poems.append({"author": author_name, "text": poem_text})
                    current_poem = []
                    inside_poem = False
                    consecutive_empty = 0
                    i += 1
                    continue
                else:
                    current_poem.append(line.rstrip())
            i += 1

    # На случай, если файл закончился внутри стиха
    if inside_poem and current_poem:
        poem_text = '\n'.join(current_poem).strip()
        poems.append({"author": author_name, "text": poem_text})

    # Запись в jsonl
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for poem in poems:
            f_out.write(json.dumps(poem, ensure_ascii=False) + '\n')

    print(f"Сохранено {len(poems)} стихотворений в {output_file}")

# Пример вызова:
# parse_lermontov_to_jsonl1(r"D:\PythonProjects\LoRA poetry\poetry\Pushkin.txt",r"D:\PythonProjects\LoRA poetry\poetry\Pushkin.jsonl","Александр Пушкин")

