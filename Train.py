import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import os
import gc

BASE_MODEL_PATH = r"D:\PythonProjects\LoRA poetry\models\Qwen2.5-3B-Instruct"

DATASETS = {
    "Иосиф Бродский": r"D:\PythonProjects\LoRA poetry\poetry\Brodskii.jsonl",
    "Михаил Лермонтов": r"D:\PythonProjects\LoRA poetry\poetry\Lermontov.jsonl",
    "Александр Пушкин": r"D:\PythonProjects\LoRA poetry\poetry\Pushkin.jsonl",
}

OUTPUT_DIR = r"D:\PythonProjects\LoRA poetry\models"

AUTHORS = {
    "Иосиф Бродский": "Brodskii",
    "Михаил Лермонтов": "Lermontov",
    "Александр Пушкин": "Pushkin",
}


def format_poem(example):
    """
    Форматируем пример для SFT/LoRA:
    добавляем инструкцию и ответ, сохраняем оригинальный текст.
    """
    return {
        "text": (
            f"### Instruction:\n"
            f"Напиши стихотворение в стиле {example['author']}.\n\n"
            f"### Response:\n"
            f"{example['text']}"
        )
    }


def train_lora(author):
    dataset_path = DATASETS[author]
    print(f"\n=== Обучение LoRA для {author} ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Загружаем токенизатор
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Загружаем базовую модель
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        dtype=torch.float16  # Используем fp16 для экономии видеопамяти
    ).to(device)

    # LoRA конфигурация
    lora_config = LoraConfig(
        r=8,  # Ранг LoRA. Меньшее значение = меньше памяти и быстрее обучение
        lora_alpha=16,  # Масштаб LoRA, уменьшено для ускорения
        target_modules=[
            "q_proj", "k_proj", "v_proj",
            "o_proj", "gate_proj", "up_proj", "down_proj"
        ],  # Модули модели для модификации
        lora_dropout=0.05,  # Дропаут LoRA для регуляризации
        bias="none",  # Смещения не изменяются
        task_type="CAUSAL_LM"  # Тип задачи — Causal Language Modeling
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Загружаем датасет
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    # Используем только 25% данных
    dataset = dataset.shuffle(seed=42).select(range(int(len(dataset) * 0.25)))

    # Форматируем текст для обучения
    dataset = dataset.map(format_poem, remove_columns=dataset.column_names)

    # Токенизация с ограничением длины последовательности
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=256)

    dataset = dataset.map(tokenize_fn, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, f"lora_{AUTHORS[author]}"),  # Куда сохранять модель
        num_train_epochs=5,  # Сокращаем количество эпох для быстрого обучения
        per_device_train_batch_size=2,  # Батч на GPU
        gradient_accumulation_steps=8,  # Эффективный батч = 1*4 = 4
        learning_rate=5e-4,  # Скорость обучения
        fp16=True,  # Используем смешанную точность
        logging_steps=20,  # Логирование каждые 20 шагов
        save_strategy="epoch",  # Сохраняем модель в конце каждой эпохи
        save_total_limit=1,  # Храним только последнюю модель
        report_to="none",  # Не отправляем логи в WandB или другие сервисы
        optim="adamw_torch"  # Оптимизатор
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args
    )

    # Запуск обучения
    trainer.train()

    # Сохраняем LoRA веса
    trainer.model.save_pretrained(
        os.path.join(OUTPUT_DIR, f"lora_{AUTHORS[author]}")
    )

    del trainer  # удаляем тренера, чтобы освободить GPU и CPU память
    del model  # удаляем модель на всякий случай
    gc.collect()  # собираем мусор Python
    torch.cuda.empty_cache()  # очищаем кэш CUDA
    print(f"LoRA для {author} сохранена.")
