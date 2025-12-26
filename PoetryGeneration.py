import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
from peft import PeftModel
import gc



def generate_poem (author, lora_path = ""):

    model_path = r"D:\PythonProjects\LoRA poetry\models\Qwen2.5-3B-Instruct"

    # -----------------------------
    # 1. Список тем для генерации
    # -----------------------------
    topics = [
        "любовь", "осень", "природа", "ночь", "дружба",
        "вдохновение", "мечты", "память", "радость", "грусть"
    ]
    topic = random.choice(topics)

    # -----------------------------
    # 2. Выбор устройства
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}")

    # -----------------------------
    # 3. Загрузка токенизатора
    # -----------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -----------------------------
    # 4. Загрузка модели
    # -----------------------------
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.float16
    ).to(device)

    if lora_path != "":
        model = PeftModel.from_pretrained(model, lora_path)

    model.eval()

    # -----------------------------
    # 5. Составление промта
    # -----------------------------
    prompt = (
        f"Напиши стихотворение из 4 четверостиший на тему '{topic}' "
        f"в стиле поэта {author}. Каждое четверостишие должно быть художественным и ритмичным."
    )

    # Токенизация и перенос на device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # -----------------------------
    # 6. Генерация текста
    # -----------------------------
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=300,   # примерно 4 четверостишия
            do_sample=True,
            temperature=0.8,
            top_p=0.95
        )

    # Декодирование
    result = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # -----------------------------
    # 7. Вывод
    # -----------------------------
    model_name = "Qwen2.5-3B-Instruct" if "lora" not in model_path else "дообученная LoRA"
    print("\n=== Генерация стихов ===")
    print(f"Модель: {model_name}\n")
    print(f"Промт: {prompt}\n")
    print(result)

    del model  # удаляем модель на всякий случай
    gc.collect()  # собираем мусор Python
    torch.cuda.empty_cache()  # очищаем кэш CUDA
