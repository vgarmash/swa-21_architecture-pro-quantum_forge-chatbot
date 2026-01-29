# core/llm_client.py
"""
Клиент для работы с локальной LLM моделью с поддержкой DirectML (AMD GPU)
"""

import warnings
import time
from typing import Optional, Dict, Any, List
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import torch
import logging

from config import CONFIG

logger = logging.getLogger(__name__)

# Проверяем доступность DirectML
try:
    import torch_directml
    DIRECTML_AVAILABLE = True
except ImportError:
    DIRECTML_AVAILABLE = False
    torch_directml = None


class QwenLLMClient:
    """Клиент для работы с моделью с поддержкой DirectML"""

    def __init__(self, config=None):
        """
        Инициализация клиента LLM

        Args:
            config: Конфигурация LLM
        """
        self.config = config or CONFIG.llm
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModelForCausalLM] = None
        self._cpu_model: Optional[AutoModelForCausalLM] = None
        self._device = None
        self._directml_device = None

    def _determine_device(self) -> str:
        """
        Определение устройства для вычислений с поддержкой DirectML

        Returns:
            str: Идентификатор устройства ('directml', 'cuda', 'mps', 'cpu')
        """
        if self.config.device != "auto":
            return self.config.device

        # 1. Проверяем DirectML (Windows AMD)
        if DIRECTML_AVAILABLE and torch_directml.is_available():
            logger.info("Обнаружена AMD GPU через DirectML")
            return "directml"

        # 2. Проверяем CUDA (NVIDIA GPU)
        if torch.cuda.is_available():
            logger.info("Обнаружена NVIDIA GPU (CUDA)")
            return "cuda"

        # 3. Проверяем MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("Обнаружен MPS (Apple Silicon)")
            return "mps"

        # 4. Иначе CPU
        logger.info("Используем CPU")
        return "cpu"

    def _setup_directml_optimizations(self):
        """Настройка оптимизаций для DirectML"""
        if not DIRECTML_AVAILABLE or self._device != "directml":
            return

        try:
            logger.info("Настройка DirectML...")

            # Получаем DirectML устройство
            self._directml_device = torch_directml.device()
            device_name = torch_directml.device_name(0) if torch_directml.device_count() > 0 else "Неизвестно"
            logger.info(f"DirectML устройство: {device_name}")

            # Базовые оптимизации
            torch.backends.cudnn.enabled = False  # cudnn не для DirectML

        except Exception as e:
            logger.warning(f"Не удалось настроить DirectML: {e}")
            self._device = "cpu"  # Fallback на CPU

    def _get_dtype(self) -> torch.dtype:
        """Определение dtype для модели"""
        if self._device in ["cuda", "directml"]:
            return torch.float16  # Используем float16 для GPU
        elif self._device == "mps":
            return torch.float16
        else:
            return torch.float32  # float32 для CPU

    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Создание конфигурации для quantization"""
        # DirectML не поддерживает quantization, только для CUDA
        if self._device != "cuda":
            return None

        if not self.config.load_in_8bit and not self.config.load_in_4bit:
            return None

        try:
            if self.config.load_in_8bit:
                logger.info("Используется 8-bit quantization")
                return BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False
                )
            elif self.config.load_in_4bit:
                logger.info("Используется 4-bit quantization")
                return BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
        except ImportError as e:
            logger.warning(f"Не удалось настроить quantization: {e}")
            return None

        return None

    def load_model(self):
        """Загрузка модели и токенизатора"""
        if self._model is not None and self._tokenizer is not None:
            return

        # Определяем устройство
        self._device = self._determine_device()

        # Настраиваем оптимизации для устройства
        if self._device == "directml":
            self._setup_directml_optimizations()

        logger.info(f"Загрузка модели {self.config.model_name} на устройство {self._device}...")

        try:
            # Специальная обработка для Phi-3
            if "phi-3" in self.config.model_name.lower():
                # Для Phi-3 загружаем токенизатор с минимальными параметрами
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_name,
                    trust_remote_code=False,  # Phi-3 не требует trust_remote_code
                    cache_dir=str(self.config.model_cache_dir),
                    use_fast=True
                )
            else:
                # Для других моделей
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_name,
                    trust_remote_code=self.config.trust_remote_code,
                    cache_dir=str(self.config.model_cache_dir)
                )

            # Устанавливаем pad_token если отсутствует
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            # Определяем dtype
            dtype = self._get_dtype()

            # Настройки загрузки модели
            model_kwargs = {
                "dtype": dtype,
                "trust_remote_code": self.config.trust_remote_code if "phi-3" not in self.config.model_name.lower() else False,
                "cache_dir": str(self.config.model_cache_dir),
                "low_cpu_mem_usage": True,
            }

            # Добавляем quantization только для CUDA
            quantization_config = self._get_quantization_config()
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
                if self.config.load_in_4bit:
                    model_kwargs["dtype"] = torch.float16

            # Настройка device_map
            if self._device == "cuda":
                model_kwargs["device_map"] = "auto"
            elif self._device in ["cpu", "directml"]:
                model_kwargs["device_map"] = None
            else:
                model_kwargs["device_map"] = self._device

            # Используем flash attention если доступно и не DirectML
            if (hasattr(self.config, 'use_flash_attention') and
                    self.config.use_flash_attention and
                    self._device == "cuda"):
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Используется flash attention 2")

            # Подавляем предупреждения
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Загружаем модель
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    **model_kwargs
                )

            # Перемещаем модель на нужное устройство
            if self._device == "directml" and self._directml_device is not None:
                self._model = self._model.to(self._directml_device)
            elif self._device == "cpu":
                self._model = self._model.to("cpu")
                self._model = self._model.float()  # Конвертируем в float32 для CPU
            elif self._device == "mps":
                self._model = self._model.to("mps")
            # Для CUDA с device_map="auto" модель уже на GPU

            self._model.eval()

            # Информация о модели
            params = sum(p.numel() for p in self._model.parameters())
            logger.info(f"Модель загружена успешно на {self._device}")
            logger.info(f"Параметров модели: {params:,}")

        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")

            # Пробуем загрузить без quantization если была ошибка
            if self.config.load_in_8bit or self.config.load_in_4bit:
                logger.info("Пробуем загрузить модель без quantization...")
                try:
                    self.config.load_in_8bit = False
                    self.config.load_in_4bit = False
                    self._model = None
                    self._tokenizer = None
                    self.load_model()
                    logger.warning("Модель загружена без quantization")
                    return
                except Exception as e2:
                    logger.error(f"Не удалось загрузить даже без quantization: {e2}")

            raise RuntimeError(f"Ошибка загрузки модели: {e}")

    def _move_to_device(self, inputs):
        """Перемещение входных данных на нужное устройство"""
        if self._device == "directml" and self._directml_device is not None:
            # ПРАВИЛЬНЫЙ СПОСОБ для DirectML
            return {k: v.to(self._directml_device) for k, v in inputs.items()}
        elif self._device == "cuda":
            return {k: v.cuda() for k, v in inputs.items()}
        elif self._device == "mps":
            return {k: v.to("mps") for k, v in inputs.items()}
        else:
            return inputs

    def generate_response(
            self,
            prompt: str,
            max_new_tokens: Optional[int] = None,
            temperature: Optional[float] = None
    ) -> str:
        """
        Генерация ответа на основе промпта

        Args:
            prompt: Текст промпта
            max_new_tokens: Максимальное количество новых токенов
            temperature: Температура для генерации

        Returns:
            str: Сгенерированный ответ
        """
        if self._model is None or self._tokenizer is None:
            self.load_model()

        # Параметры генерации
        max_tokens = self.config.max_tokens if max_new_tokens is None else max_new_tokens
        temp = self.config.temperature if temperature is None else temperature

        # Оптимизации для разных устройств
        if self._device == "cpu":
            # Для CPU используем меньшие значения
            max_tokens = min(max_tokens, 256)

        try:
            # Мягкая обрезка по символам (страховка от сбоев)
            max_prompt_chars = 100000  # 100k символов
            if len(prompt) > max_prompt_chars:
                logger.warning(f"Промпт очень длинный ({len(prompt)} символов), грубо обрезаем до {max_prompt_chars}")
                # Простое обрезание с конца, если промпт аномально огромен
                prompt = prompt[:max_prompt_chars]

            # Токенизация промпта с БОЛЬШИМ лимитом для контекста Phi-3
            max_length = min(getattr(self._tokenizer, "model_max_length", 32768), 32768)
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,  # Используем не больше model_max_length
                padding=True
            )

            # ОТЛАДОЧНЫЙ ЛОГ: проверка токенизации
            logger.debug(f"=== ПРОВЕРКА ТОКЕНИЗАЦИИ ===")
            logger.debug(f"Ключи в inputs: {list(inputs.keys())}")
            if 'input_ids' in inputs:
                input_ids = inputs['input_ids']
                logger.debug(f"input_ids - тип: {input_ids.dtype}, форма: {input_ids.shape}, устройство: {input_ids.device}")
                logger.debug(f"input_ids первые 10 токенов: {input_ids[0, :10].tolist()}")
                logger.debug(f"input_ids последние 10 токенов: {input_ids[0, -10:].tolist()}")

                # Проверка на специальные токены
                bos_token_id = self._tokenizer.bos_token_id
                eos_token_id = self._tokenizer.eos_token_id
                pad_token_id = self._tokenizer.pad_token_id
                logger.debug(f"Специальные токены: BOS={bos_token_id}, EOS={eos_token_id}, PAD={pad_token_id}")

                # Проверка, что в начале промпта нет случайных байтов/битых данных
                first_tokens = input_ids[0, :5].tolist()
                if any(token < 0 for token in first_tokens):
                    logger.error(f"ОБНАРУЖЕНЫ ОТРИЦАТЕЛЬНЫЕ ТОКЕНЫ в начале: {first_tokens}")
                if any(token > 1000000 for token in first_tokens):  # Невероятно большой ID токена
                    logger.error(f"ОБНАРУЖЕНЫ НЕВЕРОЯТНО БОЛЬШИЕ ТОКЕНЫ: {first_tokens}")

            if 'attention_mask' in inputs:
                attention_mask = inputs['attention_mask']
                logger.debug(f"attention_mask - тип: {attention_mask.dtype}, форма: {attention_mask.shape}")
                logger.debug(f"attention_mask сумма (сколько реальных токенов): {attention_mask.sum().item()}")

            logger.debug(f"=== КОНЕЦ ПРОВЕРКИ ТОКЕНИЗАЦИИ ===")

            # ▼▼▼ ПРОВЕРКА УСТРОЙСТВ ПЕРЕД ГЕНЕРАЦИЕЙ (БЕЗ ПЕРЕНОСА МОДЕЛИ) ▼▼▼
            # Важно: модель переносится только при загрузке; здесь выравниваем только входы
            model_device = next(self._model.parameters()).device
            inputs_device = inputs['input_ids'].device
            logger.debug(f"Проверка устройств: модель на {model_device}, входы на {inputs_device}")
            # ▲▲▲ КОНЕЦ ПРОВЕРКИ УСТРОЙСТВ ▲▲▲

            # Контрольная точка: логируем длину в токенах
            num_tokens = inputs['input_ids'].shape[1]
            logger.info(f"Длина промпта в токенах для генерации: {num_tokens}")

            # Перемещаем на нужное устройство
            inputs = self._move_to_device(inputs)

            # Гарантируем, что входы на том же устройстве, что и модель
            inputs_device = inputs['input_ids'].device
            if inputs_device != model_device:
                logger.warning(
                    f"Входы на {inputs_device}, модель на {model_device}. Перемещаем входы на устройство модели."
                )
                inputs = {
                    k: v.to(model_device) if isinstance(v, torch.Tensor) else v
                    for k, v in inputs.items()
                }

            # Настройки генерации в зависимости от устройства
            generation_kwargs = {
                **inputs,
                "max_new_tokens": max_tokens,
                "pad_token_id": self._tokenizer.pad_token_id,
                "eos_token_id": self._tokenizer.eos_token_id,
            }

            # ОПТИМИЗАЦИИ ДЛЯ РАЗНЫХ УСТРОЙСТВ
            if self._device == "cpu":
                # Для CPU используем greedy decoding для скорости
                if temp > 0.1:
                    generation_kwargs["do_sample"] = True
                    generation_kwargs["temperature"] = temp
                else:
                    generation_kwargs["do_sample"] = False
                    generation_kwargs["num_beams"] = 1
                generation_kwargs["no_repeat_ngram_size"] = 2
            elif self._device == "directml":
                # ДЛЯ DIRECTML: только greedy decoding для стабильности
                generation_kwargs["do_sample"] = False
                generation_kwargs["num_beams"] = 1
                logger.info("Используются упрощенные настройки генерации для DirectML (greedy)")
            else:
                # Для CUDA/MPS можно использовать все параметры
                generation_kwargs["do_sample"] = temp > 0
                if temp > 0:
                    generation_kwargs["temperature"] = temp
                generation_kwargs["top_p"] = self.config.top_p
                generation_kwargs["top_k"] = self.config.top_k
                generation_kwargs["repetition_penalty"] = self.config.repetition_penalty
                generation_kwargs["no_repeat_ngram_size"] = 3

            # ОТЛАДОЧНЫЙ БЛОК перед генерацией
            logger.debug(f"=== ПРОВЕРКА ПЕРЕД ГЕНЕРАЦИЕЙ ===")
            logger.debug(f"Параметры генерации: {generation_kwargs.keys()}")
            logger.debug(f"Тип inputs: {type(inputs['input_ids'])}, Устройство: {inputs['input_ids'].device}, Форма: {inputs['input_ids'].shape}")
            logger.debug(f"Тип модели: {type(self._model)}, Устройство модели: {next(self._model.parameters()).device}")

            # ОТЛАДОЧНЫЙ БЛОК
            logger.debug(f"Устройство генерации: {self._device}")
            logger.debug(f"Ключи generation_kwargs: {list(generation_kwargs.keys())}")
            if 'input_ids' in inputs:
                logger.debug(f"Тип/форма input_ids: {inputs['input_ids'].dtype}/{inputs['input_ids'].shape}")
                logger.debug(f"Устройство input_ids: {inputs['input_ids'].device}")
                # Проверка на NaN/Inf (редкая, но возможная проблема)
                if inputs['input_ids'].is_floating_point() and (
                        torch.isnan(inputs['input_ids']).any() or torch.isinf(inputs['input_ids']).any()
                ):
                    logger.error("Обнаружены NaN или Inf в input_ids!")
                    return "Ошибка: поврежденные входные данные"

            # ▼▼▼ ДОБАВИТЬ ЗДЕСЬ - ОБРАБОТКА ОШИБОК ГЕНЕРАЦИИ ▼▼▼
            try:
                # Генерация
                start_time = time.time()
                with torch.no_grad():
                    outputs = self._model.generate(**generation_kwargs)
                generation_time = time.time() - start_time

            except Exception as gen_error:
                logger.error(f"СБОЙ В model.generate(): {type(gen_error).__name__}: {gen_error}")
                # Пробуем на CPU как последнее средство (без переноса модели с DirectML)
                logger.info("Пробуем выполнить генерацию на CPU...")
                try:
                    # Создаем копии на CPU
                    inputs_cpu = {k: v.cpu() for k, v in inputs.items() if isinstance(v, torch.Tensor)}

                    if self._cpu_model is None:
                        logger.info("Загружаем CPU-модель для фолбэка (это может занять время)...")
                        cpu_kwargs = {
                            "dtype": torch.float32,
                            "trust_remote_code": self.config.trust_remote_code if "phi-3" not in self.config.model_name.lower() else False,
                            "cache_dir": str(self.config.model_cache_dir),
                            "low_cpu_mem_usage": True,
                            "device_map": None,
                        }
                        self._cpu_model = AutoModelForCausalLM.from_pretrained(
                            self.config.model_name,
                            **cpu_kwargs
                        )
                        self._cpu_model = self._cpu_model.to("cpu")
                        self._cpu_model = self._cpu_model.float()
                        self._cpu_model.eval()

                    model_cpu = self._cpu_model

                    # Обновляем generation_kwargs для CPU
                    cpu_generation_kwargs = {**generation_kwargs}
                    cpu_generation_kwargs.update(inputs_cpu)
                    cpu_generation_kwargs["max_new_tokens"] = min(cpu_generation_kwargs["max_new_tokens"], 128)

                    with torch.no_grad():
                        outputs = model_cpu.generate(**cpu_generation_kwargs)

                    generation_time = time.time() - start_time
                    logger.info("Генерация на CPU удалась!")

                except Exception as cpu_error:
                    logger.error(f"Генерация на CPU тоже не удалась: {cpu_error}")
                    return "Ошибка: не удалось сгенерировать ответ"
            # ▲▲▲ КОНЕЦ ДОБАВЛЕНИЯ ОБРАБОТКИ ОШИБОК ▲▲▲

            # ОТЛАДОЧНЫЙ БЛОК после генерации
            outputs_device = outputs.device
            outputs_cpu = outputs.detach().cpu()
            logger.debug(f"=== РЕЗУЛЬТАТЫ ГЕНЕРАЦИИ ===")
            logger.debug(f"Генерация заняла: {generation_time:.2f} секунд")
            logger.debug(
                f"Тип outputs: {type(outputs_cpu)}, Форма: {outputs_cpu.shape}, Устройство: {outputs_device}"
            )
            logger.debug(f"Dtype outputs: {outputs_cpu.dtype}")

            # Проверка на "странные" значения
            if outputs_cpu.is_floating_point() and torch.isnan(outputs_cpu).any():
                logger.error("Выход модели содержит NaN!")
            if outputs_cpu.is_floating_point() and torch.isinf(outputs_cpu).any():
                logger.error("Выход модели содержит Inf!")

            # Проверка диапазона значений (добавим для диагностики)
            min_val = outputs_cpu.min().item()
            max_val = outputs_cpu.max().item()
            logger.debug(f"Диапазон значений в outputs: min={min_val}, max={max_val}")

            # Дополнительно: проверим первые 10 сгенерированных токенов
            if outputs_cpu.shape[1] > inputs['input_ids'].shape[1]:
                generated_token_ids = outputs_cpu[0][
                    inputs['input_ids'].shape[1]:inputs['input_ids'].shape[1] + 10
                ]
                logger.debug(f"Первые 10 сгенерированных token_id: {generated_token_ids.tolist()}")

            # Декодирование с ЗАЩИТОЙ ОТ БИТЫХ ДАННЫХ
            generated_text = ""
            try:
                # Для стабильности декодируем с CPU (особенно для DirectML/MPS)
                decoded_ids = outputs_cpu[0][inputs['input_ids'].shape[1]:]
                generated_text = self._tokenizer.decode(
                    decoded_ids,
                    skip_special_tokens=True
                )
                logger.debug(f"Успешное декодирование, длина текста: {len(generated_text)} символов")

            except (UnicodeDecodeError, ValueError, RuntimeError) as decode_error:
                logger.error(f"Ошибка декодирования выхода модели: {decode_error}")
                return "Ошибка декодирования ответа"

            # Очистка ответа
            cleaned_response = self._clean_response(generated_text)

            return cleaned_response.strip()

        except Exception as e:
            logger.error(f"Ошибка при генерации ответа: {e}")
            return f"Ошибка при генерации ответа: {str(e)}"

    def _clean_response(self, text: str) -> str:
        """
        Очистка сгенерированного текста

        Args:
            text: Исходный текст

        Returns:
            str: Очищенный текст
        """
        # Удаляем повторяющиеся фразы
        lines = text.split('\n')
        seen = set()
        cleaned_lines = []

        for line in lines:
            line_stripped = line.strip()
            if line_stripped and line_stripped not in seen:
                seen.add(line_stripped)
                cleaned_lines.append(line)

        cleaned_text = '\n'.join(cleaned_lines)

        # Обрезаем, если модель начала задавать новый вопрос
        stop_phrases = ["Вопрос:", "Q:", "Запрос:", "Следующий вопрос:", "Question:", "Human:", "Assistant:"]
        for phrase in stop_phrases:
            if phrase in cleaned_text:
                cleaned_text = cleaned_text.split(phrase)[0]

        # Удаляем лишние пробелы
        cleaned_text = ' '.join(cleaned_text.split())

        return cleaned_text

    def test_generation(self) -> bool:
        """
        Тестирование генерации

        Returns:
            bool: Успешно ли прошло тестирование
        """
        try:
            test_prompt = "Ответь одним словом: Как называется столица Франции?"
            response = self.generate_response(test_prompt, max_new_tokens=10)
            logger.info(f"Тестовая генерация: {response}")
            return bool(response and len(response) > 0)
        except Exception as e:
            logger.error(f"Ошибка тестирования: {e}")
            return False