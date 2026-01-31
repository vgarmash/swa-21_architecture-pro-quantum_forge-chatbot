#!/usr/bin/env python3
"""
Скрипт для логирования запросов к RAG-боту.
Записывает в лог информацию о каждом запросе с полями:
- текст запроса
- timestamp
- были ли найдены чанки
- длина ответа
- флаг успешного ответа
- найденные источники
"""

import json
import time
import datetime
from typing import Dict, Any
import os

class RequestLogger:
    def __init__(self, log_file_path: str = "logs.jsonl"):
        """
        Инициализирует логгер запросов.
        
        Args:
            log_file_path (str): Путь к файлу лога
        """
        self.log_file_path = log_file_path
        # Создаем директорию, если она не существует
        os.makedirs(os.path.dirname(log_file_path) if os.path.dirname(log_file_path) else '.', exist_ok=True)
    
    def log_request(self, request_data: Dict[str, Any]):
        """
        Записывает информацию о запросе в лог.
        
        Args:
            request_data (Dict[str, Any]): Данные запроса
        """
        # Добавляем timestamp
        request_data['timestamp'] = datetime.datetime.now().isoformat()
        
        # Записываем в файл в формате JSONL (каждая строка - отдельный JSON)
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(request_data, ensure_ascii=False) + '\n')
        
        print(f"Записан запрос в лог: {request_data.get('query', 'unknown')}")

def create_sample_log_entry():
    """
    Создает пример записи лога для демонстрации структуры.
    """
    logger = RequestLogger("sample_logs.jsonl")
    
    # Примеры записей лога
    sample_entries = [
        {
            "query": "Какие есть законы о сборе мёда?",
            "chunks_found": True,
            "response_length": 150,
            "successful_answer": True,
            "sources": ["DECR_001.txt", "ENCY_002.txt"]
        },
        {
            "query": "Что такое Crystal Essence?",
            "chunks_found": False,
            "response_length": 0,
            "successful_answer": False,
            "sources": []
        },
        {
            "query": "Какие упоминания есть о Quenix?",
            "chunks_found": False,
            "response_length": 0,
            "successful_answer": False,
            "sources": []
        }
    ]
    
    for entry in sample_entries:
        logger.log_request(entry)
    
    print("Примеры записей лога созданы")

if __name__ == "__main__":
    # Создаем пример лога для демонстрации
    create_sample_log_entry()