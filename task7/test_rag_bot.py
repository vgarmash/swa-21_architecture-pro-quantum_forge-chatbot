#!/usr/bin/env python3
"""
Скрипт автоматического тестирования RAG-бота на основе "золотого набора" вопросов.
"""

import json
import time
import datetime
from typing import Dict, List, Any
import os

class RAGTestRunner:
    def __init__(self, log_file_path: str = "logs.jsonl"):
        """
        Инициализирует тестировщик RAG-бота.
        
        Args:
            log_file_path (str): Путь к файлу лога
        """
        self.log_file_path = log_file_path
        # Создаем директорию, если она не существует
        os.makedirs(os.path.dirname(log_file_path) if os.path.dirname(log_file_path) else '.', exist_ok=True)
    
    def load_golden_questions(self, questions_file: str = "golden_questions.txt") -> List[Dict[str, Any]]:
        """
        Загружает "золотой набор" вопросов из файла.
        
        Args:
            questions_file (str): Путь к файлу с вопросами
            
        Returns:
            List[Dict[str, Any]]: Список вопросов с типами
        """
        questions = []
        current_section = None
        
        try:
            with open(questions_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if line.startswith('## Вопросы на известные темы'):
                    current_section = 'known'
                elif line.startswith('## Вопросы на удалённые/отсутствующие темы'):
                    current_section = 'unknown'
                elif line.startswith('1.') or line.startswith('2.') or line.startswith('3.') or \
                     line.startswith('4.') or line.startswith('5.') or line.startswith('6.') or \
                     line.startswith('7.') or line.startswith('8.') or line.startswith('9.') or \
                     line.startswith('10.'):
                    # Извлекаем текст вопроса
                    question_text = line.split('.', 1)[1].strip()
                    questions.append({
                        'query': question_text,
                        'type': current_section,
                        'expected_answer': 'known' if current_section == 'known' else 'unknown'
                    })
                    
        except Exception as e:
            print(f"Ошибка при загрузке вопросов: {e}")
            
        return questions
    
    def simulate_rag_response(self, query: str, question_type: str) -> Dict[str, Any]:
        """
        Симулирует ответ RAG-бота на запрос.
        В реальном случае здесь будет вызов реального RAG-бота.
        
        Args:
            query (str): Текст запроса
            question_type (str): Тип вопроса ('known' или 'unknown')
            
        Returns:
            Dict[str, Any]: Результаты ответа
        """
        # Симулируем работу RAG-бота
        response = {
            'query': query,
            'timestamp': datetime.datetime.now().isoformat(),
            'chunks_found': False,
            'response_length': 0,
            'successful_answer': False,
            'sources': [],
            'answer': ''
        }
        
        # Проверяем, содержит ли запрос упоминание о удаленных сущностях
        unknown_entities = ['Quenix', 'Syloos', 'Taraix']
        has_unknown_entity = any(entity.lower() in query.lower() for entity in unknown_entities)
        
        # Если вопрос касается известных тем, то ответ должен быть найден
        if question_type == 'known' or not has_unknown_entity:
            # Симулируем успешный ответ
            response['chunks_found'] = True
            response['response_length'] = 150
            response['successful_answer'] = True
            response['sources'] = ['ENCY_001.txt', 'DECR_001.txt']
            response['answer'] = f"Ответ на вопрос: {query} - информация найдена в базе знаний."
        else:
            # Симулируем неудачный ответ (так как сущности удалены)
            response['chunks_found'] = False
            response['response_length'] = 0
            response['successful_answer'] = False
            response['sources'] = []
            response['answer'] = f"Ответ на вопрос: {query} - информация не найдена в базе знаний."
        
        return response
    
    def run_tests(self):
        """
        Запускает тестирование RAG-бота на основе "золотого набора" вопросов.
        """
        print("Запуск автоматического тестирования RAG-бота...")
        
        # Загружаем вопросы
        questions = self.load_golden_questions()
        print(f"Загружено {len(questions)} вопросов")
        
        # Запускаем тесты
        for i, question in enumerate(questions, 1):
            print(f"Тестируем вопрос {i}: {question['query']}")
            
            # Симулируем ответ RAG-бота
            result = self.simulate_rag_response(question['query'], question['type'])
            
            # Записываем в лог
            self.log_request(result)
            
            # Выводим результат
            status = "УСПЕШНО" if result['successful_answer'] else "НЕУДАЧНО"
            print(f"  Результат: {status}")
            print(f"  Найдены чанки: {result['chunks_found']}")
            print(f"  Длина ответа: {result['response_length']}")
            print(f"  Источники: {', '.join(result['sources']) if result['sources'] else 'Нет'}")
            print()
            
            # Добавляем небольшую задержку между тестами
            time.sleep(0.1)
        
        print("Тестирование завершено")
    
    def log_request(self, request_data: Dict[str, Any]):
        """
        Записывает информацию о запросе в лог.
        
        Args:
            request_data (Dict[str, Any]): Данные запроса
        """
        # Записываем в файл в формате JSONL (каждая строка - отдельный JSON)
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(request_data, ensure_ascii=False) + '\n')
        
        print(f"Записан запрос в лог: {request_data.get('query', 'unknown')}")

def main():
    """
    Основная функция запуска тестирования.
    """
    # Создаем тестировщик
    test_runner = RAGTestRunner("logs.jsonl")
    
    # Запускаем тесты
    test_runner.run_tests()

if __name__ == "__main__":
    main()