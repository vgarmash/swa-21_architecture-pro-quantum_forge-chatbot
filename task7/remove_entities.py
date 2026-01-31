#!/usr/bin/env python3
"""
Скрипт для удаления ключевых сущностей из базы знаний
для создания искусственных пробелов в базе данных.
"""

import os
import re

def remove_entities_from_file(file_path, entities_to_remove):
    """
    Удаляет упоминания ключевых сущностей из файла.
    
    Args:
        file_path (str): Путь к файлу
        entities_to_remove (list): Список сущностей для удаления
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Сохраняем оригинальное содержимое для отладки
        original_content = content
        
        # Удаляем упоминания сущностей
        for entity in entities_to_remove:
            # Удаляем строки, содержащие сущность (с учетом регистра)
            # Используем регулярные выражения для более точного удаления
            content = re.sub(rf'\b{re.escape(entity)}\b', '', content, flags=re.IGNORECASE)
            
            # Удаляем строки, содержащие сущность, но не полностью
            # Удаляем строки, где сущность встречается в контексте
            lines = content.split('\n')
            filtered_lines = []
            for line in lines:
                if entity.lower() in line.lower():
                    # Проверяем, не является ли это просто упоминанием в заголовке или описании
                    if entity in line and not any(word in line.lower() for word in ['document:', 'article:', 'file:', 'section:']):
                        # Удаляем только упоминание сущности, но сохраняем остальную строку
                        line = re.sub(rf'\b{re.escape(entity)}\b', '', line, flags=re.IGNORECASE)
                        # Если строка стала пустой или содержит только пробелы, пропускаем её
                        if line.strip():
                            filtered_lines.append(line)
                    else:
                        filtered_lines.append(line)
                else:
                    filtered_lines.append(line)
            
            content = '\n'.join(filtered_lines)
        
        # Удаляем пустые строки и строки с пробелами
        lines = content.split('\n')
        filtered_lines = [line for line in lines if line.strip()]
        content = '\n'.join(filtered_lines)
        
        # Записываем измененный контент обратно в файл
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
            
        print(f"Обработан файл: {file_path}")
        
    except Exception as e:
        print(f"Ошибка при обработке файла {file_path}: {e}")

def remove_entities_from_knowledge_base(entities_to_remove):
    """
    Удаляет упоминания ключевых сущностей из всей базы знаний.
    
    Args:
        entities_to_remove (list): Список сущностей для удаления
    """
    knowledge_base_path = 'knowledge_base'
    
    if not os.path.exists(knowledge_base_path):
        print(f"Путь {knowledge_base_path} не существует")
        return
    
    # Получаем список всех текстовых файлов в базе знаний
    txt_files = [f for f in os.listdir(knowledge_base_path) 
                 if f.endswith('.txt') and os.path.isfile(os.path.join(knowledge_base_path, f))]
    
    print(f"Найдено {len(txt_files)} текстовых файлов в базе знаний")
    
    # Обрабатываем каждый файл
    for txt_file in txt_files:
        file_path = os.path.join(knowledge_base_path, txt_file)
        remove_entities_from_file(file_path, entities_to_remove)
    
    print("Удаление сущностей завершено")

if __name__ == "__main__":
    # Список ключевых сущностей для удаления
    entities_to_remove = ['Quenix', 'Syloos', 'Taraix']
    
    print("Удаление ключевых сущностей из базы знаний...")
    print(f"Сущности для удаления: {entities_to_remove}")
    
    remove_entities_from_knowledge_base(entities_to_remove)