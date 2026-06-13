# Organizer Microservice Embeddings

Новые фичи приложения [тут](https://github.com/users/stanis1avs/projects/1)

[frontend](https://github.com/Stanislavsus-prj/organizer_frontend) 
[backend](https://github.com/Stanislavsus-prj/organizer_backend)

## О проекте

**Organizer** — система для организации личных данных с умным мультимодальным поиском. 

![f](https://github.com/stanis1avs/organizer_frontend/blob/main/readme_files/ui.jpg?raw=true)

### Основная функциональность:
- Сохранение текстовых сообщений
- Загрузка и хранение изображений
- Запись и хранение аудио файлов
- Запись и хранение видео файлов
- **Умный гибридный поиск** (BM25 + семантический + поиск по изображению)
- **Поиск по изображению** — загрузи картинку, найди похожие и связанные тексты
- Избранные сообщения
- Закреплённые сообщения
- **Offline-режим** — сообщения сохраняются в IndexedDB и отправляются автоматически при восстановлении соединения; Service Worker кэширует медиафайлы для просмотра без сети

### Умный поиск:
Система использует трёхуровневый гибридный поиск:
- **BM25** (OpenSearch, Russian analyzer) — точный поиск по ключевым словам
- **Dense Vectors** (Qdrant, 384D) — семантический поиск по смыслу на 50+ языках
- **CLIP ViT-B/32** (Qdrant, 512D) — кросс-модальный поиск: текст → изображение, изображение → текст + изображения
- **OCR** (Tesseract) — извлечение текста из изображений для BM25

---

## Технический стэк

### Backend
1. **WebSocket** — реальное время
2. **Koa** — веб-фреймворк
3. **Cassandra** — основная БД
4. **BM25 (OpenSearch)** — полнотекстовый поиск с Russian analyzer
5. **Dense Vectors (Qdrant)** — 4 коллекции:
   - `messages_text_vectors` (384D) — мультиязычный семантический поиск
   - `messages_clip_vectors` (512D) — CLIP-текст для поиска по картинке
   - `images_vectors` (2048D) — ResNet-50 визуальные признаки
   - `images_clip_vectors` (512D) — CLIP-изображения для текстового поиска
6. **Tesseract OCR** — распознавание текста с изображений
7. **Jest** — юнит + интеграционные тесты

### Микросервис эмбеддингов
- **FastAPI** — веб-фреймворк
- **paraphrase-multilingual-MiniLM-L12-v2** — мультиязычная модель (384D, 50+ языков)
- **CLIP ViT-B/32** — кросс-модальная модель (512D, текст и изображения в одном пространстве)
- **ResNet-50** — визуальные эмбеддинги изображений (2048D)

### Frontend
1. **Нативный JavaScript**
2. **WebSocket**
3. **Webpack**
4. **ESLint**
5. **Appveyor** — CI/CD
6. **MediaRecorder API** — запись аудио/видео
7. **Service Worker API** — offline-режим, кэш медиа
8. **IndexedDB** — очередь сообщений при обрыве связи
9. **Jest** — юнит + интеграционные тесты
10. **Puppeteer** — E2E тесты

### В разработке
- Web Crypto API (шифрование файлов)
- Redis (кэширование эмбеддингов и результатов поиска)
- Streams API (потоковая загрузка файлов)
- Chrome Extension

---
