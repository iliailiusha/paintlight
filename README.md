---
title: Art of Strokes
emoji: 🎨
colorFrom: '#0a0a0a'
colorTo: '#f5dbb1'
sdk: static
pinned: false
---

# Art of Strokes

Интерактивный сайт в стиле «чёрные мазки краски / белый к жёлтому».

## Страницы

| Путь | Описание |
|------|----------|
| `/` | Главная — рисуй светом, сохраняй мазки кликом |
| `/game` | Игра — платформер с фонарём на палке |
| `/works` | Работы — галерея |
| `/about` | О нас |
| `/contact` | Контакты |

## Деплой на Hugging Face

1. Создай Space типа **Static** на huggingface.co
2. Загрузи все файлы из этой папки
3. Space автоматически раздаст статику

Или через git:

```bash
git init
git add .
git commit -m "init"
git remote add space https://huggingface.co/spaces/твой_юзер/art-of-strokes
git push --force space main
```

## Управление

- **Клик** — сохранить след от курсора навсегда
- **ПКМ по следу** — удалить след
- **A/D** — бег (в игре)
- **W** — прыжок (в игре)
- **Курсор** — направление взгляда / фонаря
