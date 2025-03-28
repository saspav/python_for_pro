# Задание 1: Управление библиотекой книг
# Создайте систему управления библиотекой, которая позволяет добавлять книги, удалять книги,
# искать книги по названию и автору.

class Library:
    def __init__(self):
        # Словарь для хранения книг: ключ - название, значение - список авторов
        self.books = {}

    def add_book(self, title, authors):
        """Добавляет книгу с указанным названием и списком авторов."""
        if isinstance(authors, str):
            _authors = [authors]
        else:
            _authors = list(authors)
        if title in self.books:
            print(f"Книга '{title}' уже существует. Добавляем авторов.")
            self.books[title].extend(_authors)
            print(f"К книге '{title}' добавлен автор: {authors}.")
        else:
            self.books[title] = _authors
            print(f"Книга '{title}' добавлена.")

    def remove_book(self, title):
        """Удаляет книгу с указанным названием, если она существует."""
        if title in self.books:
            del self.books[title]
            print(f"Книга '{title}' удалена.")
        else:
            print(f"Книга '{title}' не найдена.")

    def search_by_title(self, title):
        """Ищет книгу по названию."""
        if title in self.books:
            return f"Книга '{title}' найдена. Авторы: {', '.join(self.books[title])}."
        else:
            return f"Книга '{title}' не найдена."

    def search_by_author(self, author):
        """Ищет книги по автору."""
        found_books = [title for title, authors in self.books.items() if author in authors]
        if found_books:
            return f"Книги автора '{author}': {', '.join(found_books)}."
        else:
            return f"Книги автора '{author}' не найдены."


# Пример использования:
library = Library()

# Добавляем новые книги
library.add_book("Золотой телёнок", ["Илья Ильф", "Евгений Петров"])
library.add_book("Преступление и наказание", "Фёдор Достоевский")
library.add_book("Тайный город", "Вадим Панов")
library.add_book("Доказательство силы", "Вадим Панов")
library.add_book("Доказательство силы", "Виктор Точинов")

# Ищем книги
print(library.search_by_title("Золотой телёнок"))  # Найдём книгу по названию
print(library.search_by_author("Фёдор Достоевский"))  # Найдём книги по автору
print(library.search_by_title("Доказательство силы"))  # Найдём книги по названию

# Удаляем книгу
library.remove_book("Золотой телёнок")
print(library.search_by_title("Золото́й телёнок"))  # Проверяем, что книга удалена
