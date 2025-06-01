import random
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# Функція Растригіна, яка використовується для оцінки придатності особин
def rastrigin(X, Y):
    A = 10
    return A*2 + (X**2 - A * np.cos(2 * np.pi * X)) + (Y**2 - A * np.cos(2 * np.pi * Y))

# Клас Individual представляє одну особину в популяції
class Individual:
    def __init__(self, genome):
        self.genome = np.array(genome) # Геном особини (позиція в просторі)
        self.fitness = None # Значення придатності особини

# Клас MultimodalGA реалізує логіку мультимодального генетичного алгоритму
class MultimodalGA:
    def __init__(self, pop_size, bounds, max_iter, niche_radius,
                 crossover_rate=0.8, mutation_rate=0.1):
        self.pop_size = pop_size # Розмір популяції
        self.bounds = bounds # Межі простору пошуку (для X та Y)
        self.max_iter = max_iter # Максимальна кількість ітерацій
        self.niche_radius = niche_radius # Радіус ніші для механізму спільного використання
        self.crossover_rate = crossover_rate # Ймовірність кросоверу
        self.mutation_rate = mutation_rate # Ймовірність мутації
        self.population = [] # Поточна популяція особин
        self.history = [] # Історія геномів популяції на кожній ітерації

    # Створення початкової популяції випадкових особин
    def initialize(self):
        self.population = [
            Individual([random.uniform(lb, ub) for lb, ub in self.bounds])
            for _ in range(self.pop_size)
        ]

    # Обчислення значення придатності для особини (fitness evaluation)
    # Значення функції Растригіна заперечується, оскільки ГА максимізує придатність, а ми хочемо знайти мінімум функції
    def evaluate(self, ind):
        x, y = ind.genome
        ind.fitness = -(10*2 + x**2 - 10*math.cos(2*math.pi*x) +
                        y**2 - 10*math.cos(2*math.pi*y))

    # Застосування оцінки придатності до всіх особин у популяції.
    def evaluate_all(self):
        for ind in self.population:
            self.evaluate(ind)

    # Зменшує придатність особин у щільних регіонах, сприяючи пошуку кількох оптимумів (sharing)
    def share(self):
        for ind_i in self.population:
            sharing_sum = 0
            for ind_j in self.population:
                dist = np.linalg.norm(ind_i.genome - ind_j.genome) # Відстань між двома особинами
                if dist < self.niche_radius:
                    # Функція спільного використання: 1 - (відстань / радіус ніші)
                    sharing_sum += 1 - dist/self.niche_radius
            if sharing_sum > 0:
                ind_i.fitness /= sharing_sum # Ділимо придатність на суму спільного використання

    # Вибір особин для формування нащадків (турнірний відбір)
    def select(self):
        parents = []
        for _ in range(self.pop_size):
            a, b = random.sample(self.population, 2) # Вибираємо двох випадкових особин
            parents.append(a if a.fitness > b.fitness else b) # Краща особина вибирається як батько
        return parents

    # 4) Об'єднання геномів двох батьків для створення нових особин (кросовер)
    def crossover(self, p1, p2):
        if random.random() > self.crossover_rate:
            # Якщо кросовер не відбувається, повертаємо копії батьківських геномів
            return Individual(p1.genome.copy()), Individual(p2.genome.copy())
        point = 1 # Точка кросоверу (одноточковий кросовер)
        c1_genome = np.concatenate([p1.genome[:point], p2.genome[point:]])
        c2_genome = np.concatenate([p2.genome[:point], p1.genome[point:]])
        return Individual(c1_genome), Individual(c2_genome)

    # Випадкова зміна геному особини (мутація)
    def mutate(self, ind):
        for i in range(len(ind.genome)):
            if random.random() < self.mutation_rate:
                lb, ub = self.bounds[i]
                # Додаємо випадкове значення з нормального розподілу до компонента геному
                ind.genome[i] += random.gauss(0, (ub - lb)*0.05)
                # Обмежуємо значення геному в межах простору пошуку
                ind.genome[i] = np.clip(ind.genome[i], lb, ub)

    # Виконання одного покоління генетичного алгоритму
    def step(self):
        # Застосування спільного використання придатності
        self.share()

        # Вибір батьків
        parents = self.select()

        # Створення нащадків
        offspring = []
        # Перемішування батьків, щоб уникнути повторюваних пар для кросоверу
        shuffled_parents = random.sample(parents, len(parents))
        for i in range(0, self.pop_size, 2):
            p1 = shuffled_parents[i]
            p2 = shuffled_parents[i+1 if i+1 < self.pop_size else 0]
            c1, c2 = self.crossover(p1, p2) # Виконуємо кросовер
            self.mutate(c1) # Мутуємо першого нащадка
            self.mutate(c2) # Мутуємо другого нащадка
            offspring.extend([c1, c2]) # Додаємо нащадків до списку

        # Оцінка придатності нової популяції
        for ind in offspring:
            self.evaluate(ind)

        # Об'єднання поточної популяції та нащадків, сортування за придатністю та вибір найкращих
        combined = self.population + offspring
        combined.sort(key=lambda ind: ind.fitness, reverse=True) # Сортуємо за спаданням придатності
        self.population = combined[:self.pop_size] # Вибираємо найкращих особин для наступного покоління

        # Зберігання копії геномів для історії, щоб уникнути проблем з посиланням
        self.history.append(np.array([ind.genome.copy() for ind in self.population]))

    # Запуск генетичного алгоритму
    def run(self):
        self.initialize() # Ініціалізація популяції
        self.evaluate_all() # Оцінка придатності початкової популяції
        self.history.append(np.array([ind.genome.copy() for ind in self.population])) # Зберігаємо початковий стан

        # Повторення кроків до досягнення максимальної кількості ітерацій
        for _ in range(self.max_iter):
            self.step()
        # Повернення історії популяції
        return self.history

# Налаштування та запуск GA
bounds = [(-5.12, 5.12), (-5.12, 5.12)] # Межі простору пошуку для функції Растригіна
ga = MultimodalGA(pop_size=150, bounds=bounds, max_iter=60, niche_radius=0.8,
                  crossover_rate=0.8, mutation_rate=0.05)
history = ga.run() # Запускаємо алгоритм і отримуємо історію

# Створюємо сітку точок для побудови контурного графіку функції Растригіна
x = np.linspace(bounds[0][0], bounds[0][1], 200)
y = np.linspace(bounds[1][0], bounds[1][1], 200)
X, Y = np.meshgrid(x, y)
Z = rastrigin(X, Y) # Обчислюємо значення функції для кожної точки сітки

# Створення анімації
fig, ax = plt.subplots() # Створюємо фігуру та осі для графіку
ax.contour(X, Y, Z, levels=50, cmap='viridis', alpha=0.5) # Малюємо контурний графік функції
scat = ax.scatter([], [], s=40, c='red', edgecolors='black', alpha=0.7) # Створюємо об'єкт розсіювання для особин
ax.set_xlim(bounds[0]) # Встановлюємо межі осі X
ax.set_ylim(bounds[1]) # Встановлюємо межі осі Y
ax.set_xlabel('X-координата') # Мітка осі X
ax.set_ylabel('Y-координата') # Мітка осі Y
# Створюємо текстовий об'єкт для відображення номера ітерації та найкращої придатності
title = ax.text(0.5, 1.05, '', transform=ax.transAxes, ha='center', fontsize=12)

# Функція animate оновлює дані для кожного кадру анімації
# Вона приймає індекс кадру (i) і повертає об'єкти, які були змінені
def animate(i):
    current_genomes = history[i] # Отримуємо геноми популяції для поточної ітерації
    scat.set_offsets(current_genomes) # Оновлюємо позиції точок на графіку

    # Перерахунок найкращої придатності для поточного кадру
    current_fitnesses = []
    for genome in current_genomes:
        x, y = genome
        # Використовуємо ту саму логіку обчислення придатності, що й у методі evaluate
        fitness_val = -(10*2 + x**2 - 10*math.cos(2*math.pi*x) + y**2 - 10*math.cos(2*math.pi*y))
        current_fitnesses.append(fitness_val)

    # Максимальна придатність (оскільки ми її заперечили для мінімізації)
    best_fitness_val = max(current_fitnesses)
    # Знову заперечуємо, щоб показати фактичне мінімальне значення функції Растригіна
    true_min_rastrigin = -best_fitness_val

    # Оновлюємо текст заголовка з номером ітерації та найкращим значенням функції
    title.set_text(f'Ітерація {i} | Найкраще значення Растригіна: {true_min_rastrigin:.4f}')
    # Важливо: повертаємо кортеж з усіма оновленими об'єктами для blit=True
    return scat, title

# Створюємо об'єкт анімації FuncAnimation
# fig: фігура, на якій створюється анімація
# animate: функція, яка викликається для кожного кадру
# frames: кількість кадрів (довжина історії популяції)
# interval: затримка між кадрами в мілісекундах
# blit=True: оптимізація для швидшого рендерингу (оновлює лише змінені частини)
anim = animation.FuncAnimation(
    fig, animate, frames=len(history), interval=150, blit=True
)

import matplotlib
matplotlib.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe'

# Збереження анімації у MP4
writer = animation.FFMpegWriter(fps=15)
anim.save('ga_animation.mp4', writer=writer)

print("Анімацію збережено у файлі ga_animation.mp4")
