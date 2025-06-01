import random
import math
import numpy as np
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Завантажуємо дані Iris
iris = datasets.load_iris()
X_data = iris.data
y_data = iris.target

# Класи для індивідууму та MGA

class Individual:
    """
    Клас Individual моделює одного кандидата (гіперпараметри SVM).
    genome = [x, y], де:
      x = log10(C)
      y = log10(gamma)
    fitness = середня точність крос-валідації
    """
    def __init__(self, genome):
        self.genome = np.array(genome)  # [x, y]
        self.fitness = None             # буде зберігатися точність (accuracy)


class MultimodalGA:
    def __init__(self,
                 pop_size: int,
                 bounds: list[tuple[float, float]],
                 max_iter: int,
                 niche_radius: float,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1
                 ):
        """
        pop_size      -- кількість особин у популяції
        bounds        -- [(x_min, x_max), (y_min, y_max)] для [log10(C), log10(gamma)]
        max_iter      -- кількість поколінь
        niche_radius  -- радіус "ніші" (в одиницях log-простору)
        crossover_rate-- ймовірність кросоверу
        mutation_rate -- ймовірність мутації (для кожного гену)
        """
        self.pop_size = pop_size
        self.bounds = bounds
        self.max_iter = max_iter
        self.niche_radius = niche_radius
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

        # Поточна популяція (список Individual)
        self.population: list[Individual] = []
        # Історія генотипів (масив shape=(pop_size, 2)) для кожного покоління
        self.history: list[np.ndarray] = []

        # Масиви для метрик: best_fitness і avg_fitness за ітераціями
        self.best_per_iter: list[float] = []
        self.avg_per_iter: list[float] = []


    # Ініціалізація популяції
    def initialize(self):
        self.population = []
        for _ in range(self.pop_size):
            genome = [random.uniform(self.bounds[i][0], self.bounds[i][1]) for i in range(2)]
            self.population.append(Individual(genome))


    # Оцінка придатності однієї особини
    def evaluate_fitness(self, ind: Individual):
        x, y = ind.genome
        C_val = 10 ** x
        gamma_val = 10 ** y

        model = SVC(C=C_val, gamma=gamma_val, kernel='rbf', random_state=42)
        scores = cross_val_score(model, X_data, y_data, cv=3, scoring='accuracy')
        ind.fitness = float(np.mean(scores))


    # Оцінка придатності всієї популяції
    def evaluate_population(self):
        for individual in self.population:
            self.evaluate_fitness(individual)


    # Fitness Sharing (поділ придатності)
    def fitness_sharing(self):
        """
        Для кожної пари особин i, j:
          dist = евклідова відстань(genome_i, genome_j) у log-просторі.
        Якщо dist < niche_radius, сума засіюється (1 - dist/niche_radius).
        Після підрахунку sharing_sum ділимо fitness_i на sharing_sum.
        """
        for i, ind_i in enumerate(self.population):
            sharing_sum = 0.0
            for j, ind_j in enumerate(self.population):
                if i == j:
                    continue
                dist = np.linalg.norm(ind_i.genome - ind_j.genome)
                if dist < self.niche_radius:
                    sharing_sum += 1 - (dist / self.niche_radius)
            if sharing_sum > 0:
                ind_i.fitness /= sharing_sum


    # Відбір (Tournament Selection)
    def select_parents(self) -> list[Individual]:
        parents = []
        for _ in range(self.pop_size):
            a, b = random.sample(self.population, 2)
            winner = a if a.fitness > b.fitness else b
            parents.append(winner)
        return parents


    # Кросовер (одноточковий)
    def crossover(self, p1: Individual, p2: Individual) -> tuple[Individual, Individual]:
        # Якщо random > crossover_rate — повертаємо копії батьків
        if random.random() > self.crossover_rate:
            return Individual(p1.genome.copy()), Individual(p2.genome.copy())

        # Одноточковий кросовер (для genome довжини 2 "точка" = 1)
        point = 1
        c1_genome = np.concatenate([p1.genome[:point], p2.genome[point:]])
        c2_genome = np.concatenate([p2.genome[:point], p1.genome[point:]])
        return Individual(c1_genome), Individual(c2_genome)


    # Мутація
    def mutate(self, ind: Individual):
        for idx in range(2):
            if random.random() < self.mutation_rate:
                lb, ub = self.bounds[idx]
                sigma = (ub - lb) * 0.05  # 5% від діапазону
                ind.genome[idx] += random.gauss(0, sigma)
                ind.genome[idx] = np.clip(ind.genome[idx], lb, ub)


    # Створення нового покоління (нащадки)
    def create_offspring(self, parents: list[Individual]) -> list[Individual]:
        offspring = []
        random.shuffle(parents)
        for i in range(0, self.pop_size, 2):
            p1 = parents[i]
            p2 = parents[i + 1 if i + 1 < self.pop_size else 0]
            c1, c2 = self.crossover(p1, p2)
            # Мутуємо обох дітей
            self.mutate(c1)
            self.mutate(c2)
            offspring.extend([c1, c2])
        # Якщо отримали зайву особину (через непарність), обрізаємо до pop_size
        return offspring[: self.pop_size]


    # Заміна популяції
    def replace_population(self, offspring: list[Individual]):
        combined = self.population + offspring
        combined.sort(key=lambda ind: ind.fitness, reverse=True)
        self.population = combined[: self.pop_size]


    # Умова зупинки
    def termination_condition(self, iteration: int) -> bool:
        return iteration >= self.max_iter


    # Головний цикл запуску MGA
    def run(self) -> list[np.ndarray]:
        """
        1) Ініціалізація
        2) Оцінка початкової популяції
        3) Для кожного покоління:
            - застосування fitness_sharing
            - select_parents → create_offspring → evaluate_fitness(нащадки)
            - replace_population
            - збір статистики: best та avg fitness
            - збереження поточних генів у history
        4) Повернути history (список масивів shape=(pop_size,2))
        """
        # Ініціалізація
        self.initialize()
        # Оцінка початкової популяції
        self.evaluate_population()

        # Перший вимір метрик (ітерація 0)
        fitnesses = [ind.fitness for ind in self.population]
        best0 = max(fitnesses)
        avg0 = float(np.mean(fitnesses))
        self.best_per_iter.append(best0)
        self.avg_per_iter.append(avg0)
        print(f"Iter  0 | Best fitness = {best0:.4f} | Avg fitness = {avg0:.4f}")

        # Зберігаємо початковий генотип
        self.history.append(np.array([ind.genome.copy() for ind in self.population]))

        iteration = 0
        while True:
            iteration += 1

            # Fitness sharing
            self.fitness_sharing()

            # Відбір батьків
            parents = self.select_parents()

            # Створення нащадків
            offspring = self.create_offspring(parents)

            # Оцінка придатності нащадків
            for child in offspring:
                self.evaluate_fitness(child)

            # Заміна популяції
            self.replace_population(offspring)

            # Збір статистики (best / avg)
            fitnesses = [ind.fitness for ind in self.population]
            best_f = max(fitnesses)
            avg_f = float(np.mean(fitnesses))
            self.best_per_iter.append(best_f)
            self.avg_per_iter.append(avg_f)
            print(f"Iter {iteration:2d} | Best fitness = {best_f:.4f} | Avg fitness = {avg_f:.4f}")

            # Зберігаємо поточні генетичні координати
            self.history.append(np.array([ind.genome.copy() for ind in self.population]))

            # Перевірка умови зупинки
            if self.termination_condition(iteration):
                break

        return self.history


# Параметри MGA та запуск (hyperparameter optimization)
if __name__ == "__main__":
    bounds = [(-3.0, 3.0), (-4.0, 1.0)]

    pop_size = 100
    max_iter = 50
    niche_radius = 0.5
    crossover_rate = 0.8
    mutation_rate = 0.1

    # Створюємо та запускаємо MGA
    ga = MultimodalGA(
        pop_size=pop_size,
        bounds=bounds,
        max_iter=max_iter,
        niche_radius=niche_radius,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate
    )

    history = ga.run()  # Повертає список масивів shape=(pop_size,2) для кожного покоління

    # Оновимо "чисті" fitness фінальної популяції (без sharing)
    final_inds = ga.population.copy()
    for ind in final_inds:
        ga.evaluate_fitness(ind)
    final_inds.sort(key=lambda i: i.fitness, reverse=True)

    # Виведення результатів
    print("\nТОП-10 гіперпараметрів (C, gamma) та їх accuracy")
    for rank, ind in enumerate(final_inds[:10], start=1):
        x, y = ind.genome
        C_val = 10 ** x
        gamma_val = 10 ** y
        print(f"{rank:2d}. C={C_val:.4e}, gamma={gamma_val:.4e} → Accuracy={ind.fitness:.4f}")

    # Графік динаміки Best/Avg fitness по ітераціях
    plt.figure(figsize=(8, 5))
    iters = list(range(0, max_iter + 1))
    plt.plot(iters, ga.best_per_iter, label="Best fitness", linewidth=2, marker='o')
    plt.plot(iters, ga.avg_per_iter, label="Avg fitness", linewidth=2, marker='s')
    plt.xlabel("Iteration")
    plt.ylabel("Fitness (accuracy)")
    plt.title("Динаміка Best / Avg fitness по ітераціях")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Розподіл фінальних особин у просторі [log10(C), log10(γ)]
    xs = [ind.genome[0] for ind in final_inds]
    ys = [ind.genome[1] for ind in final_inds]
    fits = [ind.fitness for ind in final_inds]

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(xs, ys, c=fits, cmap='viridis', s=50, edgecolors='k')
    plt.colorbar(sc, label="Accuracy")
    plt.title("Фінальні особини в просторі [log10(C), log10(γ)]")
    plt.xlabel("log10(C)")
    plt.ylabel("log10(γ)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
