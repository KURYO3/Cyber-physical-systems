import math
import random
import numpy as np
import pytest

from main import Individual, MultimodalGA, rastrigin

# Фіксуємо seed для repeatability
@pytest.fixture(autouse=True)
def fix_random_seed():
    random.seed(42)
    np.random.seed(42)
    yield
    # після тестів можна нічого не скидати

# initialize()
def test_initialize_population_size_and_bounds():
    pop_size = 10
    bounds = [(-5.0, 5.0), (-2.0, 2.0)]
    ga = MultimodalGA(pop_size=pop_size, bounds=bounds, max_iter=1, niche_radius=1.0)

    ga.initialize()
    # Має бути pop_size індивідуумів
    assert len(ga.population) == pop_size

    # Кожний ген (x, y) повинен бути в межах заданих bounds
    for ind in ga.population:
        assert isinstance(ind, Individual)
        assert len(ind.genome) == 2
        x, y = ind.genome
        assert bounds[0][0] <= x <= bounds[0][1]
        assert bounds[1][0] <= y <= bounds[1][1]
        # До виклику evaluate() fitness має бути None
        assert ind.fitness is None

# rastrigin() та evaluate()
def test_rastrigin_known_values():
    """
    За вашою реалізацією:
      rastrigin(0,0) = 0
      rastrigin(1,1) = 20 + (1 - 10*1) + (1 - 10*1) = 0 + (-9) + (-9) = 2
    Перевіримо:
    """
    # Для (0,0): повинно бути 0
    assert math.isclose(rastrigin(0.0, 0.0), 0.0, rel_tol=1e-9)

    # Для (1,1): A=10 → 2*10 + (1^2 - 10*cos(2π*1)) + (1^2 - 10*cos(2π*1))
    #             = 20 + (1 - 10*1) + (1 - 10*1) = 20 - 9 - 9 = 2
    val_11 = rastrigin(1.0, 1.0)
    assert math.isclose(val_11, 2.0, rel_tol=1e-9)

def test_evaluate_single_individual_known_value():
    """
    Перевіряємо, що evaluate(ind) дає ind.fitness = -rastrigin(x,y).
    Тому для (0,0) → fitness = -0.0, для інших — відповідно.
    """
    ind = Individual([0.0, 0.0])
    ga = MultimodalGA(pop_size=1, bounds=[(-5,5), (-5,5)], max_iter=1, niche_radius=1.0)
    ga.evaluate(ind)
    # rastrigin(0,0)=0 → fitness = -0 = 0 (може бути -0.0 у np.float64)
    assert math.isclose(ind.fitness, 0.0, rel_tol=1e-9)

    # Для прикладу (1,2)
    val_r = rastrigin(1.0, 2.0)
    ind2 = Individual([1.0, 2.0])
    ga.evaluate(ind2)
    assert math.isclose(ind2.fitness, -val_r, rel_tol=1e-9)

def test_evaluate_all_population_changes_fitness():
    """
    Перевіряємо, що evaluate_all() викликає evaluate() для кожної особини,
    і що після нього у жодного fitness не None.
    """
    pop_size = 5
    bounds = [(-5,5), (-5,5)]
    ga = MultimodalGA(pop_size=pop_size, bounds=bounds, max_iter=1, niche_radius=1.0)
    ga.initialize()

    # Перед evaluate_all усі fitness мають бути None
    for ind in ga.population:
        assert ind.fitness is None

    ga.evaluate_all()

    # Після evaluate_all — жоден fitness не має бути None, це має бути float
    for ind in ga.population:
        assert ind.fitness is not None
        assert isinstance(ind.fitness, float)

# select() (турнірний відбір)
def test_select_prefers_higher_fitness():
    """
    Створимо 4 індивідуума:
      - 2 з fitness=100
      - 2 з fitness=10
    Турнірний відбір має повертати із цих 4 елементів 4 “батьків”,
    але в більшості вони мають бути ті, що із вищим fitness.
    """
    ind_high = Individual([0.0, 0.0]); ind_high.fitness = 100.0
    ind_low = Individual([0.0, 0.0]); ind_low.fitness = 10.0

    ga = MultimodalGA(pop_size=4, bounds=[(-1,1), (-1,1)], max_iter=1, niche_radius=1.0)
    # Створимо “популяцію” із двох high і двох low
    ga.population = [ind_high, ind_low, ind_high, ind_low]

    parents = ga.select()
    assert len(parents) == 4
    # Перевіримо, що серед обраних є мінімум половина з high.fitness
    count_high = sum(1 for p in parents if math.isclose(p.fitness, 100.0))
    assert count_high >= 2

def test_select_tournament_extreme_case_all_equal_fitness():
    """
    Якщо у всіх особин однаковий fitness, select() просто вибирає випадково,
    але вони мають бути з існуючої популяції.
    """
    pop_size = 6
    ga = MultimodalGA(pop_size=pop_size, bounds=[(-1,1), (-1,1)], max_iter=1, niche_radius=1.0)
    ga.population = [Individual([0.0, 0.0]) for _ in range(pop_size)]
    for ind in ga.population:
        ind.fitness = 50.0

    parents = ga.select()
    assert len(parents) == pop_size
    for p in parents:
        assert p in ga.population

# crossover()
def test_crossover_no_crossover_case():
    """
    Якщо crossover_rate=0.0, маємо отримати копії батьківських геномів.
    """
    p1 = Individual([1.0, 2.0])
    p2 = Individual([3.0, 4.0])
    ga = MultimodalGA(pop_size=2, bounds=[(-10,10),(-10,10)], max_iter=1, niche_radius=1.0,
                      crossover_rate=0.0, mutation_rate=0.0)

    c1, c2 = ga.crossover(p1, p2)
    assert np.allclose(c1.genome, p1.genome)
    assert np.allclose(c2.genome, p2.genome)

def test_crossover_one_point_actual_swap():
    """
    Якщо crossover_rate=1.0, точка кросоверу=1 →
      p1=[10,20], p2=[30,40] → c1=[10,40], c2=[30,20].
    """
    p1 = Individual([10.0, 20.0])
    p2 = Individual([30.0, 40.0])
    ga = MultimodalGA(pop_size=2, bounds=[(-100,100),(-100,100)], max_iter=1, niche_radius=1.0,
                      crossover_rate=1.0, mutation_rate=0.0)

    c1, c2 = ga.crossover(p1, p2)
    assert np.allclose(c1.genome, np.array([10.0, 40.0]))
    assert np.allclose(c2.genome, np.array([30.0, 20.0]))

# mutate()
def test_mutate_within_bounds_and_change():
    """
    Якщо mutation_rate=1.0, обидва гени спробують мутувати,
    але результат має лишатися в межах bounds.
    """
    initial = np.array([0.0, 0.0])
    ind = Individual(initial.copy())
    bounds = [(-5.0, 5.0), (-2.0, 2.0)]
    ga = MultimodalGA(pop_size=1, bounds=bounds, max_iter=1, niche_radius=1.0,
                      crossover_rate=0.0, mutation_rate=1.0)

    before = ind.genome.copy()
    ga.mutate(ind)
    after = ind.genome

    # Перевіряємо, що хоча б один компонент може змінитися,
    # але в межах bounds:
    x, y = after
    assert bounds[0][0] <= x <= bounds[0][1]
    assert bounds[1][0] <= y <= bounds[1][1]
    assert isinstance(x, float) and isinstance(y, float)
    # Може трапитися, що random.gauss() дав 0,0, тому зміна не гарантується.
    # Головне — перевірити межі та тип.

def test_mutate_no_change_when_rate0():
    """
    Якщо mutation_rate=0.0, жоден ген не змінюється.
    """
    original = np.array([2.0, -3.0])
    ind = Individual(original.copy())
    ga = MultimodalGA(pop_size=1, bounds=[(-10,10),(-10,10)], max_iter=1, niche_radius=1.0,
                      crossover_rate=0.0, mutation_rate=0.0)

    before = ind.genome.copy()
    ga.mutate(ind)
    after = ind.genome
    np.testing.assert_array_equal(before, after)

# evaluate offspring (оцінка нових нащадків)
def test_evaluate_offspring_changes_fitness():
    """
    Переконаємося, що метод evaluate() змінює fitness у нової особини.
    """
    child1 = Individual([1.0, 1.0])
    child2 = Individual([2.0, 2.0])
    # До виклику evaluate() в fitness має бути None
    assert child1.fitness is None
    assert child2.fitness is None

    ga = MultimodalGA(pop_size=2, bounds=[(-5,5),(-5,5)], max_iter=1, niche_radius=1.0)
    ga.evaluate(child1)
    ga.evaluate(child2)
    assert child1.fitness is not None and isinstance(child1.fitness, float)
    assert child2.fitness is not None and isinstance(child2.fitness, float)

# Тест умови зупинки (termination condition)
def test_termination_condition_respects_max_iter():
    # Створимо малий GA, який виконає step() лише max_iter разів
    ga = MultimodalGA(pop_size=2, bounds=[(-5,5),(-5,5)], max_iter=3, niche_radius=1.0)

    # Підмінімо методи, щоб легко відслідкувати кількість викликів
    calls = {'count': 0}
    def fake_step():
        calls['count'] += 1

    ga.initialize = lambda: None
    ga.evaluate_all = lambda: None
    ga.step = fake_step

    # Виконаємо run() – step() має викликатись рівно max_iter разів
    ga.run()
    assert calls['count'] == 3

# run()
def test_run_returns_history_and_last_population_size():
    """
    Запустимо GA з малими параметрами (pop_size=3, max_iter=2),
    перевіримо:
      - довжина history = max_iter + 1
      - кожен елемент history — масив shape=(pop_size, 2)
      - фінальна популяція має розмір pop_size
    """
    pop_size = 3
    max_iter = 2
    ga = MultimodalGA(pop_size=pop_size, bounds=[(-1,1),(-1,1)], max_iter=max_iter, niche_radius=0.5,
                      crossover_rate=0.5, mutation_rate=0.5)

    history = ga.run()
    # 8.1 length = max_iter + 1
    assert isinstance(history, list)
    assert len(history) == max_iter + 1

    # 8.2 кожен елемент — np.ndarray із shape=(pop_size,2)
    for arr in history:
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (pop_size, 2)

    # 8.3 фінальна популяція
    assert len(ga.population) == pop_size
    for ind in ga.population:
        assert isinstance(ind, Individual)
        # genome мусить бути np.array довжини 2
        assert ind.genome.shape == (2,)