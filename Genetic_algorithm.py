import tkinter as tk
from tkinter import ttk
import numpy as np
import random


class GAApplication:
    def __init__(self, main_window):

        self.main_window = main_window
        self.main_window.title("Генетический алгоритм для поиска минимума заданной функции")
        main_window.grid_columnconfigure(3, weight=1)
        main_window.grid_rowconfigure(12, weight=1)
        root.geometry("1000x450")
        tk.Label(main_window, text="Оптимизируемая функция: 100*(x2 - x1^2)^2 + (1 - x1)^2"
                 ).grid(row=0, column=0, columnspan=3, sticky="w", padx=10, pady=20)

        # Параметры алгоритма
        self.mutation_probability = tk.DoubleVar(value=15.0)
        self.num_individuals = tk.IntVar(value=20)
        self.gene_lower_limit = tk.DoubleVar(value=-1.0)
        self.gene_upper_limit = tk.DoubleVar(value=1.0)
        self.generations_limit = tk.IntVar(value=20)
        self.integer_encoding = tk.BooleanVar(value=False)
        self.modified_selection = tk.BooleanVar(value=False)
        self.save_history = tk.BooleanVar(value=False)

        # UI для управления
        self._add_parameter_input("Мутация (%)", self.mutation_probability, 1)
        self._add_parameter_input("Число особей", self.num_individuals, 2)
        self._add_parameter_input("Мин. значение гена", self.gene_lower_limit, 3)
        self._add_parameter_input("Макс. значение гена", self.gene_upper_limit, 4)
        self._add_parameter_input("Число поколений", self.generations_limit, 5)

        self._add_checkbox("Целочисленная кодировка", self.integer_encoding, 6)
        self._add_checkbox("Использовать модифицированный отбор", self.modified_selection, 7)
        self._add_checkbox("Сохранять историю таблицы (может замедлить программу)", self.save_history, 8)  # Флажок на форме


        tk.Button(main_window, text="Запустить алгоритм", command=self.genetic_algorithm
                  ).grid(row=9, column=0, columnspan=2, sticky="we", padx=10)
        tk.Button(main_window, text="Очистить", command=self.reset_data
                  ).grid(row=10, column=0, columnspan=2, sticky="we", padx=10)

        # Поля для отображения результатов
        self._add_label("Пройдено поколений: ", 11)
        self.completed_generations = tk.Label(main_window, text="0")
        self.completed_generations.grid(row=11, column=1, sticky="w", pady=10)

        self._add_label("Лучшие результаты:", 12)
        self.best_x1_label = self._add_label("x1 = ", 13)
        self.best_x2_label = self._add_label("x2 = ", 14)

        self.best_fitness_label = tk.Label(self.main_window, text="Значение функции = ")
        self.best_fitness_label.grid(row=15, column=0, columnspan=2, sticky="w", padx=10, pady=10)

        # Таблица для вывода поколений
        self.history_table = ttk.Treeview(main_window, columns=("Gen", "ID", "x1", "x2", "Fitness"), show='headings')
        self.history_table.grid(row=0, column=3, rowspan=16, padx=5, sticky="nswe")
        for col, width, text in zip(["Gen", "ID", "x1", "x2", "Fitness"], [50, 50, 100, 100, 100],
                                    ["Поколение", "№", "x1", "x2", "Приспособленность"]):
            self.history_table.column(col, width=width, anchor="center")
            self.history_table.heading(col, text=text)

        # Инициализация алгоритма
        self.current_population = None
        self.best_chromosome = None
        self.best_value = float('inf')
        self.total_generations = 0

    def _add_parameter_input(self, label_text, variable, row):
        tk.Label(self.main_window, text=label_text).grid(row=row, column=0, sticky="w", padx=10)
        tk.Entry(self.main_window, textvariable=variable).grid(row=row, column=1, sticky="we")

    def _add_checkbox(self, text, variable, row):
        tk.Checkbutton(self.main_window, text=text, variable=variable
                       ).grid(row=row, column=0, columnspan=2, sticky="w", padx=10)

    def _add_label(self, text, row):
        label = tk.Label(self.main_window, text=text)
        label.grid(row=row, column=0, columnspan=2, sticky="w", padx=10)
        return label

    @staticmethod
    def fitness_function(x1, x2): 
        return 100*(x2 - x1**2)**2 + (1 - x1)**2

    def generate_population(self, size, min_val, max_val):
        if self.integer_encoding.get():
            return np.random.randint(int(min_val), int(max_val) + 1, (size, 2))
        return np.random.uniform(min_val, max_val, (size, 2))

    def select_parents(self, population, fitness_scores):
        # турнирный отбор
        tournament_size = 5
        chosen = []
        
        for _ in range(2):
            candidates = random.sample(range(len(population)), tournament_size)
            candidates.sort(key=lambda idx: fitness_scores[idx])

            if self.modified_selection.get(): 
                # модифицированный отбор: 80% для лучших и 20% для худших
                best_parent_idx = candidates[0]
                worst_parent_idx = candidates[-1]
                
                if random.random() < 0.8:
                    chosen.append(population[best_parent_idx])
                else:
                    chosen.append(population[worst_parent_idx])  
            else:
                chosen.append(population[random.choice(candidates)])

        return np.array(chosen)

    def crossover(self, p1, p2, crossover_prob=0.9):
        if random.random() < crossover_prob:
            alpha = np.random.uniform(0, 1, p1.shape)
            offspring = p1 * alpha + p2 * (1 - alpha)
            if self.integer_encoding.get():
                return np.round(offspring).astype(int)
            return offspring
        return p1 if random.random() < 0.5 else p2

    def mutate(self, individual, mutation_rate, min_val, max_val):
        if random.random() < mutation_rate:
            gene_idx = random.randint(0, len(individual) - 1)
            if self.integer_encoding.get():
                individual[gene_idx] = random.randint(int(min_val), int(max_val))
            else:
                individual[gene_idx] += np.random.uniform(-1, 1)
                individual[gene_idx] = np.clip(individual[gene_idx], min_val, max_val)
        return individual

    def genetic_algorithm(self):
        mutation_prob = self.mutation_probability.get() / 100.0
        num_individuals = self.num_individuals.get()
        min_gene = self.gene_lower_limit.get()
        max_gene = self.gene_upper_limit.get()
        generation_count = self.generations_limit.get()

        if not self.save_history.get():
            self.history_table.delete(*self.history_table.get_children())

        if self.current_population is None or self.integer_encoding.get() != self.previous_integer_encoding:
            self.current_population = self.generate_population(num_individuals, min_gene, max_gene)
            self.previous_integer_encoding = self.integer_encoding.get()  

        self.best_value = float('inf') if self.best_value == float('inf') else self.best_value
        self.best_chromosome = None

        for gen in range(generation_count):
            fitness = [self.fitness_function(*chromosome) for chromosome in self.current_population]

            if min(fitness) < self.best_value:
                self.best_value = min(fitness)
                self.best_chromosome = self.current_population[np.argmin(fitness)]

            next_gen = []
            while len(next_gen) < num_individuals:
                parents = self.select_parents(self.current_population, fitness)
                offspring = self.mutate(self.crossover(parents[0], parents[1]), mutation_prob, min_gene, max_gene)
                next_gen.append(offspring)

            self.current_population = np.array(next_gen)
            self.total_generations += 1
            if self.save_history.get():
                self._update_results(gen)

        if not self.save_history.get():
            self._update_results(gen)

    def reset_data(self):
        self.current_population = None
        self.best_chromosome = None
        self.best_value = float('inf')
        self.total_generations = 0
        self.history_table.delete(*self.history_table.get_children())
        self._update_results(-1)

    def _update_results(self, gen):
        self.completed_generations.config(text=str(self.total_generations))

        if self.best_chromosome is not None:
            if self.integer_encoding.get():
                self.best_x1_label.config(text=f"x1 = {int(self.best_chromosome[0])}")
                self.best_x2_label.config(text=f"x2 = {int(self.best_chromosome[1])}")
                self.best_fitness_label.config(text=f"Значение функции = {int(self.best_value)}")
            else:
                self.best_x1_label.config(text=f"x1 = {self.best_chromosome[0]:.9f}")
                self.best_x2_label.config(text=f"x2 = {self.best_chromosome[1]:.9f}")
                self.best_fitness_label.config(text=f"Значение функции = {self.best_value:.9f}")

        if gen >= 0:
            for idx, chromosome in enumerate(self.current_population):
                fitness = self.fitness_function(*chromosome)

                if self.integer_encoding.get():
                    fitness = int(fitness)
                    self.history_table.insert(
                        "",
                        "0",
                        values=(self.total_generations, idx + 1, f"{int(chromosome[0])}", f"{int(chromosome[1])}", f"{fitness}")
                    )
                else:
                    self.history_table.insert(
                        "",
                        "0",
                        values=(self.total_generations, idx + 1, f"{chromosome[0]:.9f}", f"{chromosome[1]:.9f}", f"{fitness:.9f}")
                    )

root = tk.Tk()
app = GAApplication(root)
root.mainloop()