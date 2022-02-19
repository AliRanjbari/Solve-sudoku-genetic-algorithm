import random
import time
import matplotlib.pyplot as plt


F_THRESH = 81
BOXES = ((0,1,2,9,10,11,18,19,20),(3,4,5,12,13,14,21,22,23),(6,7,8,15,16,17,24,25,26),
         (27,28,29,36,37,38,45,46,47),(30,31,32,39,40,41,48,49,50),(33,34,35,42,43,44,51,52,53),
         (54,55,56,63,64,65,74,75,76),(57,58,59,66,67,68,75,76,77),(60,61,62,69,70,71,78,79,80))


class Sudoku:
    
    gene_pool = [1,2,3,4,5,6,7,8,9]

    def __init__(self, initial: list):
        self.initial = initial
        self.max_fitnesses = []


    def init_population2(self, pop_number):
        population = []
        for _ in range(pop_number):
            chromosome = [random.choice(self.gene_pool) if g == 0 else g for g in self.initial]
            population.append(chromosome)
        return population

    def init_population(self, pop_number):
        population = []
        for _ in range(pop_number):
            chromosome = list(self.initial)
            for row in range(0,9):
                choices = list(self.gene_pool)
                for col in range(row*9, row*9 + 9):
                    if chromosome[col]:
                        choices.remove(chromosome[col])
                for col in range(0,9):
                    if chromosome[row*9 + col] == 0:
                        chromosome[row*9 + col] = random.choice(choices)
                        choices.remove(chromosome[row*9 + col])
            population.append(chromosome)
        return population
    
    def fitness_fn(self, x):
        fitness = 0

        for pos in range(81):
            fitness += 0 if self.all_repet(x, pos) else 1
        return fitness
    
    def all_repet(self, x, pos):
        if self.col_repet(x, pos) or self.box_repet(x, pos):
            return True
        return False
    
    def row_repet(self, x, pos):
        flag = 1
        n = x[pos]
        row = pos // 9
        for g in x[row*9: (row+1)*9]:
            if n == g:
                if flag:
                    flag -= 1
                    continue
                return True
        return False   

    def col_repet(self, x, pos):
        flag = 1
        n = x[pos]
        col = pos % 9
        for i in range(col, 81, 9):
            if n == x[i]:
                if flag:
                    flag -= 1
                    continue
                return True
        return False


    def box_repet(self, x, pos):
        flag = 1
        n = x[pos]
        for b in BOXES:
            if pos in b:
                for index in b:
                    if x[index] == n:
                        if flag:
                            flag -= 1
                            continue
                        True
        return False


    def mutate1(self, x, prob):
        if random.random() >= prob:
            return x
        
        pos = random.randrange(0, len(x))
        new_gene = random.choice(self.gene_pool)

        if self.initial[pos]:
            return x
        return  x[:pos] + [new_gene] + x[pos+1:]

    def mutate(self, x, prob):
        if random.random() >= prob:
            return x
        
        row = random.randint(0,8)
        pos1 = row*9 + random.randint(0,8)
        pos2 = row*9 + random.randint(0,8)
        while pos1 == pos2 or self.initial[pos1] or self.initial[pos2]:
            pos1 = row*9 + random.randint(0,8)
            pos2 = row*9 + random.randint(0,8)

        temp = x[pos1]
        x[pos1] = x[pos2]
        x[pos2] = temp
        return x




    def crossover2(self, x, y):
        offspring = list(x)

        boxes = random.choices([0,1,2,3,4,5,6,7,8], k=random.randint(1, 9))
        for box in boxes:
            for i in BOXES[box]:
                offspring[i] = y[i]

        return offspring

    def crossover3(self, x, y):
        offspring = list(x)
        for i, v in enumerate(offspring):
            if random.choice([True, False]):
                offspring[i] = y[i]

        return offspring


    def crossover(self, x, y):
        crossover_point1 = random.randint(0,8)
        crossover_point2 = random.randint(1,9)
        while(crossover_point1 == crossover_point2):
            crossover_point1 = random.randint(0,8)
            crossover_point2 = random.randint(1,9)

        if(crossover_point1 > crossover_point2):
                temp = crossover_point1
                crossover_point1 = crossover_point2
                crossover_point2 = temp

        p1 = crossover_point1 * 9
        p2 = crossover_point2 * 9

        return list(x[:p1]+y[p1:p2]+x[p2:])


    def select(self, r, population):
        sel = set()
        for i in range(r):
            founded = True
            while(founded):
                rand = random.randint(1, self.totals[-1])
                for j, v in enumerate(self.totals):
                    if rand < v:
                        sel.add(tuple(population[j]))
                        if len(sel) == i + 1:
                            founded = False
                        break


        return sel

    
    def update_total_fitness(self, population):
        self.totals = []
        fitnesses = map(self.fitness_fn, population)
        self.max_fitness = 0
        for i, w in enumerate(fitnesses):
            if w > self.max_fitness:
                self.max_fitness = w
                self.max_index = i
            self.totals.append(w + self.totals[-1] if self.totals else w)
        self.max_fitnesses.append(self.max_fitness)


    def solve(self, pop_number, f_thresh=F_THRESH, ngen=1000, pmut=0.1):
        population = self.init_population(pop_number)
        for i in range(ngen):
            self.update_total_fitness(population)
            population = [self.mutate(self.crossover(*self.select(2, population)), pmut)
                          for _ in range(pop_number)]

            print(i, self.max_fitness)
            if f_thresh:
                if self.max_fitness == f_thresh:
                    plt.plot(self.max_fitnesses)
                    plt.show()
                    break

        return max(population, key=self.fitness_fn)


    def display(self, state):
        for i, v in enumerate(state):
            if i % 9:
                print(v, end=' ')
            else:
                print('\n', v, end=' ')
        print()


if __name__ == "__main__":
    sudoku = Sudoku(initial=[8, 0, 6, 0, 0, 0, 1, 0, 7,
                             0, 0, 0, 6, 0, 2, 0, 0, 0,
                             0, 5, 3, 0, 0, 4, 8, 0, 6,
                             7, 0, 4, 8, 0, 0, 6, 3, 0,
                             0, 0, 0, 0, 0, 0, 0, 9, 0,
                             1, 0, 0, 5, 0, 0, 4, 0, 0,
                             0, 0, 1, 2, 0, 0, 7, 0, 9,
                             2, 0, 0, 0, 9, 6, 0, 0, 0,
                             0, 7, 0, 0, 1, 0, 0, 8, 0,
                             ])    
    
    answer = sudoku.solve(800,ngen=10000, pmut=0.17)
    sudoku.display(answer)