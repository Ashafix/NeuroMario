class Evolution:
    def __init__(self):
        self.solution = list()
        self.generations = dict()

    def mate_solutions(self, sol1, sol2, prob=0.5):
        keys = list(sol1.keys())
        keys.extend(list(sol2.keys()))
        child_solution = dict()
        for key in keys:
            if key in sol1 and key in sol2:
                if random.random() < prob:
                    child_solution[key] = sol1[key]
                else:
                    child_solution[key] = sol2[key]
            elif key in sol1:
                child_solution[key] = sol1[key]
            else:
                child_solution[key] = sol2[key]
        return child_solution

    def trim_genome(self, genome):
        output_genome = dict()
        for gene in genome:
            output_genome[gene] = genome[gene][0]
        return output_genome

    def mutate_solution(self, solution, prob=0.1, allowed=b'|..|UDLRs.YBXAlr|............|'):
        mutated_solution = dict()
        for key in solution:

            if random.random() < prob:
                print()
                print('mutate', key)
                pos = random.randint(4, 15)
                while pos in (9,):
                    pos = random.randint(4, 15)
                print(solution[key][pos:pos + 1], allowed[pos:pos + 1], pos)
                print('before', solution[key])

                if solution[key][pos:pos + 1] == allowed[pos:pos + 1]:
                    print('silencing')
                    mutated_solution[key] = solution[key][0:pos] + b'.' + solution[key][pos + 1:]
                else:
                    print('gain of function')
                    mutated_solution[key] = solution[key][0:pos] + allowed[pos:pos + 1] + solution[key][pos + 1:]
                print('after ', mutated_solution[key])
            else:
                mutated_solution[key] = solution[key]
        return mutated_solution

    def cross_solutions(self, sol1, sol2, mutate=0.1):
        new_sol = self.mate_solutions(sol1, sol2)
        if mutate > 0:
            new_sol = self.mutate_solution(new_sol, prob=mutate)

        return new_sol


class Organism:
    def __init__(self, name, sequence, run_times=list()):
        self.name = name
        self.sequence = sequence
        self.run_times = run_times


class Generation:
    def __init__(self):
        self.parents = dict()
        self.children = dict()
        self.performance = dict()
        self.statistics = dict()
        self.generation_number = 0
        self.name = ''
        self.similarity = dict()

    def add_parent(self, parent):
        if parent.name in self.parents:
            return False
        self.parents[parent.name] = parent
        return True

    def add_child(self, parent):
        if children.name in self.children:
            return False
        self.children[children.name] = children
        return True

    def get_child_statistics(self):
        if not self.statistics.get('children'):
            self.calculate_statistics()
        return self.statistics['children']

    def get_parent_statistics(self):
        if not self.statistics.get('parents') or not self.statistics.get('children'):
            self.calculate_statistics()
        return self.statistics['parents']

    def calculate_statistics(self):

        self.statistics['children'] = dict()
        self.statistics['parents'] = dict()

        for child_name, child in self.children.items():
            self.statistics['children'][child_name] = dict()
            self.statistics['children'][child_name]['run_times'] = child.run_times
            self.statistics['children'][child_name]['avg_run_times'] = statistics.mean(child.run_times)
            self.statistics['children'][child_name]['med_run_times'] = statistics.median(child.run_times)

        for parent_name, parent in self.parents.items():
            self.statistics['parents'][parent_name] = dict()
            self.statistics['parents'][parent_name]['run_times'] = parent.run_times
            self.statistics['parents'][parent_name]['avg_run_times'] = statistics.mean(parent.run_times)
            self.statistics['parents'][parent_name]['med_run_times'] = statistics.median(parent.run_times)

        self.statistics['summary']['avg_run_times_children'] = statistics.mean(
            self.statistics['children'][children_name]['avg_run_times'] for children_name in self.children)
        self.statistics['summary']['avg_run_times_parents'] = statistics.mean(
            self.statistics['parents'][parent_name]['avg_run_times'] for parent_name in self.parents)
        self.statistics['summary']['med_run_times_children'] = statistics.median(
            self.statistics['children'][children_name]['avg_run_times'] for children_name in self.children)
        self.statistics['summary']['med_run_times_parents'] = statistics.median(
            self.statistics['parents'][parent_name]['avg_run_times'] for parent_name in self.parents)

        self.statistics['comparison'][child_name]['avg_run_times'] = self.statistics['summary'][
                                                                         'avg_run_times_children'] / \
                                                                     self.statistics['summary']['avg_run_times_parents']
        self.statistics['comparison'][child_name]['med_run_times'] = self.statistics['summary'][
                                                                         'med_run_times_children'] / \
                                                                     self.statistics['summary']['med_run_times_parents']

    def calculate_similarities(self):

        for parent_name, parent in self.parents.items():
            for parent2_name, parent2 in self.parents.items():
                if parent_name == parent2_name:
                    self.similarity[parent_name] = dict()
                    self.similarity[parent2_name] = dict()
                    self.similarity[parent_name][parent2_name] = dict(homology=0, new_genes=0)
                    self.similarity[parent2_name][parent_name] = dict(homology=0, new_genes=0)
                self.calculate_similarity(parent, parent2)

    def calculate_diff(self, sequence1, sequence2):
        diff = 0
        if len(sequence1) != len(sequence2):
            return 1
        for i, s in enumerate(sequence1):
            if s != sequence2[i]:
                diff += 1
        diff /= len(sequence1)
        return diff

    def calculate_similarity(self, gene1, gene2):

        homology = 0
        new_genes = 0
        for gene in gene1.sequence:
            if gene in gene2.sequence:
                homology += self.calculate_diff(gene1.sequence[gene], gene2.sequence[gene])
            else:
                new_genes += 1
        for gene in gene2.sequence:
            if gene not in gene1.sequence:
                new_genes += 1

        if not self.similarity.get(gene1.name):
            self.similarity[gene1.name] = dict()

        self.similarity[gene1.name][gene2.name] = dict(homology=homology, new_genes=new_genes)

        if not self.similarity.get(gene2.name):
            self.similarity[gene2.name] = dict()

        self.similarity[gene2.name][gene1.name] = dict(homology=homology, new_genes=new_genes)
