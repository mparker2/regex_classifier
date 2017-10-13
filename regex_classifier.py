import re
from functools import partial
from random import shuffle

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score
from sklearn.externals.joblib import Parallel, delayed


v = int(np.__version__.split('.')[1])
assert v >= 13, ('numpy version >1.13 is required')


class MemoizeArray(object):

    def __init__(self, func):
        self.func = func
        self.memory = {}

    def __call__(self, arr):
        arr_str = arr.tostring()
        try:
            return self.memory[arr_str]
        except KeyError:
            val = self.func(arr)
            self.memory[arr_str] = val
            return val


def evolve_population(pop_size, n_gen, data_generator,
                      regex_components, early_stopping,
                      **gen_kwargs):
    np.random.seed(None)
    pop = np.random.choice(
        np.arange(gen_kwargs['n_choices']),
        size=(pop_size, gen_kwargs['ind_len']),
        p=gen_kwargs['component_weight']
    )
    early_stopping = int(early_stopping)
    for gen in range(1, n_gen + 1):
        X, y = next(data_generator)
        f_func = MemoizeArray(
            partial(evaluate_fitness,
                    X=X, y=y,
                    regex_components=regex_components)
        )
        pop, pop_fitness = generation(pop, f_func, **gen_kwargs)

        # check if population has gone to fixation
        unique_pop = np.unique(pop, axis=0)
        if unique_pop.shape[0] == 1:
            early_stopping -= 1
            if not early_stopping:
                break
    return pop, pop_fitness, gen


def generation(pop, fitness_func,
               n_selected, n_offspring,
               mut_prob, crossover_prob,
               mut_rate, crossover_rate,
               ind_len, n_choices,
               component_weight):
    # since the training data is changed at each generation we have to
    # reevaluate the pop fitness each cycle.
    pop_fitness = np.asarray([fitness_func(i) for i in pop])
    offspring_types, _ = np.histogram(np.random.random(size=n_offspring),
                                      bins=(0, crossover_prob,
                                            crossover_prob + mut_prob, 1))
    n_crossovers, n_mutants, n_selection = offspring_types
    pop_idx = np.arange(len(pop))

    offspring = []
    for _ in range(n_crossovers // 2):
        offspring += crossover(*pop[np.random.choice(pop_idx, size=2)],
                               ind_len, crossover_rate)
    offspring += [
        mutate(pop[np.random.choice(pop_idx)],
               mut_rate, n_choices, component_weight)
        for _ in range(n_mutants)]
    offspring = np.stack(offspring)
    offspring_fitness = np.asarray([fitness_func(o) for o in offspring])

    selection_idx = np.random.randint(len(pop), size=n_selection)
    selected = pop[selection_idx]
    selected_fitness = pop_fitness[selection_idx]

    next_gen = np.vstack([offspring, selected, pop])
    next_gen_fitness = np.concatenate([offspring_fitness,
                                       selected_fitness,
                                       pop_fitness])
    fittest_idx = np.argsort(next_gen_fitness)[-n_selected:]
    return next_gen[fittest_idx], next_gen_fitness[fittest_idx]


def generate_training_data(training_seqs, labels, n=1000):
    while True:
        if n is None:
            yield training_seqs, labels
        else:
            for i in range(0, len(labels), n):
                yield training_seqs[i: i + n], labels[i: i + n]
            zipped = list(zip(training_seqs, labels))
            shuffle(zipped)
            training_seqs, labels = zip(*zipped)


def crossover(ind1, ind2, ind_len, crossover_rate):
    n_crossovers = np.random.poisson(crossover_rate)
    for cross_pos in np.sort(np.random.randint(ind_len, size=n_crossovers)):
        ind1_c = ind1.copy()
        ind1[cross_pos:] = ind2[cross_pos:]
        ind2[cross_pos:] = ind1_c[cross_pos:]
    return ind1, ind2


def mutate(individual, mut_rate, n_choices, component_weight):
    n_muts = np.random.poisson(mut_rate)
    mutations = np.random.choice(np.arange(n_choices),
                                 size=n_muts,
                                 p=component_weight)
    mut_pos = np.random.randint(0, individual.size, size=n_muts)
    individual[mut_pos] = mutations
    return individual


def evaluate_fitness(individual, X, y, regex_components):
    regex = re.compile(''.join([regex_components[i] for i in individual]))
    pred = [bool(regex.search(seq)) for seq in X]
    if not any(pred):
        return 0
    else:
        score = f1_score(y, pred)
        return score


def parallel_process_evolve(X, y, batch_size, evolve_params):
    evolve_params['data_generator'] = generate_training_data(
        X, y, n=batch_size)
    return evolve_population(**evolve_params)


class RegexGeneticEnsembleClassifier(BaseEstimator, ClassifierMixin):
    '''
    Use genetic algorithm to create a ensemble of regular expressions which
    match patterns in a training dataset.

    Parameters
    ----------
    n_pops: int, optional, default: 10
        Number of separate populations to use evolve. `n_best` of the fittest
        individuals are then taken from each population and used as estimators.

    pop_size: int, optional, default: 100
        Size of each population at initialization.

    n_gen: int, optional, default: 5
        Number of generations of evolution to carry out per population.

    n_best: int, optional, default: 1
        Number of fittest individuals to be taken from each population to be
        used as estimators.

    ind_len: int, optional, default: 10
        The number of components put together to create each individual.

    regex_components: str or array like, recommended, default A-Z
        The components/building blocks for each regex.

    component_weight: array, default None
        The weights (probability of selection) for each component.

    n_selected: int, optional, default: 50
        Genetic algorithm parameter mu, the number of individuals which are
        selected to "survive" each generation.

    n_offspring: int, optional, default: 10
        Genetic algorithm parameter lambda, the total number of mutations,
        crossover events, and cloning events that occur per generation. Ratio
        of mutations to crossover to cloning is determined by `crossover_prob`
        and `mut_prob`.

    crossover_prob: float, optional, default: 0.2
        The fraction of offspring which arise through mating events per
        generation.

    mut_prob: float, optional, default: 0.4
        The fraction of offspring which arise through mutation events per
        generation.

    crossover_rate: int or float, optional, default: 1
        The average number of crossover events which occur per mating.

    mut_rate: int or float, optional, default: 1
        The average number of point mutations that occur per mutation event.

    batch_size: int, optional, default: 100
        The number of samples to use to evaluate fitness each generation.
        Using this minibatch type approach rapidly speeds up training as
        carrying out many regular expressions matches is computationally
        expensive.

    early_stopping: bool or int, optional, default: False
        If True, once a population goes to fixation (i.e. all individuals are
        genetically identical), then the evolution process is halted. If
        `early_stopping` is an integer, this is the number of generations to
        wait for a new individual with higher fitness to arise after fixation
        occurs before stopping.

    n_jobs: int, default: -1
        Number of parallel jobs. Each population is entirely independent and
        can be evolved in parallel. When `n_jobs` == -1, joblib automatically
        assesses the number of available cpus to use.
    '''
    def __init__(self, n_pops=10, pop_size=100, n_gen=5, n_best=1,
                 ind_len=10, regex_components='ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                 component_weight=None, n_selected=50, n_offspring=10,
                 crossover_prob=0.2, mut_prob=0.4,
                 crossover_rate=1, mut_rate=1,
                 batch_size=100, early_stopping=False, n_jobs=-1):
        self.n_pops = n_pops
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.n_best = n_best
        self.ind_len = ind_len
        self.regex_components = regex_components
        self.component_weight = component_weight
        self.n_selected = n_selected
        self.n_offspring = n_offspring
        self.crossover_prob = crossover_prob
        self.mut_prob = mut_prob
        self.crossover_rate = crossover_rate
        self.mut_rate = mut_rate
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.n_jobs = n_jobs

    def fit(self, X, y):
        assert self.mut_prob + self.crossover_prob <= 1
        if self.component_weight is not None:
            assert len(self.regex_components) == len(self.component_weight)
        evolve_params = self.get_params()
        evolve_params['n_choices'] = len(self.regex_components)
        n_jobs = evolve_params.pop('n_jobs')
        batch_size = evolve_params.pop('batch_size')
        n_best = evolve_params.pop('n_best')
        n_pops = evolve_params.pop('n_pops')
        pops = Parallel(n_jobs)(
            delayed(parallel_process_evolve)(X, y, batch_size, evolve_params)
            for _ in range(n_pops))
        self._pops = []
        self._fitnesses = []
        self._n_gens = []
        self.estimators_ = []
        for pop, pop_fitness, n_gens in pops:
            self._pops.append(pop)
            self._fitnesses.append(pop_fitness)
            self._n_gens.append(n_gens)
            best_idx = np.argsort(pop_fitness)[-n_best:]
            for ind in pop[best_idx]:
                self.estimators_.append(''.join(
                    self.regex_components[i] for i in ind))
        return self

    def predict(self, X):
        preds = []
        for regex in self.estimators_:
            regex = re.compile(regex)
            preds.append([bool(regex.search(seq)) for seq in X])
        preds = np.asarray(preds)
        return (preds.mean(0) > 0.5).astype('i')

    def predict_proba(self, X):
        preds = []
        for regex in self.estimators_:
            regex = re.compile(''.join(regex))
            preds.append([bool(regex.search(seq)) for seq in X])
        preds = np.asarray(preds).mean(0)
        proba = np.stack([1 - preds, preds]).T
        return proba
