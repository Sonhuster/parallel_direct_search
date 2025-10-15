import numpy as np
from multiprocessing import Pool
import copy
import itertools

# Define problem (2 variables)
class MyProblem:
    def __init__(self, error_func, n_var=2, xl=np.array([-5, -5]), xu=np.array([5, 5])):
        self.problem = error_func
        self.n_var = n_var
        self.xl = xl
        self.xu = xu
        self.history = {'x': [], 'f': []}

    def evaluate(self, x):
        x = x.tolist()

        if len(self.history['x']) == 0:
            self.history['x'].append(x)
            self.history['f'].append(self.problem(x))
            return self.history['f'][-1]
        else:
            if x in self.history['x']:
                print('Repeat prediction detected')
                idx = self.history['x'].index(x)
                return self.history['f'][idx]
            else:
                self.history['x'].append(x)
                self.history['f'].append(self.problem(x))
                return self.history['f'][-1]

class Individual:
    def __init__(self, X=None, F=None):
        self.X = X
        self.F = F

def is_better(_new, _old, eps=0.0):
    # Only for un-constraint and single objective optimization problems
    return True if _old.F - _new.F > eps else False

def step_along_axis(problem, x, delta, axis):
    # copy and add delta to the new point
    X = np.copy(x)

    # now add to the current solution
    X[axis] = X[axis] + delta[axis]

    # repair if out of bounds if necessary
    X = set_to_bounds_if_outside_by_problem(problem, X)

    return Individual(X=X)

def exploration_move(problem, center, sign, delta, rho, randomize=True):
    n_var = problem.n_var

    if randomize:
        K = np.random.permutation(n_var)
    else:
        K = np.arange(n_var)

    for k in K:
        _delta = sign[k] * rho * delta

        # Sequentially Exploration
        _explr = step_along_axis(problem, center.X, _delta, k)
        _explr = evals(problem, _explr)
        if is_better(_explr, center, eps=0.0):
            center = _explr
        else:
            # Try the opposite direction
            _explr = step_along_axis(problem, center.X, -1 * _delta, k)
            _explr = evals(problem, _explr)
            if is_better(_explr, center, eps=0.0):
                center = _explr

    return center

def exploration_move_parallel(problem, center, delta, rho, n_dims_per_parallel_computing:int=None, randomize=True, n_procs=1):
    n_var = problem.n_var
    if n_dims_per_parallel_computing is None:
        # heuristic: choose dimensions so workload ~ balanced with n_procs
        if n_procs >= np.power(3, n_var) - 1:
            n_dims_per_parallel_computing = n_var
        else:
            for i in range(1, n_var+1):
                n_evals_of_hypercube = np.power(3, i) - 1
                if n_evals_of_hypercube / n_procs <= 1.5:
                    n_dims_per_parallel_computing = i
                    continue
                elif 1.5 <= n_evals_of_hypercube / n_procs <= 2:
                    n_dims_per_parallel_computing = i
                    break
    else:
        n_dims_per_parallel_computing = min(n_dims_per_parallel_computing, n_var)

    if randomize:
        K = np.random.permutation(n_var)
    else:
        K = np.arange(n_var)

    # Step size for each dimension
    step_sizes = rho * delta * np.ones(n_var)

    for i in range(0, n_var, n_dims_per_parallel_computing):
        dims = K[i:i + n_dims_per_parallel_computing]

        # Generate all {-1,0,1}^len(dims) moves (except all zero)
        dirs = np.array(np.meshgrid(*[[-1, 0, 1]] * len(dims))).T.reshape(-1, len(dims))
        dirs = dirs[~np.all(dirs == 0, axis=1)]

        candidates = []
        for d in dirs:
            delta_vec = np.zeros(n_var)
            delta_vec[dims] = d * step_sizes[dims]
            candidates.append(step_along_vector(problem, center.X, delta_vec))

        # Evaluate in parallel
        if n_procs <= 1:
            evaluated = evals(problem, candidates)
        else:
            chunk_idx = np.array_split(np.arange(len(candidates)), n_procs)
            fargs = [(problem, np.array(candidates)[chunk]) for chunk in chunk_idx]
            with Pool(processes=min(len(candidates), n_procs)) as pool:   # change number of workers as needed
                evaluated_lists = pool.starmap(evals, fargs)
                evaluated = sum(evaluated_lists, [])

        # Update center after evaluation
        for cand in evaluated:
            if is_better(cand, center, eps=0.0):
                center = cand

    return center


def pattern_move(problem, current, direction, step_size):
    # calculate the new X and repair out of bounds if necessary
    X = current.X + step_size * direction
    set_to_bounds_if_outside_by_problem(problem, X)

    # create the new center individual
    return Individual(X=X, F=problem.evaluate(X))

def calc_sign(direction):
    sign = np.sign(direction)
    sign[sign == 0] = -1
    return sign

def step_along_vector(problem, x, delta_vec):
    """
    Move along multiple axes simultaneously by a vector of deltas.
    """
    X = np.copy(x)
    X = X + delta_vec

    # Repair if out of bounds
    X = set_to_bounds_if_outside_by_problem(problem, X)

    return Individual(X=X)

def set_to_bounds_if_outside_by_problem(problem, X):
    return set_to_bounds_if_outside(X, problem.xl, problem.xu)

def set_to_bounds_if_outside(X, xl, xu):
    _X, only_1d = at_least_2d_array(X, return_if_reshaped=True)

    if xl is not None:
        xl = np.repeat(xl[None, :], _X.shape[0], axis=0)
        _X[_X < xl] = xl[_X < xl]

    if xu is not None:
        xu = np.repeat(xu[None, :], _X.shape[0], axis=0)
        _X[_X > xu] = xu[_X > xu]

    if only_1d:
        return _X[0, :]
    else:
        return _X

def at_least_2d_array(x, extend_as="row", return_if_reshaped=False):
    if x is None:
        return x
    elif not isinstance(x, np.ndarray):
        x = np.array([x])

    has_been_reshaped = False

    if x.ndim == 1:
        if extend_as.startswith("r"):
            x = x[None, :]
        elif extend_as.startswith("c"):
            x = x[:, None]
        else:
            raise Exception("The option `extend_as` should be either `row` or `column`.")

        has_been_reshaped = True

    if return_if_reshaped:
        return x, has_been_reshaped
    else:
        return x

class PatternSearch:
    def __init__(self,
                 problem,
                 x0=None,
                 init_delta=0.25,
                 init_rho=0.5,
                 step_size=1.0,
                 seed=None,
                 n_dims_per_parallel_computing:int=None,
                 n_jobs=1):
        self.problem = problem
        self.x0 = x0
        self.init_rho = init_rho
        self.init_delta = init_delta
        self.step_size = step_size

        self.n_not_improved = 0

        self._rho = init_rho
        self._delta = None
        self._center = None
        self._explr = None
        self._current = None
        self._trial = None
        self._direction = None
        self._sign = None

        self.opt = None
        self.history = {'x': [], 'f': []}
        self.n_dims_per_parallel_computing = n_dims_per_parallel_computing
        self.n_jobs = n_jobs
        if isinstance(seed, bool) and seed:
            seed = np.random.randint(0, 10000000)
            self.seed = seed

        # if a seed is set, then use it to call the random number generators
        if seed is not None:
            import random
            random.seed(seed)
            np.random.seed(seed)

    def initialize(self):
        self._center = Individual(X=self.x0, F=self.problem.evaluate(self.x0))
        self.history['x'].append(self._center.X)
        self.history['f'].append(self._center.F)
        self._explr = copy.deepcopy(self._center)
        self._sign = np.ones(self.problem.n_var) # Only for sequential approach

        xl, xu = self.problem.xl, self.problem.xu
        self._delta = self.init_delta * (xu - xl)

    def next(self):
        has_improved = is_better(self._explr, self._center)

        if not has_improved:
            print('Not improve')
            self.n_not_improved += 1
            self._rho = self.init_rho ** self.n_not_improved
            self._explr = exploration_move_parallel(
                self.problem, self._center, self._delta, self._rho,
                n_procs=self.n_jobs,
                n_dims_per_parallel_computing=self.n_dims_per_parallel_computing
            )

            # self._explr = exploration_move(
            #     self.problem, self._center, self._sign, self._delta, self._rho)
            self.history['x'].append(self._explr.X)
            self.history['f'].append(self._explr.F)
        else:
            self._direction = (self._explr.X - self._center.X)
            print('Pattern Move:', self._direction, self._explr.X)
            self._center = self._explr

            # Explicit pattern move
            self._trial = pattern_move(self.problem, self._center, self._direction, self.step_size)

            self._sign = calc_sign(self._direction) # Only for sequential approach

            self._explr = exploration_move_parallel(
                self.problem, self._trial, self._delta, self._rho,
                n_procs=self.n_jobs,
                n_dims_per_parallel_computing=self.n_dims_per_parallel_computing
            )
            assert self._explr.F <= self._trial.F, f'MODE: EXPLORE in Pattern move - Wrong'
            self.history['x'].append(self._explr.X)
            self.history['f'].append(self._explr.F) #TODO: Explore why next iteration result is worst than ealier
            # self._explr = exploration_move(
            #     self.problem, self._trial, self._sign, self._delta, self._rho
            # )
        self.opt = self.history['x'][np.argmin(self.history['f'])]

def evals(problem, individuals: np.ndarray|list|Individual):
    def eval(problem, individual: Individual):
        if type(individual.X) is not np.ndarray or individual.X is None:
            raise 'Un-determined solution'

        individual.F = problem.evaluate(individual.X)
        print('Evaluate:', individual.X, '| Out:', individual.F)
        return individual
    if isinstance(individuals, Individual):
        return eval(problem, individuals)
    elif type(individuals) in [list, np.ndarray]:
        return [eval(problem, individual) for individual in individuals]
    else:
        raise 'Wrong argument passed'
