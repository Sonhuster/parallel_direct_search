import numpy as np
from multiprocessing import Pool
import copy
import itertools
import matplotlib.pyplot as plt

# Define problem (2 variables)
class MyProblem:
    def __init__(self, error_func, n_var=2):
        self.problem = error_func
        self.n_var = n_var
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

class MultidirectionalSimplex:
    def __init__(self,
                 problem,
                 x0,
                 bounds=None,
                 alpha=1.0,
                 muy=2.0,
                 theta=0.5,
                 initial_simplex=None,
                 n_dims_per_parallel_computing:int=None,
                 n_jobs=1):
        
        x0 = np.atleast_1d(x0).flatten()
        dtype = x0.dtype if np.issubdtype(x0.dtype, np.inexact) else np.float64
        x0 = np.asarray(x0, dtype=dtype)

        if bounds is not None:
            bounds = np.asarray(bounds)
            lower_bound, upper_bound = bounds[0], bounds[1]
            # check bounds
            if (lower_bound > upper_bound).any():
                raise ValueError("Nelder Mead - one of the lower bounds "
                                    "is greater than an upper bound.")
            x0 = np.clip(x0, lower_bound, upper_bound)

        self.problem = problem
        self.x0 = x0
        self.alpha = alpha
        self.muy = muy
        self.theta = theta
        self.bounds = bounds
        self.initial_simplex = initial_simplex

        self.sim = None
        self.opt = None
        self.history = {'sim': [], 'v0': []}
        self.n_dims_per_parallel_computing = n_dims_per_parallel_computing
        self.n_jobs = n_jobs

        self.template_coeffs = {'coeffs': None, 'sources': None, 'alphas': None}

    def initialize(self):
        nonzdelt = 0.05
        zdelt = 0.00025
        if self.initial_simplex is None:
            N = len(self.x0)
            sim = np.empty((N+1, N), dtype=self.x0.dtype)
            sim[0] = self.x0
            for k in range(N):
                y = np.array(self.x0, copy=True)
                if y[k] != 0:
                    y[k] = (1 + nonzdelt)*y[k]
                else:
                    y[k] = zdelt
                sim[k + 1] = y
        else:
            sim = np.atleast_2d(self.initial_simplex).copy()
            dtype = sim.dtype if np.issubdtype(sim.dtype, np.inexact) else np.float64
            sim = np.asarray(sim, dtype=dtype)
            if sim.ndim != 2 or sim.shape[0] != sim.shape[1] + 1:
                raise ValueError("`initial_simplex` should be an array of shape (N+1,N)")
            if len(self.x0) != sim.shape[1]:
                raise ValueError("Size of `initial_simplex` is not consistent with `x0`")

        self.sim = [Individual(X=vertex) for vertex in sim]

        # Evaluate in parallel
        if self.n_jobs <= 1 or self is None:
            self.sim = evals(self.problem, self.sim)
        else:
            chunk_idx = np.array_split(np.arange(len(self.sim)), self.n_jobs)
            fargs = [(self.problem, np.array(self.sim)[chunk]) for chunk in chunk_idx]
            with Pool(processes=min(len(self.sim), self.n_jobs)) as pool:   # change number of workers as needed
                evaluated_lists = pool.starmap(evals, fargs)
                self.sim = sum(evaluated_lists, [])

        order = np.argsort([vertex.F for vertex in self.sim])
        self.history['sim'].append(np.array(self.sim)[order].tolist())
        self.history['v0'].append(self.sim[order[0]])

    def next(self):
        order = np.argsort([vertex.F for vertex in self.sim])
        simplex = np.array(self.sim)[order]
        v0 = simplex[0]  # best vertex

        self.history['v0'].append(v0)
        self.history['sim'].append(self.sim)
        self.opt = self.history['v0'][np.argmin(v.F for v in [self.history['v0']])]

        # Reflection step
        reflected = []
        for i in range(1, len(simplex)):
            vi = copy.deepcopy(simplex[i])
            vi_reflect = Individual(X=v0.X - self.alpha * (vi.X - v0.X))
            reflected.append(vi_reflect)

        reflected = evals(self.problem, reflected)

        best_r_idx = np.argmin([vertex.F for vertex in reflected])
        best_r = reflected[best_r_idx]

        if best_r.F < v0.F:
            # Expansion step
            expanded = []
            for i in range(1, len(simplex)):
                vi = copy.deepcopy(simplex[i])
                vi_expand = Individual(X=v0.X - self.muy * (vi.X - v0.X))
                expanded.append(vi_expand)

            expanded = evals(self.problem, expanded)
            best_e_idx = np.argmin([vertex.F for vertex in expanded])
            best_e = expanded[best_e_idx]

            if best_e.F < best_r.F:
                simplex[1:] = expanded
            else:
                simplex[1:] = reflected
        else:
            # Contraction step
            contracted = []
            for i in range(1, len(simplex)):
                vi = copy.deepcopy(simplex[i])
                vi_contract = Individual(X=v0.X + self.theta * (vi.X - v0.X))
                contracted.append(vi_contract)

            simplex[1:] = evals(self.problem, contracted)

        self.sim = simplex.tolist()

    def initialize_template_for_n_iterations(self, n_iters=1):
        order = np.argsort([vertex.F for vertex in self.sim])
        simplex = np.array(self.sim)[order]
        sim_x = np.array([vertex.X for vertex in simplex])
        # sim_x = np.array([vertex.X for vertex in self.sim]) # Hard core to simulate from best vertex = 0
        fig, ax = plt.subplots()

        n_dim = len(self.sim[0].X)
        root = 0
        source_root = 0
        alpha_root = 1.0

        coeffs = [np.zeros(n_dim + 1)]
        sources = [source_root]
        alphas = [alpha_root]

        best_v = np.array([-0.25, 0])
        best_v_source = np.array([0.25, 0]) # Page 466 (v_source = best_v_source)
        for iter in range(n_iters):
            root_start = root
            root_end = len(coeffs)  # expand all current roots
            for root in range(root_start, root_end):
                coeff_root = coeffs[root]
                alpha_root = alphas[root]
                source_root = sources[root]
                # Reflection vertices
                for j in range(0, n_dim+1):
                    if j == source_root:
                        continue
                    source_i = j
                    alpha_i = - self.alpha * alpha_root
                    coeff_i = coeff_root.copy()
                    coeff_i[source_i] += alpha_i
                    coeff_i[source_root] -= alpha_i
                    coeffs.append(coeff_i)
                    sources.append(source_i)
                    alphas.append(alpha_i)

                    new_v = sim_x[0] + np.sum(coeff_i[:, None] * sim_x, axis=0)
                    ax.scatter(new_v[0], new_v[1], c='b')

                # Contraction vertices
                for j in range(0, n_dim+1):
                    source_i = j
                    alpha_i = self.theta * alpha_root
                    coeff_i = coeff_root.copy()
                    coeff_i[source_i] += alpha_i
                    coeff_i[source_root] -= alpha_i
                    coeffs.append(coeff_i)
                    sources.append(source_i)
                    alphas.append(alpha_i)

                    new_v = sim_x[0] + np.sum(coeff_i[:, None] * sim_x, axis=0)
                    ax.scatter(new_v[0], new_v[1], c='b')

                # Expansion vertices
                for j in range(0, n_dim+1):
                    if j == source_root:
                        continue
                    source_i = j
                    alpha_i = - self.muy * alpha_root
                    coeff_i = coeff_root.copy()
                    coeff_i[source_i] += alpha_i
                    coeff_i[source_root] -= alpha_i
                    coeffs.append(coeff_i)
                    sources.append(source_i)
                    alphas.append(alpha_i)

                    new_v = sim_x[0] + np.sum(coeff_i[:, None] * sim_x, axis=0)
                    ax.scatter(new_v[0], new_v[1], c='b')
            if iter == 0:
                coeffs.pop(0)
                sources.pop(0)
                alphas.pop(0)
        ax.scatter(sim_x[:, 0], sim_x[:, 1], c='red', label='Simplex Vertices')
        ax.legend()
        # plt.show()

        self.template_coeffs['coeffs'] = coeffs
        self.template_coeffs['sources'] = sources
        self.template_coeffs['alphas'] = alphas

    def next_look_ahead_n_iterations(self, n_iters=1):
        if self.template_coeffs['coeffs'] is None:
            self.initialize_template_for_n_iterations(n_iters=n_iters)
        coeffs = np.array(self.template_coeffs['coeffs'])
        sources = np.array(self.template_coeffs['sources'])
        alphas = np.array(self.template_coeffs['alphas'])

        order = np.argsort([vertex.F for vertex in self.sim])
        simplex = np.array(self.sim)[order]
        sim_x = np.array([vertex.X for vertex in simplex])
        # sim_x = np.array([vertex.X for vertex in self.sim])

        v_template = []
        for coeff, source in zip(coeffs, sources):
            v_new = sim_x[0] + np.sum(coeff[:, None] * sim_x, axis=0)
            v_template.append(v_new)

        # Eliminate duplicates due to numerical precision
        v_template = np.round(v_template, 6)
        v_template_unique, unique_idx, counts = np.unique(v_template, axis=0, return_index=True, return_counts=True)
        duplicated_points = np.setdiff1d(np.arange(len(v_template)), unique_idx)

        # Evaluate all template points
        all_individuals = [Individual(X=v) for v in v_template_unique]
        all_individuals = evals(self.problem, all_individuals)
        v0_idx = np.argmin([ind.F for ind in all_individuals])
        v0 = all_individuals[v0_idx]

        true_idx = unique_idx[v0_idx]
        if counts[v0_idx] > 1:
            possible_source = [sources[true_idx]]
            possible_alpha = [alphas[true_idx]]
            for dup_idx in duplicated_points:
                if np.allclose(v0.X, dup_idx):
                    if sources[dup_idx] not in possible_source:
                        possible_source.append(sources[dup_idx])
                        possible_alpha.append(alphas[dup_idx])

            # Update possible simplecies
            possible_simplices = []
            for source, alpha in zip(possible_source, possible_alpha):
                new_sim_x = sim_x.copy()
                new_sim_x[1:] = v0.X + alpha * (sim_x[1:] - sim_x[source])
                new_sim_x[0] = v0.X
                possible_simplices.append([Individual(X=vertex) for vertex in new_sim_x])
            possible_simplices = [evals(self.problem, simplex) for simplex in possible_simplices]
            best_sim = np.argmin([np.mean([ind.F for ind in simplex]) for simplex in possible_simplices])
            self.sim = possible_simplices[best_sim]
        else:
            # Update simplex
            source = sources[true_idx]
            alpha = alphas[true_idx]
            sim_x[1:] = v0.X + alpha * (sim_x[1:] - sim_x[source])
            sim_x[0] = v0.X
            self.sim = [Individual(X=vertex) for vertex in sim_x]
            self.sim = evals(self.problem, self.sim)
        self.history['v0'].append(v0)
        self.history['sim'].append(all_individuals)
        print("Best in template:", v0.X, v0.F)

        fig, ax = plt.subplots()
        ax.scatter(np.array(v_template)[:, 0], np.array(v_template)[:, 1], c='b', label='Template Points')
        ax.scatter(sim_x[:, 0], sim_x[:, 1], c='red', label='Simplex Vertices')
        ax.legend()
        # plt.show()
        plt.close(fig)

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