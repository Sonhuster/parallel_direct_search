import numpy as np
from multiprocessing import Pool
import copy
import matplotlib.pyplot as plt

# Define problem (2 variables)
class MyProblem:
    def __init__(self, error_func, n_var=2):
        self.problem = error_func
        self.n_var = n_var
        self.history = {'x': [], 'f': []}

    def evaluate(self, x, repeat_detector=True):
        if repeat_detector:
            is_repeat, eval = self.detect_repeat_evaluation(x)
            if is_repeat:
                return eval

        self.history['x'].append(x.tolist())
        self.history['f'].append(self.problem(x))
        return self.history['f'][-1]

    def evaluate_batch(self, Inds, n_jobs=1):
        pre_eval_mask, pre_evals_func = np.vstack([self.detect_repeat_evaluation(v.X) for v in Inds]).T
        if np.any(pre_eval_mask.astype(bool)):
            pre_eval_mask = pre_eval_mask.astype(bool)
            pre_eval_list = Inds[~pre_eval_mask]
            pre_inds_idx = np.where(pre_eval_mask)[0]

            [setattr(Inds[idx], 'F', pre_evals_func[idx]) for idx in pre_inds_idx]
            Inds[~pre_eval_mask] = evals_inparallel(self, pre_eval_list, n_jobs=n_jobs,
                                                               repeat_detector=False)
            Inds = Inds.tolist()
        else:
            Inds = evals_inparallel(self, Inds, n_jobs=n_jobs, repeat_detector=False)

        return Inds

    def detect_repeat_evaluation(self, x):
        x = x.tolist()
        try:
            idx = self.history['x'].index(x)
            return True, self.history['f'][idx]
        except ValueError:
            return False, None


class Individual:
    def __init__(self, X=None, F=None):
        self.X = X
        self.F = F

    @classmethod
    def bounded(cls, bounds, X=None, F=None):
        X = set_to_bounds_if_outside(X, bounds[0], bounds[1])
        return cls(X=X, F=F)


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

        self.sim = [Individual.bounded(self.bounds, X=vertex) for vertex in sim]

        # Evaluate
        self.sim = evals_inparallel(self.problem, self.sim, n_jobs=self.n_jobs)
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
            vi_reflect = Individual.bounded(self.bounds, X=v0.X - self.alpha * (vi.X - v0.X))
            reflected.append(vi_reflect)

        reflected = evals(self.problem, reflected)

        best_r_idx = np.argmin([vertex.F for vertex in reflected])
        best_r = reflected[best_r_idx]

        if best_r.F < v0.F:
            # Expansion step
            expanded = []
            for i in range(1, len(simplex)):
                vi = copy.deepcopy(simplex[i])
                vi_expand = Individual.bounded(self.bounds, X=v0.X - self.muy * (vi.X - v0.X))
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
                vi_contract = Individual.bounded(self.bounds, X=v0.X + self.theta * (vi.X - v0.X))
                contracted.append(vi_contract)

            simplex[1:] = evals(self.problem, contracted)

        self.sim = simplex.tolist()

    def initialize_template_for_n_iterations(self, n_iters=1):
        # order = np.argsort([vertex.F for vertex in self.sim])
        # simplex = np.array(self.sim)[order]
        # sim_x = np.array([vertex.X for vertex in simplex])
        # sim_x = np.array([vertex.X for vertex in self.sim]) # Hard core to simulate from best vertex = 0
        # fig, ax = plt.subplots()

        n_dim = len(self.sim[0].X)
        root = 0
        source_root = 0
        alpha_root = 1.0

        coeffs = [np.zeros(n_dim + 1)]
        sources = [source_root]
        alphas = [alpha_root]

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

                    # new_v = sim_x[0] + np.sum(coeff_i[:, None] * sim_x, axis=0)
                    # ax.scatter(new_v[0], new_v[1], c='b')

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

                    # new_v = sim_x[0] + np.sum(coeff_i[:, None] * sim_x, axis=0)
                    # ax.scatter(new_v[0], new_v[1], c='b')

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

                    # new_v = sim_x[0] + np.sum(coeff_i[:, None] * sim_x, axis=0)
                    # ax.scatter(new_v[0], new_v[1], c='b')
            if iter == 0:
                coeffs.pop(0)
                sources.pop(0)
                alphas.pop(0)
        # ax.scatter(sim_x[:, 0], sim_x[:, 1], c='red', label='Simplex Vertices')
        # ax.legend()
        # # plt.show()
        # plt.close(fig)

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

        v_template = []
        for coeff, source in zip(coeffs, sources):
            v_new = sim_x[0] + np.sum(coeff[:, None] * sim_x, axis=0)
            v_template.append(v_new)

        # Eliminate duplicates due to numerical precision
        v_template = np.round(v_template, 6)
        v_template_unique, unique_idx, counts = np.unique(v_template, axis=0, return_index=True, return_counts=True)
        duplicated_points = np.setdiff1d(np.arange(len(v_template)), unique_idx)
        print(f"Total evaluated vertices: {len(unique_idx)} with jumped iterations: {n_iters}")

        # Evaluate all template points (Pre-evaluation detection included)
        all_individuals = np.array([Individual.bounded(self.bounds, X=v) for v in v_template_unique])
        all_individuals = self.problem.evaluate_batch(all_individuals, n_jobs=self.n_jobs)

        v0_idx = np.argmin([ind.F for ind in all_individuals])
        v0 = all_individuals[v0_idx]

        true_idx = unique_idx[v0_idx]
        if counts[v0_idx] > 1:
            # Update possible simplexes and choose the best one
            possible_source = [sources[true_idx]]
            possible_alpha = [alphas[true_idx]]
            for dup_idx in duplicated_points:
                if np.allclose(v0.X, v_template[dup_idx]):
                    if sources[dup_idx] not in possible_source:
                        possible_source.append(sources[dup_idx])
                        possible_alpha.append(alphas[dup_idx])

            possible_sims = []
            for source_p, alpha_p in zip(possible_source, possible_alpha):
                new_sim_x = v0.X + alpha_p * (sim_x - sim_x[source_p])
                possible_sims.append([Individual.bounded(self.bounds, X=vertex) for vertex in new_sim_x])

            possible_sims = self.problem.evaluate_batch(np.array(possible_sims).reshape(-1), n_jobs=self.n_jobs)
            possible_sims = np.array(possible_sims).reshape(-1, len(self.sim)).tolist()
            best_sim = np.argmin([np.mean([ind.F for ind in simplex]) for simplex in possible_sims])
            self.sim = possible_sims[best_sim]
        else:
            # Update simplex
            source_p = sources[true_idx]
            alpha_p = alphas[true_idx]
            sim_x = v0.X + alpha_p * (sim_x - sim_x[source_p])
            self.sim = [Individual.bounded(self.bounds, X=vertex) for vertex in sim_x]
            self.sim = evals(self.problem, self.sim)
        self.history['v0'].append(v0)
        self.history['sim'].append(all_individuals)
        print("Best in template:", v0.X, v0.F)


def evals_inparallel(problem, individuals: np.ndarray|list|Individual, n_jobs=1, repeat_detector=True):
    if n_jobs <= 1:
        return evals(problem, individuals, repeat_detector=repeat_detector)
    else:
        if isinstance(individuals, Individual):
            individuals = [individuals]
        elif type(individuals) in [list, np.ndarray]:
            individuals = list(individuals)
        else:
            raise 'Wrong argument passed'

        chunk_idx = np.array_split(np.arange(len(individuals)), n_jobs)
        fargs = [(problem, np.array(individuals)[chunk], repeat_detector) for chunk in chunk_idx]
        with Pool(processes=min(len(individuals), n_jobs)) as pool:   # change number of workers as needed
            evaluated_lists = pool.starmap(evals, fargs)
            return sum(evaluated_lists, [])


def evals(problem, individuals: np.ndarray|list|Individual, repeat_detector=True):
    def eval(problem, individual):
        if type(individual.X) is not np.ndarray or individual.X is None:
            raise 'Un-determined solution'

        individual.F = problem.evaluate(individual.X, repeat_detector=repeat_detector)
        print('Evaluate:', individual.X, '| Out:', individual.F)
        return individual
    if isinstance(individuals, Individual):
        return eval(problem, individuals)
    elif type(individuals) in [list, np.ndarray]:
        return [eval(problem, individual) for individual in individuals]
    else:
        raise 'Wrong argument passed'