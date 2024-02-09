import itertools
import random
import numpy as np
import statistics
from pydantic import BaseModel, Field
import time
import pdb
import bisect
from tqdm import tqdm
from scipy.stats import beta
from scipy import stats
from typing import Optional, Literal, List, Dict
from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


RANDOM_SEED = 42


def summary_stats_str(nums: list[float]):
    mean = statistics.mean(nums)
    median = statistics.median(nums)
    minimum = min(nums)
    maximum = max(nums)
    std_dev = statistics.stdev(nums)
    return (
        f"mean: {mean} median: {median} min: {minimum} max: {maximum} stddev: {std_dev}"
    )


class MadeUpParameters(BaseModel):
    initial_year: float = 2024.0
    compute_schedule: Dict[float, float] = {
        2024.0: 9e25,
        2025: 2.5e26,
        2026: 2.5e27,
        2027: 1e28,
        2028: 9e28,
        2031: 2e30,
    }

    # Experiment design
    experiment_design_default_weeks_lognormal_mu: float = 0.2
    experiment_design_default_weeks_lognormal_sigma: float = 1

    experiment_design_automation_schedule: Dict[float, float] = {
        9e25: 1,
        1e27: 1.05,
        1e28: 1.5,
        1e30: 4,
        3e31: 10,
        1e33: 100,
        1e35: 1000,
    }

    # Software design
    software_design_default_weeks_lognormal_mu: float = 0.3
    software_design_default_weeks_lognormal_sigma: float = 1

    software_design_automation_schedule: Dict[float, float] = {
        9e25: 1,
        1e27: 1.02,
        1e28: 1.3,
        1e30: 2,
        3e31: 7,
        1e33: 25,
        1e35: 100,
    }

    # Software programming
    software_programming_default_weeks_lognormal_mu: float = 1
    software_programming_default_weeks_lognormal_sigma: float = 1

    software_programming_automation_schedule: Dict[float, float] = {
        9e25: 1,
        1e27: 1.3,
        1e28: 5,
        1e30: 15,
        3e31: 50,
        1e33: 500,
        1e35: 5000,
    }

    initial_experiment_monitoring_efficiency: float = 1
    # Experiment monitoring
    experiment_monitoring_efficiency_automation_schedule: Dict[float, float] = {
        9e25: 1,
        1e27: 0.98,
        1e28: 0.95,
        1e30: 0.85,
        3e31: 0.75,
        1e33: 0.55,
        1e35: 0.4,
        1e50: 0.4,
        # 1e26: 1,
    }

    # Wall clock runtime with default monitoring
    runtime_default_weeks_lognormal_mu: float = 1
    runtime_default_weeks_lognormal_sigma: float = 0.7
    runtime_hardness_correlation_level: float = 0.6  # lower is more correlated

    # Experiment results analysis
    experiment_analysis_default_weeks_lognormal_mu: float = 0.5
    experiment_analysis_default_weeks_lognormal_sigma: float = 1

    # TODO: make real. effective compute -> speedup
    experiment_analysis_automation_schedule: Dict[float, float] = {
        9e25: 1,
        1e27: 1.05,
        1e28: 2,
        1e30: 4,
        3e31: 10,
        1e33: 50,
        1e35: 250,
    }

    # Research taste. TODO make real. Make a taste visualization thing
    T_initial: float = 0.1
    research_taste_automation_schedule: Dict[float, float] = {
        9e25: T_initial,
        1e27: 0.1,
        1e28: 0.11,
        1e30: 0.2,
        3e31: 0.3,
        1e33: 0.4,
        1e35: 0.5,
        9e35: 1.2,
        # 10e25: T_initial,
    }

    # Alg efficieny multiplier distribution. Got from playing in Squiggle
    # ae_beta_a: float = 1
    # ae_beta_b: float = 20
    # ae_beta_exp: float = 2
    ae_beta_a: float = 0.01
    ae_beta_b: float = 0.7
    ae_beta_exp: float = 2

    N_EXPERIMENTS: int = 2000

    # How many simluations to run
    N_SIMS: int = 1

    # Stop simulation when there are X experiments left, or have gotten past year Y, or effective compute level Z
    STOP_EXPERIMENTS: int = 0
    STOP_YEAR: float = 3000
    STOP_EC: float = 1e50

    # printing stuff
    PRINT_TIMING: bool = False
    PRINT_EXPERIMENT_INFO: bool = False
    PRINT_TASTE_INFO: bool = False
    PLOT_CORRELATION: bool = False

    # How often to do experiment prioritization
    PRIORITIZATION_CADENCE: int = 50
    DO_AUTOMATION: bool = True


class AISpeedupMultipliers(BaseModel):
    experiment_design: float = 1
    software_design: float = 1
    software_programming: float = 1
    experiment_monitoring_efficiency: float = 1
    experiment_analysis: float = 1
    research_taste: float

    def update(self, effective_compute: float, made_up_parameters: MadeUpParameters):
        if made_up_parameters.DO_AUTOMATION:
            self.experiment_design = log_log_interpolation(
                effective_compute,
                made_up_parameters.experiment_design_automation_schedule,
            )
            self.software_design = log_log_interpolation(
                effective_compute,
                made_up_parameters.software_design_automation_schedule,
            )
            self.software_programming = log_log_interpolation(
                effective_compute,
                made_up_parameters.software_programming_automation_schedule,
            )
            self.experiment_monitoring_efficiency = log_linear_interpolation(
                effective_compute,
                made_up_parameters.experiment_monitoring_efficiency_automation_schedule,
            )
            self.experiment_analysis = log_log_interpolation(
                effective_compute,
                made_up_parameters.experiment_analysis_automation_schedule,
            )
            self.research_taste = log_linear_interpolation(
                effective_compute,
                made_up_parameters.research_taste_automation_schedule,
            )


def log_linear_interpolation(x: float, schedule: Dict[float, float]) -> float:
    """
    Perform log-linear interpolation or extrapolation.

    :param x: The input key for which to interpolate the value.
    :param schedule: The dictionary containing the schedule points.
    :return: The interpolated value.
    """
    # Convert the dictionary into sorted lists of keys and values
    keys = np.array(sorted(schedule.keys()))
    values = np.array([schedule[k] for k in keys])
    log_keys = np.log10(keys)

    if x < keys[0]:  # Extrapolate left
        slope = (values[1] - values[0]) / (log_keys[1] - log_keys[0])
        return values[0] + slope * (np.log10(x) - log_keys[0])
    elif x > keys[-1]:  # Extrapolate right
        slope = (values[-1] - values[-2]) / (log_keys[-1] - log_keys[-2])
        return values[-1] + slope * (np.log10(x) - log_keys[-1])
    else:
        # Perform linear interpolation on the log of the keys
        return np.interp(np.log10(x), log_keys, values)


def log_log_interpolation(x: float, schedule: Dict[float, float]) -> float:
    """
    Perform log-log interpolation or extrapolation.

    :param x: The input key for which to interpolate the value.
    :param schedule: The dictionary containing the schedule points.
    :return: The interpolated value.
    """
    # Convert the dictionary into sorted lists of keys and values
    keys = np.array(sorted(schedule.keys()))
    values = np.array([schedule[k] for k in keys])
    log_keys = np.log10(keys)
    log_values = np.log10(values)

    if x < keys[0]:  # Extrapolate left
        slope = (log_values[1] - log_values[0]) / (log_keys[1] - log_keys[0])
        extrapolated_log_value = log_values[0] + slope * (np.log10(x) - log_keys[0])
        return 10**extrapolated_log_value
    elif x > keys[-1]:  # Extrapolate right
        slope = (log_values[-1] - log_values[-2]) / (log_keys[-1] - log_keys[-2])
        extrapolated_log_value = log_values[-1] + slope * (np.log10(x) - log_keys[-1])
        return 10**extrapolated_log_value
    else:
        # Perform linear interpolation on the log of both keys and values
        return 10 ** np.interp(np.log10(x), log_keys, log_values)


def linear_log_interpolation(x: float, schedule: Dict[float, float]) -> float:
    """
    Perform linear-log interpolation or extrapolation.

    :param x: The input key for which to interpolate the value.
    :param schedule: The dictionary containing the schedule points.
    :return: The interpolated (or extrapolated) value on the original scale.
    """
    # Convert the dictionary into sorted lists of keys and values
    keys = np.array(sorted(schedule.keys()))
    log_values = np.log10(
        np.array([schedule[k] for k in keys])
    )  # Transform values to log scale

    # Check if x is outside the bounds and perform extrapolation if needed
    if x < keys[0]:  # x is less than the smallest key
        # Use slope between the first two points for extrapolation
        slope = (log_values[1] - log_values[0]) / (keys[1] - keys[0])
        extrapolated_log_value = log_values[0] + slope * (x - keys[0])
        return 10**extrapolated_log_value
    elif x > keys[-1]:  # x is greater than the largest key
        # Use slope between the last two points for extrapolation
        slope = (log_values[-1] - log_values[-2]) / (keys[-1] - keys[-2])
        extrapolated_log_value = log_values[-1] + slope * (x - keys[-1])
        return 10**extrapolated_log_value

    # Perform linear interpolation on the keys and exponential transformation on the result
    interpolated_log_value = np.interp(x, keys, log_values)
    return 10**interpolated_log_value


def weighted_sample_without_replacement(population, weights, k):
    """Select k unique items from population with given weights without replacement."""
    cumulative_weights = list(itertools.accumulate(weights))
    selections = []

    for _ in range(k):
        x = random.uniform(0, cumulative_weights[-1])
        i = bisect.bisect(cumulative_weights, x)

        # hacky fix for now, not sure of issue
        i = min(i, len(population) - 1)

        selections.append(population[i])

        # Update weights and cumulative weights to exclude the selected item
        cumulative_weights[i:] = [w - weights[i] for w in cumulative_weights[i:]]
        if i + 1 < len(cumulative_weights):
            cumulative_weights.pop(i)
        weights.pop(i)
        population.pop(i)

    return selections


# # Step 1: Define correlation and generate correlated variables
# def generate_correlated_variables(rho, size=1):
#     # Define the correlation matrix
#     C = np.array([[1, rho], [rho, 1]])
#     # Perform Cholesky decomposition
#     L = np.linalg.cholesky(C)
#     # Generate independent standard normal variables
#     Z = np.random.normal(size=(2, size))
#     # Transform to get correlated variables
#     X = L @ Z
#     return X


# # Step 2: Apply correlated variables to influence distribution parameters
# def generate_correlated_runtime_and_efficiency(made_up_parameters: MadeUpParameters):
#     X1 = np.random.normal(size=made_up_parameters.N_EXPERIMENTS)

#     # Adjust lognormal mean based on X1 (you might need to scale/transform X1 appropriately)
#     adjusted_mu = made_up_parameters.runtime_default_weeks_lognormal_mu + X1 * 0.2

#     # Generate lognormal distributed variables based on adjusted parameters
#     runtime_default_weeks = np.exp(
#         adjusted_mu
#         + made_up_parameters.runtime_default_weeks_lognormal_sigma
#         * np.random.normal(size=len(X1))
#     )

#     # Adjust beta parameters or its transformation based on X2
#     # Since beta distribution parameters must be > 0, ensure transformation is valid
#     alpha, beta_param = (
#         made_up_parameters.ae_beta_a,
#         made_up_parameters.ae_beta_b,
#     )  # Original beta parameters
#     alg_efficiency = np.maximum(
#         1,
#         (
#             beta(
#                 np.maximum(alpha + X1 * 0.01, 0.01),
#                 np.maximum(beta_param - X1 * 0.3, 0.01),
#             ).rvs()
#             + 1
#         )
#         ** made_up_parameters.ae_beta_exp,
#     )  # Example transformation

#     return runtime_default_weeks.tolist(), alg_efficiency.tolist()


def plot_correlation(runtime_default_weeks, alg_efficiency):
    # 1. Compute and print the Pearson correlation coefficient
    correlation_coefficient = np.corrcoef(runtime_default_weeks, alg_efficiency)[0, 1]
    print(f"Pearson correlation coefficient: {correlation_coefficient}")

    # 2. Scatter Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(runtime_default_weeks, alg_efficiency, alpha=0.5)
    plt.title("Scatter Plot of Runtime Default Weeks vs. Algorithm Efficiency")
    plt.xlabel("Runtime Default Weeks")
    plt.ylabel("Algorithm Efficiency")
    plt.grid(True)
    plt.show()

    # 3. Histograms
    # plt.figure(figsize=(10, 6))
    # plt.hist(runtime_default_weeks, bins=30, alpha=0.5, label="Runtime Default Weeks")
    # plt.hist(alg_efficiency, bins=30, alpha=0.5, label="Algorithm Efficiency")
    # plt.title("Histogram of Runtime Default Weeks and Algorithm Efficiency")
    # plt.legend()
    # plt.show()

    # 4. Joint Distribution Plot
    # sns.jointplot(
    #     x=runtime_default_weeks,
    #     y=alg_efficiency,
    #     kind="scatter",
    #     joint_kws={"alpha": 0.5},
    # )
    # plt.xlabel("Runtime Default Weeks")
    # plt.ylabel("Algorithm Efficiency")
    # plt.show()


class Experiment(BaseModel):
    id: int = 0
    start_year: int = None
    made_up_parameters: MadeUpParameters
    experiment_design_speed_in_weeks: float = 1
    software_design_speed_in_weeks: float = 1
    software_programming_speed_in_weeks: float = 1
    runtime_default_weeks: float = 1
    experiment_analysis_speed_in_weeks: float = 1
    algorithmic_efficiency_multiplier: float
    true_priority: float = -1
    end_runtime: float = -1

    def __init__(
        self,
        made_up_parameters: MadeUpParameters,
        **data,
    ):
        super().__init__(
            made_up_parameters=made_up_parameters,
            experiment_design_speed_in_weeks=np.random.lognormal(
                made_up_parameters.experiment_design_default_weeks_lognormal_mu,
                made_up_parameters.experiment_design_default_weeks_lognormal_sigma,
            ),
            software_design_speed_in_weeks=np.random.lognormal(
                made_up_parameters.software_design_default_weeks_lognormal_mu,
                made_up_parameters.software_design_default_weeks_lognormal_sigma,
            ),
            software_programming_speed_in_weeks=np.random.lognormal(
                made_up_parameters.software_programming_default_weeks_lognormal_mu,
                made_up_parameters.software_programming_default_weeks_lognormal_sigma,
            ),
            # runtime_default_weeks=np.random.lognormal(
            #     made_up_parameters.runtime_default_weeks_lognormal_mu,
            #     made_up_parameters.runtime_default_weeks_lognormal_sigma,
            # ),
            experiment_analysis_speed_in_weeks=np.random.lognormal(
                made_up_parameters.experiment_analysis_default_weeks_lognormal_mu,
                made_up_parameters.experiment_analysis_default_weeks_lognormal_sigma,
            ),
            **data,
        )

    def pre_run_work_time_in_weeks(self, speedup_multipliers: AISpeedupMultipliers):
        return (
            self.experiment_design_speed_in_weeks
            / speedup_multipliers.experiment_design
            + self.software_design_speed_in_weeks / speedup_multipliers.software_design
            + self.software_programming_speed_in_weeks
            / speedup_multipliers.software_programming
        )

    def runtime_in_weeks(self, speedup_multipliers: AISpeedupMultipliers):
        return (
            self.runtime_default_weeks
            * speedup_multipliers.experiment_monitoring_efficiency
        )

    def post_run_work_time_in_weeks(self, speedup_multipliers: AISpeedupMultipliers):
        return (
            self.experiment_analysis_speed_in_weeks
            / speedup_multipliers.experiment_analysis
        )

    def update_true_priority(self, speedup_multipliers: AISpeedupMultipliers):
        time_to_complete = self.pre_run_work_time_in_weeks(
            speedup_multipliers
        ) + self.post_run_work_time_in_weeks(speedup_multipliers)
        # Copying prioritization advice from ChatGPT https://chat.openai.com/share/3628828c-edd1-4b6c-97d7-9c2470a2bdbe
        self.true_priority = (
            self.algorithmic_efficiency_multiplier - 1
        ) / time_to_complete

    def start(self, year: float, speedup_multipliers: AISpeedupMultipliers):
        self.start_year = year
        self.end_runtime = (
            self.start_year
            + (
                self.pre_run_work_time_in_weeks(speedup_multipliers)
                + self.runtime_in_weeks(speedup_multipliers)
            )
            / 52
        )

    def __str__(self):
        return (
            f"(algorithmic_efficiency_multiplier={self.algorithmic_efficiency_multiplier}, "
            f"true_priority={self.true_priority}, "
            f"end_runtime={self.end_runtime})"
        )

    def __repr__(self):
        return (
            f"(algorithmic_efficiency_multiplier={self.algorithmic_efficiency_multiplier}, "
            f"true_priority={self.true_priority}, "
            f"end_runtime={self.end_runtime})"
        )


class WorldState(BaseModel):
    finished_experiments: int = 0
    started_experiments: int = 0
    year: float
    made_up_parameters: MadeUpParameters
    ai_speedup_multipliers: AISpeedupMultipliers
    algorithmic_efficiency: float = 1
    physical_compute: float = None
    prospective_experiments: List[Experiment] = Field(default_factory=list)
    next_experiments: List[Experiment] = Field(default_factory=list)
    in_progress_experiments: List[Experiment] = Field(default_factory=list)
    # todo experiment monitoring

    def __init__(
        self,
        made_up_parameters: MadeUpParameters,
        ai_speedup_multipliers: AISpeedupMultipliers,
        **data,
    ):
        super().__init__(
            physical_compute=made_up_parameters.compute_schedule[2024],
            year=made_up_parameters.initial_year,
            prospective_experiments=self._initialize_prospective_experiments(
                made_up_parameters, ai_speedup_multipliers
            ),
            made_up_parameters=made_up_parameters,
            ai_speedup_multipliers=ai_speedup_multipliers,
            **data,
        )

    def _initialize_prospective_experiments(
        self,
        made_up_parameters: MadeUpParameters,
        ai_speedup_multipliers: AISpeedupMultipliers,
    ):
        # Generate N_EXPERIMENTS beta-distributed random variates
        random_variates = beta.rvs(
            made_up_parameters.ae_beta_a,
            made_up_parameters.ae_beta_b,
            size=made_up_parameters.N_EXPERIMENTS,
        )

        # Increment by 1 and ensure each value is at least 1, then raise to the exponent
        algorithmic_efficiency_multipliers = (
            np.maximum(1, random_variates + 1) ** made_up_parameters.ae_beta_exp
        )

        # Convert to list if necessary
        algorithmic_efficiency_multipliers_list = (
            algorithmic_efficiency_multipliers.tolist()
        )

        runtime_default_weeks_values = np.random.lognormal(
            mean=1, sigma=0.7, size=made_up_parameters.N_EXPERIMENTS
        )

        # Convert lists to numpy arrays for easier manipulation
        efficiency_array = np.array(algorithmic_efficiency_multipliers_list)
        runtime_array = np.array(runtime_default_weeks_values)

        # 1. Get sorted indices for both arrays
        efficiency_sorted_indices = np.argsort(efficiency_array)
        runtime_sorted_indices = np.argsort(runtime_array)

        # To maintain a base level of correlation, we start by aligning one list with the sorted order of the other.
        # Here, we'll align runtime to the sorted order of efficiency.
        runtime_aligned_with_efficiency = runtime_array[runtime_sorted_indices][
            np.argsort(efficiency_sorted_indices)
        ]

        # 2. Introduce partial disorder by swapping elements within the aligned list.
        # Determine the number of elements to swap to control the level of correlation; more swaps = less correlation.
        num_swaps = int(
            len(runtime_array) * made_up_parameters.runtime_hardness_correlation_level
        )

        for _ in range(num_swaps):
            # Randomly select two indices to swap
            idx1, idx2 = np.random.choice(len(runtime_array), 2, replace=False)
            # Perform the swap
            (
                runtime_aligned_with_efficiency[idx1],
                runtime_aligned_with_efficiency[idx2],
            ) = (
                runtime_aligned_with_efficiency[idx2],
                runtime_aligned_with_efficiency[idx1],
            )

        if made_up_parameters.PLOT_CORRELATION:
            plot_correlation(
                runtime_aligned_with_efficiency,
                algorithmic_efficiency_multipliers_list,
            )

        print(
            f"Total alg efficiency multiplier avaliable: {np.prod(algorithmic_efficiency_multipliers_list)}"
        )

        experiments = [
            Experiment(
                id=i,
                made_up_parameters=made_up_parameters,
                algorithmic_efficiency_multiplier=algorithmic_efficiency_multipliers_list[
                    i
                ],
                runtime_default_weeks=runtime_aligned_with_efficiency[i],
            )
            for i in range(made_up_parameters.N_EXPERIMENTS)
        ]
        if made_up_parameters.PRINT_TASTE_INFO:
            self.sweep_taste_and_print_info(experiments, ai_speedup_multipliers)
        return experiments

    def sweep_taste_and_print_info(
        self,
        experiments: List[Experiment],
        ai_speedup_multipliers: AISpeedupMultipliers,
    ):
        taste_options = np.arange(0, 1.5, 0.1).tolist()
        for e in experiments:
            e.update_true_priority(ai_speedup_multipliers)
        # experiments = sorted(experiments, key=lambda e: e.true_priority)
        N_TRIALS = 10
        NUM_TO_SAMPLE = MadeUpParameters().PRIORITIZATION_CADENCE
        for taste in taste_options:
            true_prio_means = []
            for _ in range(N_TRIALS):
                sampled_experiments = self.sample_next_experiments(
                    deepcopy(experiments),
                    taste,
                    NUM_TO_SAMPLE,
                )
                true_prio_means.append(
                    sum([e.true_priority for e in sampled_experiments]) / NUM_TO_SAMPLE
                )
            print(
                f"taste: {taste} N_TRIALS {N_TRIALS} {summary_stats_str(true_prio_means)}"
            )

    def start_next_experiment(self):
        # start_time = time.time()
        # print(self.next_experiments)
        if not self.next_experiments:
            self.select_next_experiments()
        next_experiment = self.next_experiments.pop(0)
        next_experiment.start(self.year, self.ai_speedup_multipliers)
        self.in_progress_experiments.append(next_experiment)
        if self.started_experiments % (self.made_up_parameters.N_EXPERIMENTS / 50) == 0:
            print(f"Starting experiment {self.started_experiments}: {next_experiment}]")
        self.started_experiments += 1
        self.in_progress_experiments = sorted(
            self.in_progress_experiments,
            key=lambda e: e.end_runtime,
        )
        self.update_year(
            self.year
            + (
                next_experiment.pre_run_work_time_in_weeks(self.ai_speedup_multipliers)
                / 52
            )
        )
        # if self.made_up_parameters.DO_TIMING:
        #     print(f"Next eperiment took: {time.time() - start_time:.2f} seconds")

    def sample_next_experiments(self, experiments, taste, num_to_sample):
        # Assumes experiments already sorted by ascending priority
        weights = [e.true_priority**taste for e in experiments]
        return weighted_sample_without_replacement(
            population=experiments,
            weights=weights,
            k=min(num_to_sample, len(experiments)),
        )

    def select_next_experiments(self):
        start_time = time.time()
        for e in self.prospective_experiments:
            e.update_true_priority(self.ai_speedup_multipliers)
        # self.prospective_experiments = sorted(
        #     self.prospective_experiments, key=lambda e: e.true_priority
        # )

        self.next_experiments = self.sample_next_experiments(
            self.prospective_experiments,
            self.ai_speedup_multipliers.research_taste,
            self.made_up_parameters.PRIORITIZATION_CADENCE,
        )
        if self.made_up_parameters.PRINT_EXPERIMENT_INFO:
            print(f"\ntop 10 experiments: {self.prospective_experiments[-10:]}\n")
            print(f"some selected experiments: {self.next_experiments[:10]}\n")
            print(f"bottom 10 experiments: {self.prospective_experiments[:10]}\n")
        # pdb.set_trace()
        # for e in self.next_experiments:
        #     print(e)
        #     self.prospective_experiments.remove(e)
        if self.made_up_parameters.PRINT_TIMING:
            print(f"Updating priorites took: {time.time() - start_time:.2f} seconds")

    def update_year(self, new_year):
        prev_year = self.year
        self.year = new_year
        if int(new_year) > int(prev_year):
            self.print_properties()

    def research_step(self):
        if not self.in_progress_experiments and self.prospective_experiments:
            self.start_next_experiment()
        while (
            self.in_progress_experiments[0].end_runtime > self.year
            and self.prospective_experiments
        ):
            self.start_next_experiment()

        finished_experiment = self.in_progress_experiments.pop(0)
        self.update_year(
            max(finished_experiment.end_runtime, self.year)
            + (
                finished_experiment.post_run_work_time_in_weeks(
                    self.ai_speedup_multipliers
                )
                / 52
            )
        )
        self.physical_compute = linear_log_interpolation(
            self.year, self.made_up_parameters.compute_schedule
        )
        self.algorithmic_efficiency *= (
            finished_experiment.algorithmic_efficiency_multiplier
        )
        # print(self.algorithmic_efficiency)
        # print(finished_experiment.algorithmic_efficiency_multiplier)
        self.ai_speedup_multipliers.update(
            self.effective_compute, self.made_up_parameters
        )
        if (
            self.finished_experiments % (self.made_up_parameters.N_EXPERIMENTS / 50)
            == 0
        ):
            self.print_properties()
        self.finished_experiments += 1

        # update AI multipliers

    def is_done(self):
        return (
            len(self.prospective_experiments + self.in_progress_experiments)
            <= self.made_up_parameters.STOP_EXPERIMENTS
            or self.year >= self.made_up_parameters.STOP_YEAR
            or self.effective_compute > self.made_up_parameters.STOP_EC
        )

    @property
    def effective_compute(self):
        return self.algorithmic_efficiency * self.physical_compute

    def print_properties(self, true_prio_mean=None):
        print(
            f"year: {self.year} alg eff {self.algorithmic_efficiency} eff compute {self.effective_compute} started_experiments: {self.started_experiments} finished_experiments: {self.finished_experiments} phys compute {self.physical_compute} ai speedup multipliers: {self.ai_speedup_multipliers} next experiments true prio mean {true_prio_mean}"
        )

    def get_properties_dict(self):
        return {
            "year": self.year,
            "algorithmic efficiency": self.algorithmic_efficiency,
            "effective_compute": self.effective_compute,
            "physical_compute": self.physical_compute,
            "started_experiments": self.started_experiments,
            "finished_experiments": self.finished_experiments,
            "software_design_multiplier": self.ai_speedup_multipliers.software_design,
        }


class SimResult(BaseModel):
    timesteps: List[WorldState] = Field(default_factory=list)

    def print_results(self):
        self.timesteps[-1].print_properties()
        data = [t.get_properties_dict() for t in self.timesteps]

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Convert 'year' to a proper datetime type if needed (optional, for better x-axis formatting)
        # df["year"] = pd.to_datetime(df["year"], format="%Y.%f")

        properties = [
            "algorithmic efficiency",
            "effective_compute",
            "physical_compute",
            "started_experiments",
            "finished_experiments",
            "software_design_multiplier",
        ]  # List all properties

        # Plotting with log scale and x-axis adjustment
        # plt.figure(figsize=(12, 20))  # Adjust figure size as needed

        # num_plots = len(properties)

        # for i, prop in enumerate(properties, start=1):
        #     ax = plt.subplot(num_plots, 1, i)
        #     df.plot(x="year", y=prop, ax=ax, legend=True)
        #     ax.set_title(prop)
        #     ax.set_ylabel("Value (log scale)")
        #     ax.set_xlabel("Year")
        #     ax.set_yscale("log")  # Set y-axis to logarithmic scale
        #     ax.set_xlim(
        #         df["year"].min(), df["year"].max()
        #     )  # Adjust x-axis to cover only the years mentioned

        # plt.tight_layout()
        # plt.show()

        # Initialize a single figure and axis for plotting
        plt.figure(figsize=(12, 8))  # Adjust figure size as needed
        ax = (
            plt.gca()
        )  # Get the current Axes instance on the current figure matching the given keyword args, or create one.

        # Plot each property on the same axis
        for prop in properties:
            df.plot(x="year", y=prop, ax=ax, label=prop, logy=True)

        # Set the y-axis to logarithmic scale (optional, based on your preference and data range)
        # ax.set_yscale('log')

        # Set labels and title
        plt.xlabel("Year")
        plt.ylabel("Value")
        plt.title("Properties Over Time")

        # Optionally adjust the x-axis to cover only the years mentioned, if necessary
        min_year, max_year = df["year"].min(), df["year"].max()
        ax.set_xlim(left=min_year, right=max_year)

        plt.legend()  # Show legend to identify each line
        plt.tight_layout()
        plt.show()


class AllSimResults(BaseModel):
    sim_results: List[SimResult] = Field(default_factory=list)

    def print_results(self):
        return
        for sim_result in self.sim_results:
            sim_result.print_results()


def main():
    random.seed(42)

    s = time.time()
    all_sim_results = AllSimResults()

    for _ in tqdm(range(MadeUpParameters().N_SIMS)):
        made_up_parameters = MadeUpParameters()
        sim_result = SimResult()
        world_state = WorldState(
            made_up_parameters,
            AISpeedupMultipliers(research_taste=made_up_parameters.T_initial),
        )
        sim_result.timesteps.append(deepcopy(world_state))

        while not world_state.is_done():
            world_state.research_step()
            sim_result.timesteps.append(deepcopy(world_state))

        sim_result.print_results()
        all_sim_results.sim_results.append(deepcopy(sim_result))

    print(
        f"Ran {made_up_parameters.N_SIMS} simluations in {round(time.time() - s, 2)} seconds\n"
    )
    all_sim_results.print_results()


if __name__ == "__main__":
    main()
