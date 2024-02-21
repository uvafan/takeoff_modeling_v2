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
import math


RANDOM_SEED = 42


def p5_and_95_to_mu_and_sigma(p5_value, p95_value):
    # Convert percentiles to z-scores of the standard normal distribution
    z5 = stats.norm.ppf(0.05)
    z95 = stats.norm.ppf(0.95)

    # Solve for sigma and mu of the log-normal distribution
    # Using the equations derived from the log-normal CDF and the given percentiles
    sigma = (np.log(p95_value) - np.log(p5_value)) / (z95 - z5)
    mu = np.log(p5_value) - z5 * sigma

    return mu, sigma


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

    # Physical compute schedule

    compute_schedule: Dict[float, float] = {
        2024.0: 9e25,
        2025: 2.5e26,
        2026: 2.5e27,
        2027: 1e28,
        2028: 9e28,
        2031: 2e30,
        2035: 5e31,
    }

    # Speed automation schedules. Mapping effective compute -> speedup

    experiment_design_automation_schedule: Dict[float, float] = {
        9e25: 1,
        1e27: 1.05,
        1e28: 1.5,
        1e30: 4,
        3e31: 10,
        1e33: 100,
        1e35: 1000,
    }

    software_design_automation_schedule: Dict[float, float] = {
        9e25: 1,
        1e27: 1.02,
        1e28: 1.3,
        1e30: 2,
        3e31: 7,
        1e33: 25,
        1e35: 100,
    }

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
    experiment_monitoring_efficiency_automation_schedule: Dict[float, float] = {
        9e25: 1,
        1e27: 0.98,
        1e28: 0.95,
        1e30: 0.85,
        3e31: 0.75,
        1e33: 0.65,
        1e35: 0.5,
        1e50: 0.5,
        # 1e26: 1,
    }

    experiment_analysis_automation_schedule: Dict[float, float] = {
        9e25: 1,
        1e27: 1.05,
        1e28: 2,
        1e30: 4,
        3e31: 7,
        1e33: 20,
        1e35: 80,
    }

    # See https://squigglehub.org/models/takeoff-modeling-stuff/time-length-modeling-options for viewing some relevant distributions
    # Distribution of task times
    # TODO change to 5/95

    # Experiment design
    experiment_design_default_weeks_lognormal_mu: float = 0.2
    experiment_design_default_weeks_lognormal_sigma: float = 1

    # Software design
    software_design_default_weeks_lognormal_mu: float = 0.3
    software_design_default_weeks_lognormal_sigma: float = 1

    # Software programming
    software_programming_default_weeks_lognormal_mu: float = 1
    software_programming_default_weeks_lognormal_sigma: float = 1

    runtime_hardness_correlation_level: float = 0.6  # lower is more correlated

    # Experiment results analysis
    experiment_analysis_default_weeks_lognormal_mu: float = 0.5
    experiment_analysis_default_weeks_lognormal_sigma: float = 1

    # ----- BEGIN VERY IMPORTANT PARAMS -----

    DO_TOP_K_TASTE: bool = True

    # Represents the fraction of experiments that are randomly looked at before choosing the top one
    T_initial: float = 0.0025
    research_taste_automation_schedule: Dict[float, float] = {
        9e25: T_initial,
        1e27: 0.003,
        1e28: 0.005,
        1e30: 0.01,
        3e31: 0.02,
        1e33: 0.04,
        1e35: 0.06,
        2e37: 0.08,
        1e40: 0.1,
        # 10e25: T_initial,
    }

    # Wall clock runtime with default monitoring in weeks, 5th and 95th percentiles
    runtime_default_weeks_p5: float = 1
    runtime_default_weeks_p95: float = 20

    # How many research teams and experimets per team
    RESEARCH_TEAMS: int = 10

    # alg efficiency -> max experiments per team
    max_experiments_per_team_automation_schedule: Dict[float, float] = {
        1: 3,
        1e10: 3,
    }

    # Alg efficieny multiplier distribution. Very sensitive to changes which can screw things up
    # so make sure to check no automation behavior. 5th and 95th percentiles
    # Make sure initial pool of AEMs multipoies to something reasonable
    algorithmic_efficiency_multiplier_p5: float = 0.0001
    algorithmic_efficiency_multiplier_p95: float = 0.015

    # ---- END VERY IMPORTANT PARAMS -----

    N_EXPERIMENTS: int = 10000

    # Stop simulation when there are X experiments left, or have gotten past year Y, or effective compute level Z
    STOP_EXPERIMENTS: int = 0
    STOP_YEAR: float = 2040
    STOP_EC: float = 1e50
    STOP_AE: float = 1e12

    # printing stuff
    PRINT_TIMING: bool = True
    PRINT_EXPERIMENT_INFO: bool = False
    PRINT_TASTE_INFO: bool = False
    PLOT_CORRELATION: bool = False
    PRINT_CHECK_TASTE: bool = False

    # How often to do experiment prioritization
    PRIORITIZATION_CADENCE: int = 50
    DO_AUTOMATION: bool = True
    DO_NO_AUTOMATION_FIRST: bool = True

    # How many simluations to run
    N_SIMS: int = 1


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
        # cumulative_weights[i:] = [w - weights[i] for w in cumulative_weights[i:]]
        # if i + 1 < len(cumulative_weights):
        #     cumulative_weights.pop(i)

        weights.pop(i)
        population.pop(i)
        cumulative_weights = list(
            itertools.accumulate(weights)
        )  # Recalculate cumulative weights

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
    start_runtime: float = -1

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
        time_to_complete = (
            self.pre_run_work_time_in_weeks(speedup_multipliers)
            # + self.runtime_in_weeks(speedup_multipliers)
            + self.post_run_work_time_in_weeks(speedup_multipliers)
        )
        # Copying prioritization advice from ChatGPT https://chat.openai.com/share/3628828c-edd1-4b6c-97d7-9c2470a2bdbe
        self.true_priority = (
            self.algorithmic_efficiency_multiplier - 1
        ) / time_to_complete

    def start(self, year: float, speedup_multipliers: AISpeedupMultipliers):
        self.start_year = year
        self.start_runtime = (
            self.start_year + self.pre_run_work_time_in_weeks(speedup_multipliers) / 52
        )
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
            f"start_year={self.start_year}, "
            f"start_runtime={self.start_runtime}, "
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
    max_experiments: float = 10
    prospective_experiments: List[Experiment] = Field(default_factory=list)
    next_experiments: List[Experiment] = Field(default_factory=list)
    in_progress_experiments: List[Experiment] = Field(default_factory=list)
    research_team_next_free_years: List[float] = Field(default_factory=list)
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
            max_experiments=log_linear_interpolation(
                1, made_up_parameters.max_experiments_per_team_automation_schedule
            )
            * made_up_parameters.RESEARCH_TEAMS,
            research_team_next_free_years=[
                made_up_parameters.initial_year
                for _ in range(made_up_parameters.RESEARCH_TEAMS)
            ],
            **data,
        )

    def _initialize_prospective_experiments(
        self,
        made_up_parameters: MadeUpParameters,
        ai_speedup_multipliers: AISpeedupMultipliers,
    ):
        # Generate N_EXPERIMENTS beta-distributed random variates
        # random_variates = beta.rvs(
        #     made_up_parameters.ae_beta_a,
        #     made_up_parameters.ae_beta_b,
        #     size=made_up_parameters.N_EXPERIMENTS,
        # )

        # # Increment by 1 and ensure each value is at least 1, then raise to the exponent
        # algorithmic_efficiency_multipliers = (
        #     np.maximum(1, random_variates + 1) ** made_up_parameters.ae_beta_exp
        # )

        # # Convert to list if necessary
        # algorithmic_efficiency_multipliers_list = (
        #     algorithmic_efficiency_multipliers.tolist()
        # )
        ae_mu, ae_sigma = p5_and_95_to_mu_and_sigma(
            made_up_parameters.algorithmic_efficiency_multiplier_p5,
            made_up_parameters.algorithmic_efficiency_multiplier_p95,
        )

        algorithmic_efficiency_multipliers = (
            np.random.lognormal(
                mean=ae_mu,
                sigma=ae_sigma,
                size=made_up_parameters.N_EXPERIMENTS,
            )
            + 1
        )

        runtime_mu, runtime_sigma = p5_and_95_to_mu_and_sigma(
            made_up_parameters.runtime_default_weeks_p5,
            made_up_parameters.runtime_default_weeks_p95,
        )
        runtime_default_weeks_values = np.random.lognormal(
            mean=runtime_mu,
            sigma=runtime_sigma,
            size=made_up_parameters.N_EXPERIMENTS,
        )

        # Convert lists to numpy arrays for easier manipulation
        efficiency_array = np.array(algorithmic_efficiency_multipliers)
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
                algorithmic_efficiency_multipliers,
            )

        print(
            f"Total alg efficiency multiplier available: {np.prod(algorithmic_efficiency_multipliers): .2e}"
        )

        experiments = [
            Experiment(
                id=i,
                made_up_parameters=made_up_parameters,
                algorithmic_efficiency_multiplier=algorithmic_efficiency_multipliers[i],
                runtime_default_weeks=runtime_aligned_with_efficiency[i],
            )
            for i in range(made_up_parameters.N_EXPERIMENTS)
        ]
        if made_up_parameters.PRINT_TASTE_INFO:
            self.sweep_taste_and_print_info(
                experiments, ai_speedup_multipliers, made_up_parameters
            )

        print(
            f"top experimernt aems: {[e.algorithmic_efficiency_multiplier for e in sorted(experiments, key=lambda e: -e.algorithmic_efficiency_multiplier)][:10]}"
        )
        return experiments

    def sweep_taste_and_print_info(
        self,
        experiments: List[Experiment],
        ai_speedup_multipliers: AISpeedupMultipliers,
        made_up_parameters: MadeUpParameters,
    ):
        # taste_options = np.arange(0, 200, 20).tolist()
        taste_options = np.arange(0.005, 0.1, 0.01).tolist()
        for e in experiments:
            e.update_true_priority(ai_speedup_multipliers)
        print(
            f"true prio stats {summary_stats_str([e.true_priority for e in experiments])}"
        )
        # experiments = sorted(experiments, key=lambda e: e.true_priority)
        N_TRIALS = 5
        NUM_TO_SAMPLE = MadeUpParameters().PRIORITIZATION_CADENCE
        for taste in taste_options:
            true_prio_means = []
            for _ in range(N_TRIALS):
                sampled_experiments = self.sample_next_experiments(
                    deepcopy(experiments),
                    taste,
                    NUM_TO_SAMPLE,
                    made_up_parameters.DO_TOP_K_TASTE,
                )
                # pdb.set_trace()
                true_prio_means.append(
                    sum([e.true_priority for e in sampled_experiments]) / NUM_TO_SAMPLE
                )
            print(
                f"taste: {taste} N_TRIALS {N_TRIALS} {summary_stats_str(true_prio_means)}"
            )

    def start_next_experiment(self):
        start_time = time.time()
        # print(self.next_experiments)
        if not self.next_experiments:
            self.select_next_experiments()
        next_experiment = self.next_experiments.pop(0)
        next_experiment.start(self.year, self.ai_speedup_multipliers)
        self.in_progress_experiments.append(next_experiment)
        self.started_experiments += 1
        if (self.started_experiments - 1) % (
            self.made_up_parameters.N_EXPERIMENTS / 50
        ) == 0:
            print(
                f"Starting experiment {self.started_experiments} at {self.year}: {next_experiment}]"
            )
        self.in_progress_experiments = sorted(
            self.in_progress_experiments,
            key=lambda e: e.end_runtime,
        )
        research_team_idx = self.research_team_next_free_years.index(
            min(self.research_team_next_free_years)
        )
        self.research_team_next_free_years[research_team_idx] += (
            next_experiment.pre_run_work_time_in_weeks(self.ai_speedup_multipliers) / 52
        )
        self.update_year(min(self.research_team_next_free_years))
        # if self.made_up_parameters.PRINT_TIMING:
        #     print(f"Next eperiment took: {time.time() - start_time:.2f} seconds")

    def sample_next_experiments(
        self,
        experiments: List[Experiment],
        taste,
        num_to_sample,
        do_top_k_taste=None,
    ):
        s = time.time()
        if do_top_k_taste is None:
            do_top_k_taste = self.made_up_parameters.DO_TOP_K_TASTE
        if do_top_k_taste:
            sampled_experiments = []
            for _ in range(num_to_sample):
                to_sample = max(min(int(taste * len(experiments)), len(experiments)), 1)
                # print(to_sample)

                sample = random.sample(
                    experiments,
                    to_sample,
                )

                highest_priority_item = max(sample, key=lambda e: e.true_priority)
                sampled_experiments.append(highest_priority_item)
                experiments.remove(highest_priority_item)
            print(f"Selecting next eperiments took: {time.time() - s:.2f} seconds")
            return sampled_experiments
        else:
            weights = [(e.true_priority + 1) ** taste for e in experiments]
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

        sorted_exps = sorted(
            self.prospective_experiments, key=lambda e: -e.true_priority
        )
        # if self.made_up_parameters.PRINT_TIMING:
        #     print(f"Updating priorites took: {time.time() - start_time:.2f} seconds")
        self.next_experiments = self.sample_next_experiments(
            self.prospective_experiments,
            self.ai_speedup_multipliers.research_taste,
            self.made_up_parameters.PRIORITIZATION_CADENCE,
        )
        sorted_next_exps = sorted(self.next_experiments, key=lambda e: -e.true_priority)
        if self.made_up_parameters.PRINT_CHECK_TASTE:
            print(f"\ntop 10 experiments: {sorted_exps[:10]}\n")
            print(f"top selected experiments: {sorted_next_exps[:10]}\n")
            print(f"bottom 10 experiments: {sorted_exps[-10:]}\n")
        # pdb.set_trace()
        # for e in self.next_experiments:
        #     print(e)
        #     self.prospective_experiments.remove(e)

    def update_year(self, new_year):
        prev_year = self.year
        self.year = new_year
        if math.floor(new_year) > math.floor(prev_year):
            self.print_properties()

    def research_step(self):
        s = time.time()
        if not self.in_progress_experiments and self.prospective_experiments:
            self.start_next_experiment()
        while (
            self.in_progress_experiments[0].end_runtime > self.year
            and self.prospective_experiments
        ):
            num_running_experiments = len(
                [e for e in self.in_progress_experiments if e.start_runtime < self.year]
            )
            # print(num_running_experiments)
            # print(len(self.in_progress_experiments))
            if num_running_experiments >= math.floor(self.max_experiments):
                # print(self.in_progress_experiments)
                self.year = self.in_progress_experiments[0].end_runtime
            else:
                self.start_next_experiment()

        finished_experiment = self.in_progress_experiments.pop(0)
        research_team_idx = self.research_team_next_free_years.index(
            min(self.research_team_next_free_years)
        )
        self.research_team_next_free_years[research_team_idx] = max(
            self.research_team_next_free_years[research_team_idx],
            finished_experiment.end_runtime,
        ) + (
            finished_experiment.post_run_work_time_in_weeks(self.ai_speedup_multipliers)
            / 52
        )
        self.update_year(min(self.research_team_next_free_years))
        self.physical_compute = linear_log_interpolation(
            self.year, self.made_up_parameters.compute_schedule
        )
        self.algorithmic_efficiency *= (
            finished_experiment.algorithmic_efficiency_multiplier
        )
        self.ai_speedup_multipliers.update(
            self.effective_compute, self.made_up_parameters
        )
        self.max_experiments = (
            log_linear_interpolation(
                self.algorithmic_efficiency,
                self.made_up_parameters.max_experiments_per_team_automation_schedule,
            )
            * self.made_up_parameters.RESEARCH_TEAMS
        )
        self.finished_experiments += 1
        # print(
        #     f"fe {finished_experiment} me {self.max_experiments} ipe {self.in_progress_experiments}"
        # )

        # if self.made_up_parameters.PRINT_TIMING:
        #     print(f"research step took {time.time() - s:.2f} seconds")

        if (self.finished_experiments - 1) % (
            self.made_up_parameters.N_EXPERIMENTS / 50
        ) == 0:
            print(
                f"Finished experiment {self.finished_experiments} at {self.year}: {finished_experiment}]"
            )
            self.print_properties()

        if self.made_up_parameters.PRINT_EXPERIMENT_INFO:
            print(
                f"Finished experiment {self.finished_experiments} at {self.year}: {finished_experiment}]"
            )
        # update AI multipliers

    def is_done(self):
        return (
            len(self.prospective_experiments + self.in_progress_experiments)
            <= self.made_up_parameters.STOP_EXPERIMENTS
            or self.year >= self.made_up_parameters.STOP_YEAR
            or self.effective_compute > self.made_up_parameters.STOP_EC
            or self.algorithmic_efficiency > self.made_up_parameters.STOP_AE
        )

    @property
    def effective_compute(self):
        return self.algorithmic_efficiency * self.physical_compute

    def print_properties(self):
        print(
            f"\nyear: {self.year} alg eff {self.algorithmic_efficiency: .2e} eff compute {self.effective_compute: .2e} started_experiments: {self.started_experiments} finished_experiments: {self.finished_experiments} phys compute {self.physical_compute: .2e} ai speedup multipliers: {self.ai_speedup_multipliers}"
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
            "software_programming_multiplier": self.ai_speedup_multipliers.software_programming,
            "experiment_analysis_multiplier": self.ai_speedup_multipliers.experiment_analysis,
            "research_taste": self.ai_speedup_multipliers.research_taste,
        }


class SimResult(BaseModel):
    timesteps: List[Dict] = Field(default_factory=list)
    name: str = "with automation"

    def print_results(self):
        # Convert to DataFrame
        df = pd.DataFrame(self.timesteps)

        # Ensure the DataFrame is sorted by year
        df.sort_values("year", inplace=True)

        # Calculate the difference in 'algorithmic efficiency' and 'year' over 50 rows
        NUM_PERIODS = 100
        df["efficiency_change"] = df["algorithmic efficiency"].diff(periods=NUM_PERIODS)
        df["year_change"] = df["year"].diff(periods=NUM_PERIODS)

        # Calculate the ratio of 'algorithmic efficiency' over 10 rows for the doubling time calculation
        df["efficiency_ratio"] = df["algorithmic efficiency"] / df[
            "algorithmic efficiency"
        ].shift(NUM_PERIODS)

        # Now calculate the doubling time using these 100-row horizon differences
        # Ensure to handle division by zero or log of zero by replacing with NaN or infinity as appropriate
        df["doubling_time_100_row"] = np.where(
            df["efficiency_ratio"] > 0,
            (df["year_change"] * np.log(2)) / np.log(df["efficiency_ratio"]),
            np.nan,  # Use NaN or another placeholder for cases where calculation cannot be performed
        )

        # pdb.set_trace()

        # # Calculate the year difference (Δt)
        # df["delta_t"] = df["year"].diff()

        # # Calculate the ratio of subsequent algorithmic efficiencies
        # df["efficiency_ratio"] = df["algorithmic efficiency"].diff() / df[
        #     "algorithmic efficiency"
        # ].shift(1)

        # # Calculate the instantaneous growth rate r for each interval
        # df["growth_rate"] = np.log(df["efficiency_ratio"] + 1) / df["delta_t"]

        # # Calculate the doubling time Td for each interval
        # df["doubling_time"] = np.log(2) / df["growth_rate"]

        # # Handling the first row which will have NaN values due to the diff operation
        # df.fillna(value={"doubling_time": np.nan}, inplace=True)

        # Convert 'year' to a proper datetime type if needed (optional, for better x-axis formatting)
        # df["year"] = pd.to_datetime(df["year"], format="%Y.%f")

        properties = [
            "algorithmic efficiency",
            "effective_compute",
            "physical_compute",
            "started_experiments",
            "finished_experiments",
            # "software_design_multiplier",
            "software_programming_multiplier",
            "experiment_analysis_multiplier",
            "research_taste",
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
        plt.close()

        plt.figure(figsize=(10, 6))  # Adjust figure size as needed
        ax = plt.gca()  # Get the current Axes instance

        # Plot only the doubling time
        df.plot(
            x="year",
            y="doubling_time_100_row",
            ax=ax,
            label="Doubling Time",
            color="blue",
        )

        # Setting labels and title for clarity
        plt.xlabel("Year")
        plt.ylabel("Doubling Time (years)")
        plt.title("Algorithmic Efficiency Doubling Time Over Years")

        # Setting y-axis scale to linear or log based on the range of doubling times
        # Uncomment the following line if you prefer a logarithmic scale
        # ax.set_yscale('log')

        plt.legend()  # Show legend
        plt.grid(True)  # Add grid for better readability
        plt.tight_layout()  # Adjust layout to make room for the labels
        plt.show()


def process_df(df: pd.DataFrame):
    # Ensure the DataFrame is sorted by year
    df.sort_values("year", inplace=True)

    # Calculate the difference in 'algorithmic efficiency' and 'year' over 50 rows
    NUM_PERIODS = 100
    df["efficiency_change"] = df["algorithmic efficiency"].diff(periods=NUM_PERIODS)
    df["year_change"] = df["year"].diff(periods=NUM_PERIODS)

    # Calculate the ratio of 'algorithmic efficiency' over 10 rows for the doubling time calculation
    df["efficiency_ratio"] = df["algorithmic efficiency"] / df[
        "algorithmic efficiency"
    ].shift(NUM_PERIODS)

    # Now calculate the doubling time using these 100-row horizon differences
    # Ensure to handle division by zero or log of zero by replacing with NaN or infinity as appropriate
    df["doubling_time_100_row"] = np.where(
        df["efficiency_ratio"] > 0,
        (df["year_change"] * np.log(2)) / np.log(df["efficiency_ratio"]),
        np.nan,  # Use NaN or another placeholder for cases where calculation cannot be performed
    )

    return df


class AllSimResults(BaseModel):
    sim_results: List[SimResult] = Field(default_factory=list)

    def print_results(self):
        data_a = self.sim_results[0].timesteps
        data_b = self.sim_results[1].timesteps

        # Convert to DataFrame
        df_a = process_df(pd.DataFrame(data_a))
        df_b = process_df(pd.DataFrame(data_b))

        properties = [
            "algorithmic efficiency",
            "effective_compute",
            "physical_compute",
            # "started_experiments",
            # "finished_experiments",
            # "software_design_multiplier",
            "software_programming_multiplier",
            "experiment_analysis_multiplier",
            # "research_taste",
        ]  # List all properties

        # Initialize a single figure and axis for plotting
        plt.figure(figsize=(12, 8))  # Adjust figure size as needed
        ax = (
            plt.gca()
        )  # Get the current Axes instance on the current figure matching the given keyword args, or create one.

        # Plot each property on the same axis
        for prop in properties:
            df_a.plot(
                x="year",
                y=prop,
                ax=ax,
                label=f"{prop} (no automation)",
                logy=True,
                linestyle="-",
            )
            df_b.plot(
                x="year",
                y=prop,
                ax=ax,
                label=f"{prop} (automation)",
                logy=True,
                linestyle="-",
            )

        # Set labels and title
        plt.xlabel("Year")
        plt.ylabel("Value")
        plt.title("Properties Over Time")

        # Optionally adjust the x-axis to cover only the years mentioned, if necessary
        min_year, max_year = df_a["year"].min(), max(
            df_a["year"].max(), df_b["year"].max()
        )
        ax.set_xlim(left=min_year, right=max_year)

        plt.legend()  # Show legend to identify each line
        plt.tight_layout()
        plt.show()
        plt.close()

        plt.figure(figsize=(10, 6))  # Adjust figure size as needed
        ax = plt.gca()  # Get the current Axes instance

        # Plot only the doubling time
        df_a.plot(
            x="year",
            y="doubling_time_100_row",
            ax=ax,
            label=f"Doubling time (no automation)",
            linestyle="-",
        )
        df_b.plot(
            x="year",
            y="doubling_time_100_row",
            ax=ax,
            label=f"Doubling (automation)",
            linestyle="-",
        )

        # Setting labels and title for clarity
        plt.xlabel("Year")
        plt.ylabel("Doubling Time (years)")

        plt.title("Algorithmic Efficiency Doubling Time Over Years")

        plt.legend()  # Show legend
        plt.grid(True)  # Add grid for better readability
        plt.tight_layout()  # Adjust layout to make room for the labels
        plt.show()


def main():
    random.seed(RANDOM_SEED)

    s = time.time()
    all_sim_results = AllSimResults()

    if MadeUpParameters().DO_NO_AUTOMATION_FIRST:
        made_up_parameters = MadeUpParameters(DO_AUTOMATION=False)
        sim_result = SimResult(name="no automation")
        world_state = WorldState(
            made_up_parameters,
            AISpeedupMultipliers(research_taste=made_up_parameters.T_initial),
        )
        sim_result.timesteps.append(world_state.get_properties_dict())

        while not world_state.is_done():
            world_state.research_step()
            s = time.time()
            # TODO only add the data that I want to collect, whole thing is wha'ts being slow
            sim_result.timesteps.append(world_state.get_properties_dict())

        # sim_result.print_results()
        all_sim_results.sim_results.append(deepcopy(sim_result))

    for _ in tqdm(range(MadeUpParameters().N_SIMS)):
        made_up_parameters = MadeUpParameters()
        sim_result = SimResult()
        world_state = WorldState(
            made_up_parameters,
            AISpeedupMultipliers(research_taste=made_up_parameters.T_initial),
        )
        sim_result.timesteps.append(world_state.get_properties_dict())

        while not world_state.is_done():
            world_state.research_step()
            s = time.time()
            # TODO only add the data that I want to collect, whole thing is wha'ts being slow
            sim_result.timesteps.append(world_state.get_properties_dict())

        # sim_result.print_results()
        all_sim_results.sim_results.append(deepcopy(sim_result))

    print(
        f"Ran {made_up_parameters.N_SIMS} simluations in {round(time.time() - s, 2)} seconds\n"
    )
    all_sim_results.print_results()


if __name__ == "__main__":
    main()
