import random
import numpy as np
import statistics
from pydantic import BaseModel, Field
import time
from tqdm import tqdm
from scipy.stats import beta
from typing import Optional, Literal, List, Dict


class MadeUpParameters(BaseModel):
    initial_year: float = 2024.0
    compute_schedule: Dict[float, float] = {
        2024.0: 9e25,
        2025: 2.5e26,
        2026: 2.5e27,
        2027: 1e28,
        2028.0: 9e28,
    }

    # Experiment design
    experiment_design_default_weeks_lognormal_mean: float = 1
    experiment_design_default_weeks_lognormal_stdev: float = 1

    # TODO: make real. effective compute -> speedup
    experiment_design_automation_schedule: Dict[float, float] = {
        9e25: 1,
        9e30: 100,
        9e35: 100000,
    }

    # Software design
    software_design_default_weeks_lognormal_mean: float = 1
    software_design_default_weeks_lognormal_stdev: float = 1

    # TODO: make real. effective compute -> speedup
    software_design_automation_schedule: Dict[float, float] = {
        9e25: 1,
        9e30: 100,
        9e35: 100000,
    }

    # Software programming
    software_programming_default_weeks_lognormal_mean: float = 1
    software_programming_default_weeks_lognormal_stdev: float = 1

    # TODO: make real. effective compute -> speedup
    software_programming_automation_schedule: Dict[float, float] = {
        9e25: 1,
        9e30: 100,
        9e35: 100000,
    }

    initial_experiment_monitoring_efficiency: float = 1
    # Experiment monitoring
    experiment_monitoring_efficiency_automation_schedule: Dict[float, float] = {
        9e25: 1,
        9e30: 0.7,
        9e35: 0.4,
    }

    # Wall clock runtime with default monitoring
    runtime_default_weeks_lognormal_mean: float = 1
    runtime_default_weeks_lognormal_stdev: float = 1

    # Experiment results analysis
    experiment_analysis_default_weeks_lognormal_mean: float = 1
    experiment_analysis_default_weeks_lognormal_stdev: float = 1

    # TODO: make real. effective compute -> speedup
    experiment_analysis_automation_schedule: Dict[float, float] = {
        9e25: 1,
        9e30: 100,
        9e35: 100000,
    }

    # Research taste. TODO make real
    T_initial: float = 0.2
    research_taste_automation_schedule: Dict[float, float] = {
        9e25: 0.2,
        9e30: 0.4,
        9e35: 1,
    }

    # AE distribution. Got from playing in Squiggle
    ae_beta_a: float = 1
    ae_beta_b: float = 20
    ae_beta_exp: float = 0.1

    N_EXPERIMENTS: int = 10 * 1000

    # How many simluations to run
    N_SIMS: int = 1

    # Stop simulation when there are X experiments left, or have gotten past year Y, or effective compute level Z
    STOP_EXPERIMENTS: int = 0
    STOP_YEAR: float = 3000
    STOP_EC: float = 1e50


class AISpeedupMultipliers(BaseModel):
    experiment_design: float = 1
    software_design: float = 1
    software_programming: float = 1
    experiment_monitoring_efficiency: float = 1
    experiment_analysis: float = 1

    def update(self, effective_compute: float, made_up_parameters: MadeUpParameters):
        self.experiment_design = log_log_interpolation(
            effective_compute, made_up_parameters.experiment_design_automation_schedule
        )
        self.software_design = log_log_interpolation(
            effective_compute, made_up_parameters.software_design_automation_schedule
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

    # Perform linear interpolation on the log of the keys
    return np.interp(x, np.log10(keys), values)


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

    # Perform linear interpolation on the log of both keys and values
    return 10 ** np.interp(np.log10(x), np.log10(keys), np.log10(values))


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

    # Perform linear interpolation on the keys and exponential transformation on the result
    interpolated_log_value = np.interp(x, keys, log_values)
    return 10**interpolated_log_value


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

    def __init__(
        self,
        made_up_parameters: MadeUpParameters,
        **data,
    ):
        super().__init__(
            made_up_parameters=made_up_parameters,
            experiment_design_speed_in_weeks=np.random.lognormal(
                made_up_parameters.experiment_design_default_weeks_lognormal_mean,
                made_up_parameters.experiment_design_default_weeks_lognormal_stdev,
            ),
            software_design_speed_in_weeks=np.random.lognormal(
                made_up_parameters.software_design_default_weeks_lognormal_mean,
                made_up_parameters.software_design_default_weeks_lognormal_stdev,
            ),
            software_programming_speed_in_weeks=np.random.lognormal(
                made_up_parameters.software_programming_default_weeks_lognormal_mean,
                made_up_parameters.software_programming_default_weeks_lognormal_stdev,
            ),
            runtime_default_weeks=np.random.lognormal(
                made_up_parameters.runtime_default_weeks_lognormal_mean,
                made_up_parameters.runtime_default_weeks_lognormal_stdev,
            ),
            experiment_analysis_speed_in_weeks=np.random.lognormal(
                made_up_parameters.experiment_analysis_default_weeks_lognormal_mean,
                made_up_parameters.experiment_analysis_default_weeks_lognormal_stdev,
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

    def end_runtime(self, speedup_multipliers: AISpeedupMultipliers):
        return (
            self.start_year
            + (
                self.pre_run_work_time_in_weeks(speedup_multipliers)
                + self.post_run_work_time_in_weeks(speedup_multipliers)
            )
            / 52
        )


class WorldState(BaseModel):
    idx: int = 0
    year: float
    made_up_parameters: MadeUpParameters
    ai_speedup_multipliers: AISpeedupMultipliers
    algorithmic_efficiency: float = 1
    physical_compute: float = None
    prospective_experiments: List[Experiment] = Field(default_factory=list)
    in_progress_experiments: List[Experiment] = Field(default_factory=list)
    # todo experiment monitoring

    def __init__(
        self,
        made_up_parameters: MadeUpParameters,
        **data,
    ):
        super().__init__(
            physical_compute=made_up_parameters.compute_schedule[2024],
            year=made_up_parameters.initial_year,
            prospective_experiments=self._initialize_prospective_experiments(
                made_up_parameters
            ),
            made_up_parameters=made_up_parameters,
            **data,
        )

    def _initialize_prospective_experiments(self, made_up_parameters: MadeUpParameters):
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

        print(
            f"Total alg efficiency multiplier avaliable: {np.prod(algorithmic_efficiency_multipliers_list)}"
        )
        return [
            Experiment(
                id=i,
                made_up_parameters=made_up_parameters,
                algorithmic_efficiency_multiplier=algorithmic_efficiency_multipliers_list[
                    i
                ],
            )
            for i in range(made_up_parameters.N_EXPERIMENTS)
        ]

    @property
    def research_taste(self):
        return log_linear_interpolation(
            self.effective_compute,
            self.made_up_parameters.research_taste_automation_schedule,
        )

    def next_experiment(self):
        weights = [
            index**self.research_taste
            for index in range(1, len(self.prospective_experiments) + 1)
        ]
        selected_experiment = random.choices(
            self.prospective_experiments, weights=weights, k=1
        )[0]
        selected_experiment.start_year = self.year
        self.prospective_experiments.remove(selected_experiment)
        self.in_progress_experiments.append(selected_experiment)
        self.in_progress_experiments = sorted(
            self.in_progress_experiments,
            key=lambda e: e.end_runtime(self.ai_speedup_multipliers),
        )
        self.year += (
            selected_experiment.pre_run_work_time_in_weeks(self.ai_speedup_multipliers)
            / 52
        )

    def research_step(self):
        if self.idx % 100 == 0:
            for e in self.prospective_experiments:
                e.update_true_priority(self.ai_speedup_multipliers)
            self.prospective_experiments = sorted(
                self.prospective_experiments, key=lambda e: e.true_priority
            )
            self.print_properties()
        if not self.in_progress_experiments:
            self.next_experiment()
        while (
            self.in_progress_experiments[0].end_runtime(self.ai_speedup_multipliers)
            > self.year
        ):
            self.next_experiment()

        finished_experiment = self.in_progress_experiments.pop(0)
        self.year += (
            finished_experiment.post_run_work_time_in_weeks(self.ai_speedup_multipliers)
            / 52
        )
        self.physical_compute = linear_log_interpolation(
            self.year, self.made_up_parameters.compute_schedule
        )
        self.algorithmic_efficiency *= (
            finished_experiment.algorithmic_efficiency_multiplier
        )
        self.ai_speedup_multipliers.update(
            self.effective_compute, self.made_up_parameters
        )
        self.idx += 1

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

    def print_properties(self):
        print(
            f"year: {self.year} end alg eff {self.algorithmic_efficiency} end ai speedup multipliers: {self.ai_speedup_multipliers}"
        )


class SimResult(BaseModel):
    timesteps: List[WorldState] = Field(default_factory=list)

    def print_results(self):
        print(
            f"end year: {self.timesteps[-1].year} end alg eff {self.timesteps[-1].algorithmic_efficiency} end ai speedup multipliers: {self.timesteps[-1].ai_speedup_multipliers}"
        )


class AllSimResults(BaseModel):
    sim_results: List[SimResult] = Field(default_factory=list)

    def print_results(self):
        return
        for sim_result in self.sim_results:
            sim_result.print_results()


def main():
    s = time.time()
    all_sim_results = AllSimResults()

    for _ in tqdm(range(MadeUpParameters().N_SIMS)):
        made_up_parameters = MadeUpParameters()
        sim_result = SimResult()
        world_state = WorldState(
            made_up_parameters, ai_speedup_multipliers=AISpeedupMultipliers()
        )
        sim_result.timesteps.append(world_state)

        while not world_state.is_done():
            world_state.research_step()
            sim_result.timesteps.append(world_state)

        sim_result.print_results()
        all_sim_results.sim_results.append(sim_result)

    print(
        f"Ran {made_up_parameters.N_SIMS} simluations in {round(time.time() - s, 2)} seconds\n"
    )
    all_sim_results.print_results()


if __name__ == "__main__":
    main()
