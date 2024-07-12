import numpy as np
from scipy.stats import pearsonr


class Evaluate:
    @staticmethod
    def nash_sutcliffe(observed, simulated):
        return 1 - (np.sum((observed - simulated) ** 2) / np.sum((observed - np.mean(observed)) ** 2))

    @staticmethod
    def percent_bias(observed, simulated):
        return 100 * np.sum(observed - simulated) / np.sum(observed)

    @staticmethod
    def root_mean_squared_error(observed, simulated):
        return np.sqrt(np.mean((observed - simulated) ** 2))

    @staticmethod
    def kling_gupta_efficiency(observed, simulated):
        r, _ = pearsonr(observed, simulated)
        alpha = np.std(simulated) / np.std(observed)
        beta = np.mean(simulated) / np.mean(observed)
        kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        return kge