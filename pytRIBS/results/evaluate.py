import numpy as np
from scipy.stats import pearsonr


class Evaluate:
    """
    A collection of static methods for evaluating the performance of simulated data against observed data.

    """
    @staticmethod
    def nash_sutcliffe(observed, simulated):
        """
        Calculate the Nash-Sutcliffe efficiency coefficient.

        The Nash-Sutcliffe efficiency (NSE) is a normalized statistic that determines the relative magnitude of the residual variance
        compared to the measured data variance. It ranges from -∞ to 1, with 1 indicating a perfect match between observed and simulated values.

        Parameters
        ----------
        observed : numpy.ndarray
            Array of observed data values.
        simulated : numpy.ndarray
            Array of simulated data values.

        Returns
        -------
        float
            The Nash-Sutcliffe efficiency coefficient.
        """
        return 1 - (np.sum((observed - simulated) ** 2) / np.sum((observed - np.mean(observed)) ** 2))

    @staticmethod
    def percent_bias(observed, simulated):
        """
        Calculate the percent bias.

        The percent bias (PBIAS) measures the average tendency of the simulated data to be larger or smaller than the observed data.
        Positive values indicate model underestimation, while negative values indicate model overestimation.

        Parameters
        ----------
        observed : numpy.ndarray
            Array of observed data values.
        simulated : numpy.ndarray
            Array of simulated data values.

        Returns
        -------
        float
            The percent bias.
        """
        return 100 * np.sum(observed - simulated) / np.sum(observed)

    @staticmethod
    def root_mean_squared_error(observed, simulated):
        """
        Calculate the root mean squared error (RMSE).

        RMSE measures the square root of the average squared differences between observed and simulated values. It provides a measure
        of how well the model predicts the observed data.

        Parameters
        ----------
        observed : numpy.ndarray
            Array of observed data values.
        simulated : numpy.ndarray
            Array of simulated data values.

        Returns
        -------
        float
            The root mean squared error.
        """
        return np.sqrt(np.mean((observed - simulated) ** 2))

    @staticmethod
    def kling_gupta_efficiency(observed, simulated):
        """
        Calculate the Kling-Gupta efficiency (KGE).

        The Kling-Gupta efficiency is a metric that evaluates model performance based on correlation, variability, and bias. It ranges
        from -∞ to 1, with 1 indicating perfect model performance.

        Parameters
        ----------
        observed : numpy.ndarray
            Array of observed data values.
        simulated : numpy.ndarray
            Array of simulated data values.

        Returns
        -------
        float
            The Kling-Gupta efficiency coefficient.
        """
        r, _ = pearsonr(observed, simulated)
        alpha = np.std(simulated) / np.std(observed)
        beta = np.mean(simulated) / np.mean(observed)
        kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        return kge