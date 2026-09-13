from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class BaseVaRModel(ABC):
    """
    Abstract blueprint for all VaR models.
    """
    def __init__(self, confidence_level: float = 0.99, portfolio_value: float = 1.0):
        self.alpha = confidence_level
        self.portfolio_value = portfolio_value

    @abstractmethod
    def calculate(self, returns: pd.DataFrame, weights: np.ndarray) -> dict:
        """
        Every VaR model MUST implement this method.
        
        :param returns: DataFrame of historical log returns
        :param weights: Numpy array of portfolio weights
        :return: A dictionary containing {'VaR': float, 'ES': float}
        """
        pass