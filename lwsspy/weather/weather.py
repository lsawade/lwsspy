import numpy as np
from typing import Union, Iterable
from datetime import datetime


class weather:

    def __init__(self, dates: list = [],
                 minTemp: np.ndarray = np.zeros(0),
                 maxTemp: np.ndarray = np.zeros(0),
                 rain: np.ndarray = np.zeros(0),
                 relativeHumidity: np.ndarray = np.zeros(0),
                 unitTemp: str = r'$^\circ$',
                 unitRain: str = r'$^\circ$'):
        """Forecast class that contains everything you need for the weather.

        Args:
            dates (list, optional): 
                Datetime object list. Defaults to [].
            minTemp (``numpy.ndarray``, optional):
                Array of minimum temperatures. Defaults to ``np.zeros(0)``.
            maxTemp (``numpy.ndarray``, optional):
                Array of maximum temperatures. Defaults to ``np.zeros(0)``.
            rain (``numpy.ndarray``, optional):
                Array of rainfall values. Defaults to ``np.zeros(0)``.
            unitTemp (str, optional): [description]. Defaults to ``r'$^\circ$'``.
            unitRain (st, optional): [description]. Defaults to ``r'$^\circ$'``.

        Returns:
            ``weather`` ``Class``.
        """
        self.dates = dates
        self.minTemp = minTemp
        self.maxTemp = maxTemp
        self.meanTemp = (maxTemp + minTemp)/2
        self.rain = rain
        self.unitTemp = unitTemp
        self.unitRain = unitRain

    def __getitem__(self, ind: Union[Iterable, int, np.ndarray]):
        if type(ind) is int:
            ind = [ind]
        return [[self.dates[_i],
                 self.minTemp[_i], self.maxTemp[_i], self.rain[_i]]
                for _i in ind]

    def addDate(self, date: datetime, minTemp: float, maxTemp: float,
                rain: float):
        # Add date and data
        self.dates = np.append(self.dates, date)
        self.minTemp = np.append(self.minTemp, minTemp)
        self.maxTemp = np.append(self.maxTemp, maxTemp)
        self.meanTemp = np.append(self.meanTemp, (maxTemp + minTemp)/2)
        self.rain = np.append(self.rain, rain)

    def __str__(self):
        string = ""
        for key, value in self.__dict__.items():
            string += f"{key}:\n{value}\n"

        return string