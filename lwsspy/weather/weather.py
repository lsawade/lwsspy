import numpy as np
from typing import Union, Iterable
from datetime import datetime
import pickle


class weather:

    def __init__(self, dates: list = [],
                 temperature: np.ndarray = np.zeros(0),
                 relativeHumidity: np.ndarray = np.zeros(0),
                 heatStressIndex: np.ndarray = np.zeros(0),
                 dewPoint: np.ndarray = np.zeros(0),
                 wetBulbTemperature: np.ndarray = np.zeros(0),
                 stationPressure: np.ndarray = np.zeros(0),
                 rain: np.ndarray = np.zeros(0),
                 deviceName: str = "", deviceModel: str = "",
                 serialNumber: str = "",
                 unitTemp: str = r'$^\circ C$',
                 unitRelativeHumidity: str = "%",
                 unitHeatStressIndex: str = r"$^\circ C$",
                 unitRain: str = 'mm',
                 unitDew: str = r'$^\circ C$',
                 unitWetBulbTemp: str = r'$^\circ C$',
                 unitStationPressure: str = r'mb',
                 ):
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
            unitTemp (str, optional):
                [description]. Defaults to ``r'$^\\circ$'``.
            unitRain (st, optional):
                [description]. Defaults to ``r'$^\\circ$'``.

        Returns:
            ``weather`` ``Class``.

        Last modified: Lucas Sawade, 2020.09.22 12.00 (lsawade@princeton.edu)
        """

        self.dates = dates
        self.temperature = temperature
        self.relativeHumidity = relativeHumidity
        self.heatStressIndex = heatStressIndex
        self.dewPoint = dewPoint
        self.wetBulbTemperature = wetBulbTemperature
        self.stationPressure = stationPressure
        self.rain = rain
        self.deviceName = deviceName
        self.serialNumber = serialNumber
        self.unitTemp = unitTemp
        self.unitRelativeHumidity = unitRelativeHumidity
        self.unitHeatStressIndex = unitHeatStressIndex
        self.unitRain = unitRain
        self.unitDew = unitDew
        self.unitWetBulbTemp = unitWetBulbTemp
        self.unitStationPressure = unitStationPressure

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

    def save(self, filename: str):
        """Save weather to pickle

        Args:
            filename (str): outputfilename
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(self, filename: str):
        """Load class from pickle

        Args:
            filename (str): input filename
        """

        with open(filename, "rb") as f:
            self = pickle.load(f)

        return self

    def __str__(self):
        string = ""
        for key, value in self.__dict__.items():
            if type(value) is np.ndarray:
                value = f"{value.shape} ndarray"
            elif type(value) is list:
                value = f"{len(value)} list"
            string += f"{key:>22}:{value:_>18}\n"

        return string

    def __repr__(self):
        return self.__str__()
