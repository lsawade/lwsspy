# External
import csv
import numpy as np
import os.path as p
from typing import Union
from datetime import datetime

# Internal
from .weather import weather


def drop2pickle(filename: Union[str, None] = None,
                outfilename: Union[str, None] = None,
                default_run: bool = False):
    """Takes in an exported Kestrel file and converts it to a weather object.

    Args:
        filename (Union[str, None], optional):
            exported kestrel file. Defaults to None.
        outfilename (Union[str, None], optional):
            outputfilename. Defaults to None, then
            <filename>.csv --> <filename>.p
        default_run (bool, optional):
            Default file is called from ``data`` dir. Defaults to False.

    Returns:
        weather object.
    """

    if default_run or (type(filename) is None):
        filename = p.join(p.dirname(p.abspath(__file__)), 'data',
                          'export_lsawade_2020_9_8_17_5_27.csv')
    if outfilename is None:
        outfilename = p.basename(filename.split(".")[0] + ".p")

    with open(filename, 'r') as f:
        c = csv.reader(f, delimiter=',')

        for line_count, row in enumerate(c):
            if line_count == 0:
                devicename = row[1]
                print(f"Device Name: {devicename}")
            elif line_count == 1:
                devicemodel = row[1]
                print(f"Device Model: {devicemodel}")
            elif line_count == 2:
                serialno = row[1]
                print(f"Serial Number: {serialno}")
            elif line_count == 3:
                valdict = {}
                keylist = {}
                for _i, column in enumerate(row):
                    keylist[_i] = column
                    valdict[keylist[_i]] = {"values": [], "unit": ""}
            elif line_count == 4:
                for _i, unit in enumerate(row):
                    valdict[keylist[_i]]["unit"] = unit
            else:
                for _i, measurement in enumerate(row):
                    valdict[keylist[_i]]["values"].append(measurement)

                line_count += 1

    # Fix arrays
    for key, dd in valdict.items():
        if key == "FORMATTED DATE-TIME":
            dd["values"] = [datetime.fromisoformat(date)
                            for date in dd["values"]]
        else:
            dd["values"] = np.array(dd["values"])

    w = weather(
        dates=valdict["FORMATTED DATE-TIME"]["values"],
        temperature=valdict["Temperature"]["values"],
        relativeHumidity=valdict["Relative Humidity"]["values"],
        heatStressIndex=valdict["Heat Stress Index"]["values"],
        dewPoint=valdict["Dew Point"]["values"],
        wetBulbTemperature=valdict["Wet Bulb Temperature"]["values"],
        stationPressure=valdict["Station Pressure"]["values"],
        unitTemp=valdict["Temperature"]["unit"],
        unitRelativeHumidity=valdict["Relative Humidity"]["unit"],
        unitHeatStressIndex=valdict["Heat Stress Index"]["unit"],
        unitDew=valdict["Dew Point"]["unit"],
        unitWetBulbTemp=valdict["Wet Bulb Temperature"]["unit"],
        unitStationPressure=valdict["Station Pressure"]["unit"],
        deviceModel=devicemodel, deviceName=devicename, serialNumber=serialno)

    if p.exists(outfilename):
        print("Outputfile exists already, delete before writing.")
    else:
        w.save(filename=outfilename)
    return weather
