import os
import json
import requests
from typing import Union
from datetime import datetime
import numpy as np
from .weather import weather


class requestweather:
    """
    This class requires a bit of a setup. To make this work you will have to

    1. Setup an Accuweather Developer account (Limited Trial, which let's you
       request data 50 times a day for free.)
    2. Setup an app under the `My Apps` tab.
        a. Choose limited trial
        b. Weather app and
        c. Python as language
        d. after creating the app get the API key that is displayed when
           clicking on the app.
    3. Then you can use the class to get both the location key for your lat
       lon (it will provide you the closets station) and get the forecast.

    Last modified: Lucas Sawade, 2020.09.22 12.00 (lsawade@princeton.edu)
    """

    def __init__(self,
                 lat: Union[float, None] = 40.351978,
                 lon: Union[float, None] = -74.660744,
                 apikey: str = "x0UHCli9qKjkJzxHRDwRa9FWmn4XW31h",
                 locationkey: Union[str, None] = '339523',
                 historical: Union[bool, None] = None,
                 city: Union[str, None] = 'Princeton',
                 countrycode: Union[str, None] = 'US',
                 forecast: Union[weather, None] = None,
                 default_run: bool = False):
        """Creates request weather class to request weather forecast from
        the accuweather website.

        Args:
            lat (Union[float, None], optional):
                latitude location. Defaults to 40.351978.
            lon (Union[float, None], optional):
                longitude location. Defaults to -74.660744.
            apikey (str, optional):
                API access key for accuweather.
                Defaults to "x0UHCli9qKjkJzxHRDwRa9FWmn4XW31h".
            locationkey (Union[str, None], optional):
                locationkey.Defaults to '339523'.
            historical (Union[bool, None], optional):
                To be used to get request historical data, but not implemeted
                yet. Defaults to None.
            city (Union[str, None], optional):
                City string. Defaults to 'Princeton'.
            countrycode (Union[str, None], optional): Countrycode string.
                Defaults to 'US'.
            default_run (bool, optional):
                If set to true the program will run the default set of
                parameters to create a plot. Defaults to False.

        Raises:
            ValueError: [description]
        """

        # Assign the parameters.
        self.lat = lat
        self.lon = lon
        self.apikey = apikey
        self.locationkey = locationkey
        self.historical = historical
        self.countrycode = countrycode
        self.city = city

        if forecast is None:
            self.forcecast = weather([], np.zeros(0), np.zeros(0), np.zeros(0))
        else:
            self. forecast = forecast

        # Through error if neither location or location key is given.
        if ((self.lat is None) or (self.lon is None)) \
                and self.locationkey is None \
                and ((self.city is None) or (self.countrycode is None)):
            raise ValueError("Either location or location key have to be "
                             "provided.")

        # Get Location key if not given
        if self.locationkey is None:
            self._get_location()

        if default_run:
            self._default()

    def _default(self):
        print(self.forcecast)
        # Get forecast for 5 days:
        fc = self.get_forecast(5)
        print(self.forcecast)

        # Plot stuff
        import matplotlib.pyplot as plt
        from ..plot_util.updaterc import updaterc
        from .. import DOCFIGURES
        updaterc()

        fig = plt.figure(figsize=(7, 5))

        # Plot temperature
        axTemp = plt.gca()
        color = 'tab:red'
        axTemp.fill_between(fc.dates, fc.minTemp,
                            fc.maxTemp, interpolate=True, alpha=0.25,
                            color='gray', label="Temp.-Range")
        axTemp.plot(fc.dates, fc.meanTemp, color=color)
        plt.xlabel('Date')
        fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
        plt.ylabel('Temperature [C]')
        axTemp.tick_params(axis='y', labelcolor=color)
        plt.axis('tight')
        plt.grid('off')

        # # Plot Rain Fall
        axRain = axTemp.twinx()
        color = 'tab:blue'
        axRain.bar(fc.dates, fc.rain, width=10, color=color)
        axRain.tick_params(axis='y', labelcolor=color)
        plt.ylabel('Rain [mm]')
        plt.ylim(bottom=0.0)
        plt.grid('off')
        fig.legend()

        plt.savefig(os.path.join(DOCFIGURES, 'requestweather.svg'), dpi=300)
        plt.show()

    def _get_location(self):
        """Uses location and location search api url to get the location key
        of the closest weather station to the given location. If geolocation
        is given when class is initiated, geolocation will be prioritized and
        city and country variables will be overwritten
        """

        # Check method of search for location
        if ((self.lat is not None) and (self.lon is not None)):

            self.geolocation_search_url = \
                "http://dataservice.accuweather.com/locations/v1/cities/"\
                f"geoposition/search?" \
                f"apikey={self.apikey}&q={self.lat},{self.lon}"

            # Send request
            try:
                r = requests.get(self.geolocation_search_url)
            except requests.exceptions.RequestException as e:
                raise SystemExit(e)

            location_dict = json.loads(r.content)

            self.locationkey = location_dict["Key"]
            self.city = location_dict["LocalizedName"]
            self.countrycode = location_dict["country"]["ID"]

        elif ((self.lat is None) or (self.lon is None)) \
                and ((self.city is not None)
                     and (self.countrycode is not None)):

            self.city_search_url = \
                f"http://dataservice.accuweather.com/locations/v1/cities/" \
                f"{self.countrycode}/search?" \
                f"apikey={self.apikey}&q={self.city}&details=true"\

            # Send request
            try:
                r = requests.get(self.city_search_url)
            except requests.exceptions.RequestException as e:
                raise SystemExit(e)

            # Read list
            location_list = json.loads(r.content)

            # If there are multiple choice in city choose from list via input.
            if len(location_list) > 1:

                print("Choose from the following list:\n")
                for _i, city in enumerate(location_list):
                    print(f"  {_i+1:>3}. {city['LocalizedName']}, "
                          f"{city['PrimaryPostalCode']}, "
                          f"{city['AdministrativeArea']['ID']}, "
                          f"{city['Country']['ID']}")

                checklist = [i+1 for i in range(len(location_list))]
                choice = int(input("Choose: "))
                while (choice not in checklist):
                    choice = int(input("Choose from the above cities by "
                                       "there number: "))
                print("Choice: ", choice)

            self.locationkey = location_list[choice - 1]["Key"]
            self.lat = location_list[choice - 1]['GeoPosition']['Latitude']
            self.lon = location_list[choice - 1]['GeoPosition']['Longitude']

        else:
            raise ValueError("Need either full geolocation or full "
                             "City, Country, Combination")

    def get_forecast(self, days: int):
        """Gets forcast for a given number of days using the found or given
        location key.

        Args:
            days (int): Number of days to forecast (must be in[1, 5, 10, 15])

        Returns:
            populates ``self.forecast``
        """
        # Check if days value possible, otherwise through error
        if days not in [1, 5]:
            raise ValueError("Only 1, 5 day forecasts "
                             "available for free account.")

        # Define URL
        self.forecast_url = \
            f"http://dataservice.accuweather.com/forecasts/v1/" \
            f"daily/{days}day/{self.locationkey}" \
            f"?apikey={self.apikey}" \
            f"&details=true&metric=true"

        # Request data
        try:
            r = requests.get(self.forecast_url)
        except requests.exceptions.RequestException as e:
            raise SystemExit(e)

        # Read list
        forecast_dict = json.loads(r.content)

        # Get daily forcecasts
        try:
            dailyforecasts = forecast_dict["DailyForecasts"]
        except KeyError as e:
            print(forecast_dict)
            raise SystemExit(e)

        dates = []
        minTemp = []
        maxTemp = []
        rain = []
        for _fcast in dailyforecasts:
            dates.append(datetime.fromisoformat(_fcast["Date"]))
            minTemp.append(_fcast["Temperature"]["Minimum"]["Value"])
            maxTemp.append(_fcast["Temperature"]["Maximum"]["Value"])
            rain.append(_fcast["Day"]["Rain"]["Value"])

        self.forecast = weather(dates=dates, minTemp=np.array(minTemp),
                                maxTem=np.array(maxTemp),
                                rain=np.array(rain))
        return self.forcecast

    def _check_error_code(self, status: int):
        """Checks Accuweather request code and throws error if request is bad.

        Args:
            status (int): error code
        """

        if status != 200:
            print(f"Request Error: {status}")
            if status == 400:
                raise ValueError('Request had bad syntax or the '
                                 'parameters supplied were invalid.')
            elif status == 401:
                raise ValueError('Unauthorized. API authorization failed.')
            elif status == 403:
                raise ValueError('Unauthorized. You do not have permission '
                                 'to access this endpoint.')
            elif status == 404:
                raise ValueError('Server has not found a route matching the '
                                 'given URI.')
            elif status == 500:
                raise ValueError('Server encountered an unexpected condition '
                                 'which prevented it from fulfilling the '
                                 'request.')
        return None

    def __str__(self):

        if self.lat is None:
            lat = 'None'
        else:
            lat = self.lat
        if self.lon is None:
            lon = 'None'
        else:
            lon = self.lon
        if self.city is None:
            city = 'None'
        else:
            city = self.city
        if self.countrycode is None:
            countrycode = 'None'
        else:
            countrycode = self.countrycode
        string = "WeatherForecast:\n"
        string += f"    Location Key:_____{self.locationkey:_>35}\n"
        string += f"    Lat:______________{lat:_>35}\n"
        string += f"    Lon:______________{lon:_>35}\n"
        string += f"    City:_____________{city:_>35}\n"
        string += f"    Country Code:_____{countrycode:_>35}\n"
        string += f"    API Key:__________{self.apikey:_>35}"

        return string
