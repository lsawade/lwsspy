import numpy as np


class Calculator:
    """

    All the caluclations are done the same way. You have set of percentages with
    respect to flour, where flour is 100% weight percent. Then, for a given
    style, say Neapolitan, the weights are computed using the following method.
    For Neapolitan we have the following percentages:

        flour:  100.0 %
        salt:     3.0 %
        yeast:    0.2 %
        water:   65.0 % (of course variable)

    factor = 1/(flour + salt + water + yeast) * weight of doughball * N doughballs

    ingredients = {
        flour = 100.0 * factor,
        salt  =   3.0 * factor,
        yeast =   0.2 * factor,
        water =  65.0 * factor
    }

    If we want to make 1 doughball that weighs 250 gr this results in:

    amounts = {
        flour = 149.0 gr,
        salt  =   4.5 gr,
        yeast =   0.3 gr
        water =  97.0 ml
    }

    Values from:

        stadlermade.com


    Half Active dry yeast as compared to fresh yeast value.

    """

    def __init__(self, style='neapolitan', N=1, doughball=235, watercontent=55,
                 custom: dict = None):

        self.style = style.lower()
        self.custom = custom
        self.N = N
        self.doughball = doughball
        self.watercontent = watercontent

    def reset(self):

        self.amounts = dict(
            flour=0.0,
            water=0.0,
            salt=0.0,
            yeast=0.0,
            oil=0.0,
            sugar=0.0,
            starter=0.0
        )

        self.percentages = dict(
            flour=0.0,
            water=0.0,
            salt=0.0,
            yeast=0.0,
            oil=0.0,
            sugar=0.0,
            starter=0.0,
        )

    def __check_custom_dict__(self):
        """Just checking whether the custom dictionary containns the required
        inputs."""

        if self.custom:
            checkif = ['water', 'salt', 'yeast']

            if all([i in checkif for i in self.custom]) is False:
                raise ValueError(f'Custom dictionary must contain: {checkif}')

        else:
            raise ValueError("Must provide custom weight dictionary.")

    def __check_style__(self):

        available_styles = [
            'Neapolitan', 'American', 'Sicilian', 'Sourdough', 'Canotto',
            'Custom']

        if (self.style not in available_styles) \
                and (self.style not in [i.lower() for i in available_styles]):
            raise ValueError(f"Style: {self.style} not implemented.")

        if self.style.lower() == 'custom':
            self.__check_custom_dict__()

    def custom(self):

        if 'flour' not in self.custom:
            self.custom['flour'] = 100.0
        self.percentages.update(self.custom)

    def neapolitan(self):
        """

        - Proofing at room temperature(min 6 - max 24 hours) or cold 
          fermenting(min 8 - max 72 hours).
        - Oven temperature: 400 - 550°C / 750 - 1020℉.
        - Baking time: 1 – 4 mins.

        """

        percentages = dict(
            flour=100.0,
            salt=3.0,
            yeast=0.2,
            water=self.watercontent)

        self.percentages.update(percentages)

    def american(self):
        """

        - Proofing at room temperature(min 6 - max 24 hours) or cold 
          fermenting(min 8 - max 72 hours)
        - Oven temperature: 200 – 300°C / 400 - 580℉.
        - Baking time: 8 - 15 min s.

        """

        percentages = dict(
            flour=100.0,
            salt=1.5,
            yeast=0.4,
            water=self.watercontent,
            oil=2.5,
            sugar=2.0)

        self.percentages.update(percentages)

    def sicilian(self):
        """

        - Proofing at room temperature(min 6 - max 24 hours) or cold 
          fermenting(min 8 - max 72 hours)
        - Oven temperature:  250 - 280°C / 480 - 540 ℉.
        - Baking time: 15 - 20 mins.

        """

        percentages = dict(
            flour=100.0,
            salt=2.0,
            yeast=1.5,
            water=self.watercontent,
            oil=1.5)

        self.percentages.update(percentages)

    def canotto(self):
        """

        - Cold fermenting (min 24 - max 72 hours)
        - Oven temperature: 450 - 520°C / 840 - 950℉.
        - Baking time: 1 – 3 mins.

        """

        percentages = dict(
            flour=100.0,
            salt=3.0,
            yeast=0.4,
            water=self.watercontent,
            oil=2.5,
            sugar=2.0)

        self.percentages.update(percentages)

    def sourdough(self):

        percentages = dict(
            flour=100.0,
            salt=3.0,
            starter=16.0,
            water=self.watercontent)

        self.percentages.update(percentages)

    def get_amounts(self):
        """Transform percentages to amounts."""

        # Percentages to fractions
        total = 0.0
        for k, v in self.percentages.items():
            total += v

        # Get the total_weight of a doughball
        total_weight = self.N * self.doughball

        # Get amounts
        for k, v in self.amounts.items():
            self.amounts[k]
            self.amounts[k] = self.percentages[k] / total * total_weight

    def compute(self, style='Neapolitan'):

        # new style
        if style == self.style.lower():
            pass
        else:
            self.style = style.lower()
            self.__check_style__()

        # Reset
        self.reset()

        # Get correct percentages
        get_percentages = getattr(self, self.style)
        get_percentages()

        # Get amounts
        self.get_amounts()

        # Print
        self.print()

    def print(self):

        string = f"\nAmounts for your {self.style.capitalize()} style pizza:\n\n"

        for ingr, amount in self.amounts.items():

            if np.isclose(amount, 0.0):
                continue
            if ingr == 'water':
                unit = 'ml'
            else:
                unit = 'g'

            if amount >= 10.0:
                amount = int(np.round(amount))
                format = '>18d'
                space = '   '
            else:
                format = '>20.1f'
                space = ' '

            string += f"    {ingr.capitalize() + ':':<8}{amount:{format}}{space}{unit}\n"

        print(string)
