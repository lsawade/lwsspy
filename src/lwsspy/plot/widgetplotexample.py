# %%
from matplotlib import pyplot as plt
import mpl_toolkits.axes_grid1
import matplotlib.patches
import matplotlib.widgets
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider, Slider, RadioButtons

from lwsspy.plot.pageslider import PageSlider
# %%

import numpy as np

# osc_amp = 0.25
# osc_freq = 1.0
# fact = 0.25
# climb = 10.0
# xmin = 1
# xmax = 21
# x = 20*np.random.rand(1000) + 1

# y = np.log10(
#     fact*x**2
#     + (5 - x)**2 * np.random.rand(x.size)
#     + climb * x * np.random.rand(x.size)) \
#     + osc_amp * x/np.max(x) * np.sin(osc_freq*x)


# plt.figure(figsize=(16, 10))
# plt.plot(x, y, 'o')

# %%


class ScatterExplore:

    def __init__(self):
        self.color = 'k'
        self.marker = 'v'

        self.osc_amp = 0.25
        self.osc_freq = 1.0
        self.fact = 0.25
        self.climb = 10.0
        self.xmin = 1
        self.xmax = 21

        self.update_data()

        self.nbins = 100
        gs = GridSpec(
            ncols=4, nrows=4,
            width_ratios=(1, 0.25, 4, 1), wspace=0.0,
            height_ratios=(1, 5, 0.5, 2), hspace=0.0)

        self.fig = plt.figure(figsize=(16, 10))

        # Plot Data
        self.axdata = self.fig.add_subplot(gs[1, 2], zorder=100)
        self.scatter, = self.axdata.plot(self.x, self.y, 'o', c=self.color)
        print(self.scatter)

        # Plot x histogram
        self.axhistx = self.fig.add_subplot(gs[0, 2], sharex=self.axdata)
        _, _, self.histx = self.axhistx.hist(
            self.x, bins=self.nbins, facecolor=self.color)
        self.axhistx.axis('off')

        # Plot y histogram
        self.axhisty = self.fig.add_subplot(gs[1, 3], sharey=self.axdata)
        _, _, self.histy = self.axhisty.hist(
            self.y, bins=self.nbins, orientation="horizontal",
            facecolor=self.color)
        self.axhisty.axis('off')

        self.subgs_guileft = gs[1, 0].subgridspec(4, 5, wspace=0.3)
        self.subgs_guibott = gs[3, 2].subgridspec(5, 2, wspace=0.3, hspace=0.4)

        self.axcolorradio = self.fig.add_subplot(self.subgs_guileft[0:1, 0:2])
        self.colorradio = RadioButtons(
            self.axcolorradio, ('black', 'blue', 'red', 'green'))
        self.colorradio.on_clicked(self.update_color)

        # X slider
        self.xrangesliderax = self.fig.add_subplot(self.subgs_guibott[0, 0])
        self.xrange = RangeSlider(
            self.xrangesliderax, 'x',
            valmin=1.0, valmax=100.0, valinit=[5.0, 10.0], valfmt=None,
            closedmin=True, closedmax=True, dragging=True,
            valstep=None, orientation='horizontal',
        )

        # Y slider
        self.yrangesliderax = self.fig.add_subplot(self.subgs_guileft[:, -1])
        self.yrange = RangeSlider(
            self.yrangesliderax, 'y-range',
            valmin=-100.0, valmax=100.0, valinit=[-1.0, 10.0], valfmt=None,
            closedmin=True, closedmax=True, dragging=True,
            valstep=None, orientation='vertical',
        )

        # Amplitude slider
        self.ax_amp_slider = self.fig.add_subplot(self.subgs_guileft[2:, 0])
        self.amp_slider = Slider(
            self.ax_amp_slider, 'Osc.Amp',
            valmin=0.0, valmax=5.0, valinit=self.osc_amp,
            orientation='vertical',
        )
        self.amp_slider.on_changed(self.update_amp)

        # Factor slider
        self.ax_fact_slider = self.fig.add_subplot(self.subgs_guileft[2:, 1])
        self.fact_slider = Slider(
            self.ax_fact_slider, 'Factor',
            valmin=0.0, valmax=5.0, valinit=self.fact,
            orientation='vertical',
        )
        self.fact_slider.on_changed(self.update_fact)

        # Climb slider
        self.ax_climb_slider = self.fig.add_subplot(self.subgs_guileft[2:, 2])
        self.climb_slider = Slider(
            self.ax_climb_slider, 'Climb',
            valmin=0.0, valmax=100.0, valinit=self.climb,
            orientation='vertical',
        )
        self.climb_slider.on_changed(self.update_climb)

        # Osfreq slider
        self.ax_oscfreq_slider = self.fig.add_subplot(self.subgs_guibott[1, :])
        self.oscfreq_slider = Slider(
            self.ax_oscfreq_slider, 'Climb',
            valmin=0.0, valmax=5, valinit=self.osc_freq, valstep=0.01,
            orientation='horizontal',
        )
        self.oscfreq_slider.on_changed(self.update_freq)

        # self.xrange = RangeSlider(
        #     self.rangesliderax, 'x-range',
        #     valmin=-100, valmax=100.0, valinit=[-1.0, 5.0], valfmt=None,
        #     closedmin=True, closedmax=True, dragging=True,
        #     valstep=None, orientation='vertical',  # track_color='lightgrey',
        #     # handle_style=None
        # )  # , **kwargs)

        # print(self.xrange)
        self.xrange.on_changed(self.updatexrange)
        self.yrange.on_changed(lambda x: self.axdata.set_ylim(x))

        # _, _, self.histy = self.axhisty.hist(x)
        plt.show()

    def updatexrange(self, val):

        self.xmin = val[0]
        self.xmax = val[1]
        self.update_data()
        self.update_plots()

    def update_color(self, label):

        self.color = label

        self.scatter.set_color(self.color)
        _ = [_b.set_facecolor(self.color) for _b in self.histx]
        _ = [_b.set_facecolor(self.color) for _b in self.histy]
        self.fig.canvas.draw()

    def update_amp(self, val):

        self.osc_amp = val
        self.update_data()
        self.update_plots()

    def update_freq(self, val):

        self.osc_freq = val
        self.update_data()
        self.update_plots()

    def update_fact(self, val):

        self.fact = val
        self.update_data()
        self.update_plots()

    def update_climb(self, val):

        self.osc_ = val
        self.update_data()
        self.update_plots()

    def update_plots(self):

        self.scatter.set_xdata(self.x)
        self.scatter.set_ydata(self.y)
        self.axdata.set_xlim(self.xmin, self.xmax)
        self.axdata.set_ylim(self.ymin, self.ymax)
        # self.xrange.set_val([self.xmin, self.xmax])
        self.yrange.set_val([self.ymin, self.ymax])
        self.update_hists()

    def update_hists(self):

        _ = [_b.remove() for _b in self.histx]
        _ = [_b.remove() for _b in self.histy]

        _, _, self.histx = self.axhistx.hist(
            self.x, bins=self.nbins, orientation="vertical",
            facecolor=self.color)
        _, _, self.histy = self.axhisty.hist(
            self.y, bins=self.nbins, orientation="horizontal",
            facecolor=self.color)

    def update_data(self):

        self.x = (self.xmax-self.xmin)*np.random.rand(1000) + self.xmin

        self.y = np.log10(
            self.fact*self.x**2
            + (5 - self.x)**2 * np.random.rand(self.x.size)
            + self.climb * self.x * np.random.rand(self.x.size)) \
            + self.osc_amp * self.x / \
            np.max(self.x) * np.sin(self.osc_freq*self.x)

        self.ymin = np.min(self.y)
        self.ymax = np.max(self.y)


ScatterExplore()

# %%


# %%


num_pages = 23
data = np.random.rand(9, 9, num_pages)

fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.18)

im = ax.imshow(data[:, :, 0], cmap='viridis', interpolation='nearest')

ax_slider = fig.add_axes([0.05, 0.1, 0.05, 0.8])

# %%
slider = PageSlider(ax_slider, 'Page', num_pages, activecolor="orange",
                    orientation='vertical')
# %%

ax_slider.axis('off')


def update(val):
    i = int(slider.val)
    im.set_data(data[:, :, i])


slider.on_changed(update)

plt.show()

# %%
