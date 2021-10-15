import matplotlib.pyplot


x = 10*random.rand(1000)
y = 0.25*x**2 + (5 - x)**2*random.rand(x.size) + x*random.rand(x.size)
plotdict = dict(
    blines=dict(lw=1.0),
    mean=dict(ls='-', marker='o'),
    std=dict(ls='-', marker='-'),
    quantile=dict(ls='', marker='+')
)
plot(x, y, 'o')
plot_binnedstats(x, y, bins=bins, orientation='horizontal')
bins = linspace(0, 10, 11)
plot_binnedstats(x, y, bins=bins, orientation='horizontal')
plotdict = dict(
    blines=dict(
        lw=1.0,
        zorder=1000
    ),
    mean=dict(
        ls='-',
        marker='o',

    ),
    std=dict(
        ls='-',
        marker='-',
    ),
    quantile=dict(
        ls='',
        marker='+'
    )
)
plot_binnedstats(x, y, bins=bins, orientation='horizontal', plotdict=plotdict)
