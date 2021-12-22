import matplotlib
import mpl_toolkits.axes_grid1


class PageSlider(matplotlib.widgets.Slider):

    def __init__(self, ax, label, numpages=10, valinit=0, valfmt='%1d',
                 closedmin=True, closedmax=True, orientation='horizontal',
                 dragging=True, **kwargs):

        self.facecolor = kwargs.get('facecolor', "w")
        self.activecolor = kwargs.pop('activecolor', "b")
        self.fontsize = kwargs.pop('fontsize', 10)
        self.numpages = numpages
        self.orientation = orientation

        super(PageSlider, self).__init__(
            ax, label, 0, numpages, valinit=valinit, valfmt=valfmt,
            orientation=orientation,
            **kwargs)

        self.poly.set_visible(False)
        if 'orientation' in kwargs:
            if kwargs['orientation'] == 'horizontal':
                self.vline.set_visible(False)

        self.pageRects = []
        for i in range(numpages):
            facecolor = self.activecolor if i == valinit else self.facecolor
            if orientation == 'horizontal':
                r = matplotlib.patches.Rectangle(
                    (float(i)/numpages, 0), 1./numpages, 1,
                    transform=ax.transAxes, facecolor=facecolor)
            elif orientation == 'vertical':
                r = matplotlib.patches.Rectangle(
                    (0, (float(i))/numpages), 1, 1./numpages,
                    transform=ax.transAxes, facecolor=facecolor)

            ax.add_artist(r)
            self.pageRects.append(r)
            if orientation == 'vertical':
                ax.text(0.5, (float(i))/numpages+0.5/numpages,
                        str(i+1), ha="center", va="center",
                        transform=ax.transAxes, fontsize=self.fontsize)

            elif orientation == 'horizontal':
                ax.text(float(i)/numpages+0.5/numpages, 0.5, str(i+1),
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=self.fontsize)
        self.valtext.set_visible(False)

        divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
        if orientation == 'vertical':
            bax = divider.append_axes("bottom", size="5%", pad=0.025)
            fax = divider.append_axes("bottom", size="5%", pad=0.025)
        else:
            bax = divider.append_axes("right", size="5%", pad=0.025)
            fax = divider.append_axes("right", size="5%", pad=0.025)

        self.button_back = matplotlib.widgets.Button(
            bax, label=u'\u25C0', color=self.facecolor,
            hovercolor=self.activecolor)
        self.button_forward = matplotlib.widgets.Button(
            fax, label=u'\u25B6', color=self.facecolor,
            hovercolor=self.activecolor)
        self.button_back.label.set_fontsize(self.fontsize)
        self.button_forward.label.set_fontsize(self.fontsize)
        self.button_back.on_clicked(self.backward)
        self.button_forward.on_clicked(self.forward)

    def _update(self, event):
        super(PageSlider, self)._update(event)
        i = int(self.val)
        if i >= self.valmax:
            return
        self._colorize(i)

    def _colorize(self, i):
        for j in range(self.numpages):
            self.pageRects[j].set_facecolor(self.facecolor)
        self.pageRects[i].set_facecolor(self.activecolor)

    def forward(self, event):
        current_i = int(self.val)
        i = current_i+1
        if (i < self.valmin) or (i >= self.valmax):
            return
        self.set_val(i)
        self._colorize(i)

    def backward(self, event):
        current_i = int(self.val)
        i = current_i-1
        if (i < self.valmin) or (i >= self.valmax):
            return
        self.set_val(i)
        self._colorize(i)
