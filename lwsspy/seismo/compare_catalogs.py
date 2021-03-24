from .cmt_catalog import CMTCatalog
from .source import CMTSource


class CompareCatalogs:

    def __init__(self, old: CMTCatalog, new: CMTCatalog,
                 oldlabel: str = 'Old', newlabel: str = 'New'):

        # Assign
        self.oldlabel = oldlabel
        self.newlabel = newlabel

        # Fix up so they can be compared (if both catalogs are very large this
        # can take some time)
        old = old.unique(ret=True)
        self.old, self.new = old.check_ids(new)
