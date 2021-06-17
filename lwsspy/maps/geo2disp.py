from cartopy import crs


def geo2disp(target_crs, x, y, orig_crs=crs.PlateCarree()):
    return target_crs.transform_points(orig_crs, x, y)
