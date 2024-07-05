import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors


def cbar_norm_colors(values, cbar='turbo', *, min_value=None, max_value=None, norm='linear') -> tuple[np.ndarray, plt.cm.ScalarMappable]:
    """
    Get the colors and normalizer for a colorbar based on the values and the colormap.

    Parameters
    ----------
    cbar: str
        The name of the colormap to use
    values: np.ndarray | list | int
        The values use for the colormap, if int, so many values will be created between `min_value` (default=0) and `max_value` (default=1)
    min_value: float or int
        The minimum value for the colormap
    max_value: float or int
        The maximum value for the colormap
    norm: str or plt.Normalize
        The normalization to use for the colormap. Can be string of 'linear' or 'log', or a plt.Normalize object.

    Returns
    -------
    colors: np.ndarray
        The colors for the colorbar
    scalar_mappable: plt.cm.ScalarMappable
    """
    if isinstance(norm, str):
        if norm == 'linear':
            norm = plt.Normalize(vmin=min_value, vmax=max_value, clip=True)
        elif norm == 'log':
            norm = mpl.colors.LogNorm(vmin=min_value, vmax=max_value, clip=True)
        else:
            raise ValueError(f'Unknown norm: {norm}')
    elif isinstance(norm, plt.Normalize):
        pass
    else:
        raise TypeError(f'Unknown type for norm: {type(norm)}')

    if isinstance(values, int):
        if min_value is None:
            min_value = 0
        if max_value is None:
            max_value = 1
        values = np.linspace(min_value, max_value, values)
    normalized_values = norm(values)

    return plt.get_cmap(cbar)(normalized_values), plt.cm.ScalarMappable(norm, plt.get_cmap(cbar))


def bounded_cbar(values, cmap='turbo', *, equidistant=True, boundaries=None):
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    else:
        cmap = cmap

    if boundaries is None:
        boundaries = ([1.5 * values[0] - 0.5 * values[1]] + [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]
                      + [1.5 * values[-1] - 0.5 * values[1]])

    if equidistant:
        norm = mpl.colors.BoundaryNorm(boundaries, cmap.N)
    else:
        raise NotImplementedError
        # vmin = boundaries[0][0]
        # vmax = boundaries[-1][1]
        # locations = (values - vmin)/(vmax-vmin)
        #
        # def custom_norm(value):
        #     return np.searchsorted(boundaries, value, side='right')/len(boundaries)
        #
        # def inv_custom_norm(value):
        #     return boundaries[int(value * len(boundaries))]
        #
        # norm = mcolors.FuncNorm((custom_norm, inv_custom_norm), vmin=boundaries[0][0], vmax=boundaries[-1][1])

    colors = cmap(norm(values))

    ticker = mpl.ticker.FixedFormatter([f'{int(x)}' for x in values])
    tick_loc = [(boundaries[i] + boundaries[i + 1]) / 2 for i in range(len(boundaries) - 1)]
    t_cbar_kwargs = {'mappable': plt.cm.ScalarMappable(norm=norm, cmap=cmap), 'ticks': tick_loc, 'format': ticker}
    return colors, t_cbar_kwargs


