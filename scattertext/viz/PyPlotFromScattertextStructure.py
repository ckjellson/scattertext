import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random


def pyplot_from_scattertext_structure(
    scatterplot_structure,
    figsize,
    textsize,
    distance_margin_fraction,
    scatter_size,
    cmap,
    sample,
    xlabel,
    ylabel,
    dpi,
):
    """
    Parameters
    ----------
    scatterplot_structure : ScatterplotStructure
    figsize : Tuple[int,int]
        Size of ouput pyplot figure
    textsize : int
        Size of text terms in plot
    distance_margin_fraction : float
        Fraction of the 2d space to use as margins for text bboxes
    scatter_size : int
        Size of scatter disks
    cmap : str
        Matplotlib compatible colormap string
    sample : int
        if >0 samples a subset from the scatterplot_structure, used for testing
    xlabel : str
        Overrides label from scatterplot_structure
    ylabel : str
        Overrides label from scatterplot_structure
    dpi : int
        Pyplot figure resolution

    Returns
    -------
    matplotlib.figure.Figure
    matplotlib figure that can be used with plt.show() or plt.savefig()

    """
    # Extract the data
    if sample > 0:
        subset = random.sample(
            scatterplot_structure._visualization_data.word_dict["data"], sample
        )
    else:
        subset = scatterplot_structure._visualization_data.word_dict["data"]
    df = pd.DataFrame(subset)
    if (
        "etc" in scatterplot_structure._visualization_data.word_dict["data"][0]
        and "ColorScore"
        in scatterplot_structure._visualization_data.word_dict["data"][0]["etc"]
    ):
        df["s"] = [d["etc"]["ColorScore"] for d in subset]
    info = scatterplot_structure._visualization_data.word_dict["info"]
    n_docs = len(scatterplot_structure._visualization_data.word_dict["docs"]["texts"])
    n_words = df.shape[0]

    if scatterplot_structure._show_characteristic:
        characteristic_terms = list(
            df.sort_values("bg", axis=0, ascending=False).iloc[:23].term
        )

    if df.s.isna().sum() > 0:
        colors = "k"
    else:
        colors = df.s

    # Initiate plotting
    if scatterplot_structure._ignore_categories:
        if scatterplot_structure._show_characteristic:
            fig, axs = plt.subplots(1, 2, figsize=figsize, width_ratios=[5, 1], dpi=dpi)
            ax_char = axs[1]
        else:
            fig, axs = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    else:
        if scatterplot_structure._show_characteristic:
            fig, axs = plt.subplots(
                1, 3, figsize=figsize, width_ratios=[6, 1, 1], dpi=dpi
            )
            ax_cat = axs[1]
            ax_char = axs[2]
        else:
            fig, axs = plt.subplots(1, 2, figsize=figsize, width_ratios=[5, 1], dpi=dpi)
            ax_cat = axs[1]
    plt.tight_layout()
    ax_plot = axs[0]
    ax_plot.scatter(df.x, df.y, c=colors, s=scatter_size, cmap=cmap)

    # Run pre-labeling to determine textbox sizes
    print("Creating patches")
    original_patches = []
    for x, y, s in tqdm(zip(df.x, df.y, df.term)):
        ann = ax_plot.text(x, y, s, size=textsize)
        patch = ax_plot.transData.inverted().transform(
            ann.get_tightbbox(fig.canvas.get_renderer())
        )
        w, h = patch[1][0] - patch[0][0], patch[1][1] - patch[0][1]
        original_patches.append((w, h, s))
        ann.remove()
    xlims = ax_plot.get_xlim()
    ylims = ax_plot.get_ylim()

    # Process extracted textboxes
    print("Processing")
    non_overlapping_patches = get_non_overlapping_patches(
        original_patches,
        df[["x", "y"]].to_numpy(),
        xlims,
        ylims,
        distance_margin_fraction=distance_margin_fraction,
    )

    # Plot once again
    print("Plotting")
    for x, y, s in non_overlapping_patches:
        ax_plot.text(x, y, s, size=textsize)

    # Design settings
    ax_plot.spines.right.set_visible(False)
    ax_plot.spines.top.set_visible(False)
    if xlabel is not None:
        ax_plot.set_xlabel(xlabel)
    elif scatterplot_structure._x_label is not None:
        ax_plot.set_xlabel(scatterplot_structure._x_label)
    elif not scatterplot_structure._ignore_categories:
        ax_plot.set_xlabel(info["not_category_name"])
    else:
        pass
    if ylabel is not None:
        ax_plot.set_ylabel(ylabel)
    elif scatterplot_structure._y_label is not None:
        ax_plot.set_ylabel(scatterplot_structure._y_label)
    elif not scatterplot_structure._ignore_categories:
        ax_plot.set_ylabel(info["category_name"])
    else:
        pass
    ax_plot.locator_params(axis="y", nbins=3)
    ax_plot.locator_params(axis="x", nbins=3)
    try:
        if scatterplot_structure._x_axis_labels is not None:
            ax_plot.set_xticks(
                ax_plot.get_xticks()[1:-1], scatterplot_structure._x_axis_labels, size=7
            )
        else:
            ax_plot.set_xticks(
                ax_plot.get_xticks()[1:-1], ["Low", "Medium", "High"], size=7
            )
    except:
        pass
    try:
        if scatterplot_structure._y_axis_labels is not None:
            ax_plot.set_yticks(
                ax_plot.get_yticks()[1:-1], scatterplot_structure._y_axis_labels, size=7, rotation=90
            )
        else:
            ax_plot.set_yticks(
                ax_plot.get_yticks()[1:-1], ["Low", "Medium", "High"], size=7, rotation=90
            )
    except:
        pass
    if scatterplot_structure._show_diagonal:
        ax_plot.plot(
            [xlims[0], xlims[1]],
            [ylims[0], ylims[1]],
            color="k",
            linestyle="dashed",
            linewidth=1,
            alpha=0.3,
        )

    # Categories
    alignment = {"horizontalalignment": "left", "verticalalignment": "top"}
    if not scatterplot_structure._ignore_categories:
        yp = [i / 22 for i in range(22)]
        yp.reverse()
        ax_cat.text(
            0.0,
            yp[0],
            "Top " + info["category_name"],
            weight="bold",
            size="medium",
            **alignment,
        )
        for i, term in enumerate(info["category_terms"]):
            ax_cat.text(0.0, yp[i + 1], term, size="small", **alignment)
        ax_cat.text(
            0.0,
            yp[11],
            "Top " + info["not_category_name"],
            weight="bold",
            size="medium",
            **alignment,
        )
        for i, term in enumerate(info["not_category_terms"]):
            axs[1].text(0.0, yp[i + 12], term, size="small", **alignment)
        ax_cat.spines.right.set_visible(False)
        ax_cat.spines.top.set_visible(False)
        ax_cat.spines.bottom.set_visible(False)
        ax_cat.spines.left.set_visible(False)
        ax_cat.set_xticks([])
        ax_cat.set_yticks([])

    # Characteristics
    if scatterplot_structure._show_characteristic:
        yp = [i / 24 for i in range(24)]
        yp.reverse()
        ax_char.text(
            0.0, yp[0], "Characteristic", weight="bold", size="medium", **alignment
        )
        for i, term in enumerate(characteristic_terms):
            ax_char.text(0.0, yp[i + 1], term, size="small", **alignment)
        ax_char.spines.right.set_visible(False)
        ax_char.spines.top.set_visible(False)
        ax_char.spines.bottom.set_visible(False)
        ax_char.spines.left.set_visible(False)
        ax_char.set_xticks([])
        ax_char.set_yticks([])

    fig.suptitle(f"Document count: {n_docs} - Word count: {n_words}", ha="right")
    return fig


def get_non_overlapping_patches(
    original_patches, pointarr, xlims, ylims, distance_margin_fraction=0.01
):
    """
    Parameters
    ----------
    original_patches : list
        List of tuples containing width, height and term of each original text-
        box (w,h,s) for all N original patches
    pointarr : np.ndarray
        Array of shape (N,2) containing coordinates for all scatter-points
    xlims : tuple
        (xmin, xmax) of plot
    ylims : tuple
        (ymin, ymax) of plot
    distance_margin_fraction : float
        Fraction of the 2d space to use as margins for text bboxes

    Returns
    -------
    list
    List of tuples containing x, y-coordinates and term of each non-overlapping text-
    box (x,y,s) considering both other patches and the scatterplot-points

    """
    xmin_bound, xmax_bound = xlims
    ymin_bound, ymax_bound = ylims

    xfrac = (xmax_bound - xmin_bound) * distance_margin_fraction
    yfrac = (ymax_bound - ymin_bound) * distance_margin_fraction

    rectangle_arr = np.zeros((0, 4))
    non_overlapping_patches = []

    # Iterate original patches and find ones that do not overlap by creating multiple candidates
    non_overlapping_patches = []
    for i, patch in tqdm(enumerate(original_patches)):
        x_original = pointarr[i, 0]
        y_original = pointarr[i, 1]
        w, h, _ = patch
        candidates = generate_candidates(w, h, x_original, y_original, xfrac, yfrac)

        # Check for overlapping
        non_op = non_overlapping_with_points(pointarr, candidates, xfrac, yfrac)
        if rectangle_arr.shape[0] == 0:
            non_orec = np.zeros((candidates.shape[0],)) == 0
        else:
            non_orec = non_overlapping_with_rectangles(
                rectangle_arr, candidates, xfrac, yfrac
            )
        inside = inside_plot(xmin_bound, ymin_bound, xmax_bound, ymax_bound, candidates)

        # Validate
        ok_candidates = np.where(
            np.bitwise_and(non_op, np.bitwise_and(non_orec, inside))
        )[0]
        if len(ok_candidates) > 0:
            candidate = candidates[ok_candidates[0], :]
            rectangle_arr = np.vstack(
                [
                    rectangle_arr,
                    np.array(
                        [candidate[0], candidate[1], x_original + w, y_original + h]
                    ),
                ]
            )
            non_overlapping_patches.append((candidate[0], candidate[1], patch[2]))
    return non_overlapping_patches


def generate_candidates(w, h, x, y, xfrac, yfrac):
    """
    Parameters
    ----------
    w : float
        width of text box
    h : float
        height of text box
    x : float
        point x-coordinate
    y : float
        point y-coordinate
    xfrac : float
        fraction of the x-dimension to use as margins for text bboxes
    yfrac : float
        fraction of the y-dimension to use as margins for text bboxes

    Returns
    -------
    np.ndarray
    Array of shape (K,4) with K candidate patches

    """
    candidates = np.array(
        [
            [x - w - xfrac, y - h / 2, x - xfrac, y + h / 2],  # left side
            [x - w - xfrac, y + yfrac, x - xfrac, y + h + yfrac],  # upper left side
            [x - w - xfrac, y - h - yfrac, x - xfrac, y - yfrac],  # lower left side
            [x + xfrac, y - h / 2, x + w + xfrac, y + h / 2],  # right side
            [x + xfrac, y + yfrac, x + w + xfrac, y + h + yfrac],  # upper right side
            [x + xfrac, y - h - yfrac, x + w + xfrac, y - yfrac],  # lower right side
            [x - w / 2, y + yfrac, x + w / 2, y + h + yfrac],  # above
            [x - 3 * w / 4, y + yfrac, x + w / 4, y + h + yfrac],  # above left
            [x - w / 4, y + yfrac, x + 3 * w / 4, y + h + yfrac],  # above right
            [x - w / 2, y - h - yfrac, x + w / 2, y - yfrac],  # below
            [x - 3 * w / 4, y - h - yfrac, x + w / 4, y - yfrac],  # below left
            [x - w / 4, y - h - yfrac, x + 3 * w / 4, y - yfrac],  # below right
        ]
    )
    return candidates


def non_overlapping_with_points(pointarr, candidates, xfrac, yfrac):
    """
    Parameters
    ----------
    pointarr : np.ndarray
        Array of shape (N,2) containing coordinates for all scatter-points
    candidates : np.ndarray
        Array of shape (K,4) with K candidate patches
    xfrac : float
        fraction of the x-dimension to use as margins for text bboxes
    yfrac : float
        fraction of the y-dimension to use as margins for text bboxes

    Returns
    -------
    np.array
    Boolean array of shape (K,) with True for non-overlapping candidates with points

    """
    return np.invert(
        np.bitwise_or.reduce(
            np.bitwise_and(
                candidates[:, 0][:, None] - xfrac < pointarr[:, 0],
                np.bitwise_and(
                    candidates[:, 2][:, None] + xfrac > pointarr[:, 0],
                    np.bitwise_and(
                        candidates[:, 1][:, None] - yfrac < pointarr[:, 1],
                        candidates[:, 3][:, None] + yfrac > pointarr[:, 1],
                    ),
                ),
            ),
            axis=1,
        )
    )


def non_overlapping_with_rectangles(rectangle_arr, candidates, xfrac, yfrac):
    """
    Parameters
    ----------
    rectangle_arr : np.ndarray
        Array of shape (N,4) containing patches of all added patches so far
    candidates : np.ndarray
        Array of shape (K,4) with K candidate patches
    xfrac : float
        fraction of the x-dimension to use as margins for text bboxes
    yfrac : float
        fraction of the y-dimension to use as margins for text bboxes

    Returns
    -------
    np.array
    Boolean array of shape (K,) with True for non-overlapping candidates with points

    """
    return np.invert(
        np.any(
            np.invert(
                np.bitwise_or(
                    candidates[:, 0][:, None] - xfrac > rectangle_arr[:, 2],
                    np.bitwise_or(
                        candidates[:, 2][:, None] + xfrac < rectangle_arr[:, 0],
                        np.bitwise_or(
                            candidates[:, 1][:, None] - yfrac > rectangle_arr[:, 3],
                            candidates[:, 3][:, None] + yfrac < rectangle_arr[:, 1],
                        ),
                    ),
                )
            ),
            axis=1,
        )
    )


def inside_plot(xmin_bound, ymin_bound, xmax_bound, ymax_bound, candidates):
    """
    Parameters
    ----------
    xmin_bound : float
    ymin_bound : float
    xmax_bound : float
    ymax_bound : float
    candidates : np.ndarray
        Array of shape (K,4) with K candidate patches

    Returns
    -------
    np.array
    Boolean array of shape (K,) with True for non-overlapping candidates with points

    """
    return np.invert(
        np.bitwise_or(
            candidates[:, 0] < xmin_bound,
            np.bitwise_or(
                candidates[:, 1] < ymin_bound,
                np.bitwise_or(
                    candidates[:, 2] > xmax_bound, candidates[:, 3] > ymax_bound
                ),
            ),
        )
    )
