import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd


def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    """
    Returns pca circles for axis_ranks passed as list of tuples in parameters.
     Pcs is data, n_comps is needed to error check.
    Minor clarifications and linting from
     original function https://github.com/stenier-oc/realisez-une-analyse-de-donnees-exploratoire
    """
    for d1, d2 in axis_ranks:  # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:

            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(18, 18))

            # détermination des limites du graphique
            if lims is not None:
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30:
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else:
                xmin, xmax, ymin, ymax = min(pcs[d1, :]), max(pcs[d1, :]), min(pcs[d2, :]), max(pcs[d2, :])

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30:
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                           pcs[d1, :], pcs[d2, :],
                           angles='xy', scale_units='xy', scale=1, color="grey"
                           )

                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0, 0], [x, y]] for x, y in pcs[[d1, d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))

            # affichage des noms des variables
            if labels is not None:
                for i, (x, y) in enumerate(pcs[[d1, d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax:
                        plt.text(
                                    x, y, labels[i], fontsize='14', ha='center', va='center',
                                    rotation=label_rotation, color="blue", alpha=0.5
                                )

            # affichage du cercle
            circle = plt.Circle((0, 0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)

            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel(f"PC{d1+1} ({round(100*pca.explained_variance_ratio_[d1],1)}%)")
            plt.ylabel(f"PC{d2+1} ({round(100*pca.explained_variance_ratio_[d2],1)}%)")

            plt.title(f"Cercle des corrélations (PC{d1 + 1} et PC{d2 + 1})")
            plt.tight_layout()
            plt.show(block=False)


def show_contribution(pca, columns_pca: list, lim_pc: int = None) -> pd.DataFrame:
    """
    Takes the pca object, the list of target columns for the pca and the percentages of variation
    to return a pd.DataFrame object with details on the principal components

    Args :
    - pca : The PCA object
    - columns_pca : list of column upon which the PCA was applied
    - lim_pc : Optionnal, default = None, the last Principal component (PC after PC_lim_pc will be ignoredS)
    Returns :
    - DataFrame with detailed characteristics of each PC
    """

    percentage_variation = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    sers = {}
    contrib_dict = {}

    if lim_pc is None:

        for i in range(0, len(pca.components_)):
            sers[f"PC{i + 1}"] = pd.Series(pca.components_[i], index=columns_pca)
            contrib_dict[f"PC{i + 1}"] = pd.Series(percentage_variation[i], index=["contribution"])
            contrib_dict["commulative_contribution"] = percentage_variation[i]

        cummulative_contribution = percentage_variation[i]

        if i == 0:
            contrib_dict["commulative_contribution"] = cummulative_contribution
        elif i != 0:
            for index in range(0, i - 1, 1):
                cummulative_contribution += percentage_variation[index]

    elif lim_pc is not None:

        for i in range(0, lim_pc):
            sers[f"PC{i + 1}"] = pd.Series(pca.components_[i], index=columns_pca)
            contrib_dict[f"PC{i + 1}"] = pd.Series(percentage_variation[i], index=["contribution"])

        cummulative_contribution = percentage_variation[i]

        if i == 0:
            contrib_dict["commulative_contribution"] = cummulative_contribution
        elif i != 0:
            for index in range(0, i - 1, 1):
                cummulative_contribution += percentage_variation[index]

    components_df = pd.DataFrame(sers)
    temp = pd.DataFrame(contrib_dict)
    frames = [components_df, temp]
    components_df = pd.concat(frames)

    return components_df
