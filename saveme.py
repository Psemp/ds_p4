import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def display_circles(pca, pca_cols: list, couple_pc: tuple):
    """
    Function : Display pca of pcs couple_pc correlation circles for pca of dataframe
    in parameter.
    Original : https://github.com/AnisHdd

    Args:
    - pca : the pca object using sklearn
    - pca_cols : the columns of df upon which the pca was performed
    - couple_pc : tuple (x, y) as x and y indexes of principal components

    Returns:
    - Void functio, displays plot
    """
    x_pc = couple_pc[0]
    y_pc = couple_pc[1]

    fig, (ax1) = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(10, 10),
        dpi=150,
    )

    for i in range(0, pca.components_.shape[1]):
        ax1.arrow(
            0,
            0,  # origin
            pca.components_[x_pc, i],
            pca.components_[y_pc, i],
            head_width=0.1,
            head_length=0.1
        )

        plt.text(
            pca.components_[x_pc, i] + 0.05,
            pca.components_[y_pc, i] + 0.05,
            pca_cols[i]
        )

    an = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(an), np.sin(an))  # Draw circle

    ###
    # Titles/Lables
    ax1.set_title(f"Correlation Circle : PC{x_pc + 1} / PC{y_pc + 1} ")
    ax1.set_xlabel(f"PC{x_pc + 1}")
    ax1.set_ylabel(f"PC{y_pc + 1}")
    #
    ###

    plt.show()


def show_contribution(pca, columns_pca: list, lim_pc: int = None) -> pd.DataFrame:
    """
    Takes the pca object, the list of target columns for the pca and the percentages of variation
    to return a pd.DataFrame object with details on the principal components

    Args :
    - pca : The PCA object
    - columns_pca : list of column upon which the PCA was applied
    - lim_pc : Optionnal, default = None, the last Principal component (PC after PC_lim_pc will be ignored)
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

    elif lim_pc is not None:

        for i in range(0, lim_pc):
            sers[f"PC{i + 1}"] = pd.Series(pca.components_[i], index=columns_pca)
            contrib_dict[f"PC{i + 1}"] = pd.Series(percentage_variation[i], index=["contribution"])

    components_df = pd.DataFrame(sers)
    temp = pd.DataFrame(contrib_dict)
    frames = [components_df, temp]
    components_df = pd.concat(frames)
    return components_df

