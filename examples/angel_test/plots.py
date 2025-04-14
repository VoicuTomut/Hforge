import matplotlib.pyplot as plt
import numpy as np

def plot_matrices_true_prediction_difference(M_true, M_pred, label='', path=''):
    '''
    Plot the true matrix, the predicted matrix and the difference between both matrices.
    Aimed to study the Hamiltonian matrices and overlap matrices predictions.
    
    Parameters
    ----------
    M_true : np.ndarray
        The true matrix.
    M_pred : np.ndarray
        The predicted matrix.
    label : str
        The label of the matrix to display in the title.
    '''
    matrices = [M_true, M_pred]

    # Compute the difference matrix (in %)
    M_diff = M_true - M_pred

    # Avoid division by zero and ensure stability
    epsilon = np.finfo(float).eps  # Smallest positive number
    M_true = np.where(M_true == 0, epsilon, M_true)  # Replace zeros with epsilon
    # print(M_diff[37,31])
    # print(M_true[37,31])
    M_diff = M_diff / M_true

    # for i, row in enumerate(M_true):
    #     for j, element in enumerate(row):
    #         if element > 1e-2:
    #             M_diff[i,j] = M_diff[i,j]/M_true[i,j]
    #         elif np.isclose(M_diff[i,j], element, atol=0.01, rtol=0):
    #             M_diff[i,j] = 0
    #         else:
    #             # M_diff[i,j] = M_diff[i,j]/M_true[i,j]
    #             M_diff[i,j] = 0


    # Optionally, clip extreme values to prevent overflow or underflow
    M_diff = np.clip(M_diff, -1e100, 1e100)
    
    #############################

    # Plot the true matrix, the predicted matrix.
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    cmap_string = 'RdYlBu'
    vmin = np.min([np.min(M_true), np.min(M_pred)])
    vmax = np.max([np.max(M_true), np.max(M_pred)])
    
    # Put the zero in the middle of the colorbar
    vmax_absolute = np.max([np.abs(vmin), np.abs(vmax)])

    for i, matrix in enumerate(matrices):
        image = axes[i].imshow(matrix, cmap=cmap_string, vmin=-vmax_absolute, vmax=vmax_absolute)#, extent=[-matrix.shape[1]//2, matrix.shape[1]//2, -matrix.shape[0]//2, matrix.shape[0]//2])

        # axes[i].set_xlabel("x")
        # axes[i].set_ylabel("y")

        titles = ["True " + label, "Predicted " + label]
        axes[i].set_title(titles[i])
        axes[i].xaxis.tick_top()

        fig.colorbar(image)

    #############################

    # Plot the difference matrix
    vmin = np.min(M_diff)
    vmax = np.max(M_diff)
    vmax_absolute = np.max([np.abs(vmin), np.abs(vmax)])

    max_error = 100
    if vmax_absolute>max_error:
        image = axes[2].imshow(M_diff, cmap=cmap_string,  vmin=-max_error, vmax=max_error)#, extent=[-matrix.shape[1]//2, matrix.shape[1]//2, -matrix.shape[0]//2, matrix.shape[0]//2])
    else:
        image = axes[2].imshow(M_diff, cmap=cmap_string,  vmin=-vmax_absolute, vmax=vmax_absolute)#, extent=[-matrix.shape[1]//2, matrix.shape[1]//2, -matrix.shape[0]//2, matrix.shape[0]//2])
    axes[2].set_title('Relative error (M-M\')/M')
    fig.colorbar(image)

    # Plot the max and min
    max_diff = np.max(M_diff)
    min_diff = np.min(M_diff)
    axes[2].text(0.5, -0.1, f'max = {max_diff:.2f},  min = {min_diff:.2f}', ha='center', va='center', transform=axes[2].transAxes, fontsize=12)
    axes[2].xaxis.tick_top()

    if path=="":
        plt.show()
    else:
        plt.savefig(path)


def plot_loss_vs_epochs(loss_vs_epochs, loss_vs_epochs_validation = None, loss_fn_title="MSE", path=""):
    """Plot the loss function vs the number of epoch.

    Args:
        loss_vs_epochs (List[float]): Loss values ordered as increasing epoch of the train dataset.
        loss_vs_epochs_validation (List[float], optional): Loss values ordered as increasing epoch of the validation dataset. If None, it won't plot. Defaults to None.
        loss_fn_title (str, optional): Label of the loss function. Default to "MSE".
    """
    plt.figure()

    # Axis
    n_epochs = len(loss_vs_epochs)
    x_axis = range(0, n_epochs)
    y_axis = loss_vs_epochs
    # y_axis = [loss_vs_epochs[i].cpu().detach().numpy() for i in range(len(loss_vs_epochs))]

    # Plot the loss of the train dataset.
    plt.plot(x_axis,y_axis, label="Train loss")

    # Plot the loss of the test dataset if inputted.
    if loss_vs_epochs_validation != None:
        y_axis = loss_vs_epochs_validation
        # y_axis = [loss_vs_epochs_validation[i].cpu().detach().numpy() for i in range(len(loss_vs_epochs_validation))]
        plt.plot(x_axis, y_axis, label="Validation loss")

    # Plot options.
    plt.legend()
    plt.ylabel("Loss function (" + loss_fn_title + ")")
    plt.xlabel("Epoch")
    # plt.xticks(x_axis)
    minimum = min(loss_vs_epochs)
    plt.title(f"Loss minimum: {minimum: .2f}. Loss last iteration: {loss_vs_epochs[-1]: .2f}.")

    if path=="":
        plt.show()
    else:
        plt.savefig(path)