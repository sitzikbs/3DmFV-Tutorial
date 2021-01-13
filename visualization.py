import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def visualize_fv_gaussian_and_points(gmm, points, fv_per_point, fv, display=False, export=False, export_path=''):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    derivatives = [r"$g_{\alpha}$", r"$g_{\mu_x}$", r"$g_{\mu_y}$", r"$g_{\sigma_x}$", r"$g_{\sigma_y}$"]
    sigma = np.sqrt(gmm.covariances_)
    centers = gmm.means_
    circles = []
    for i, center in enumerate(centers):
        circles.append(plt.Circle((centers[i,0], centers[i,1]), sigma[i,0], color='k', fill=False, linestyle='--'))
    fig = plt.figure()
    ax1 = plt.subplot2grid((6, 6), (0, 0), colspan=5, rowspan=5) # gaussian and points
    #ax2 = plt.subplot2grid((3, 3), (0, 2), colspan=1, rowspan=1)
    ax3 = plt.subplot2grid((6, 6), (0, 5), colspan=1, rowspan=5) # fisher vector

    for i,circle in enumerate(circles):
        ax1.add_patch(circle)

    # plot the points
    ax1.plot(points[:, 0], points[:, 1], 'o', color='black')
    # assuming data is within the unit sphere
    ax1.set_xlim((-2, 2))
    ax1.set_ylim((-2, 2))
    ax1.set_aspect(1)
    #plt.axis('off')
    ax3.imshow(np.expand_dims(fv, axis=1), cmap=plt.get_cmap('seismic'), vmin=-1, vmax=1)
    tick_marks = np.arange(len(derivatives))
    ax3.set_xticks([])
    ax3.set_yticks(tick_marks)
    ax3.set_yticklabels(derivatives)
    for i, fv_val in enumerate(fv):
        plt.text(0, i, round(fv_val, 2),
                 horizontalalignment="center", fontsize=10,
                 color="black" if fv_val < 0.15 and fv_val > -0.15 else "white")
    if display:
        plt.show()
    if export:
        plt.savefig(export_path, format='png', dpi=150)

    fig.canvas.draw()

    # Now we can save it to a numpy array.
    fig_image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    fig_image = fig_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return fig_image


def visualize_3dmfv_gaussian_and_points(gmm, points, pc_3dmfv, display=False, export=False, export_path=''):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    derivatives = [r"$\max{g_{\alpha}}$",r"$\sum g_{\alpha}$",
                   r"$\max{g_{\mu_x}}$", r"$\max{g_{\mu_y}}$", r"$\max{g_{\mu_z}}$",
                   r"$\min{g_{\mu_x}}$", r"$\min{g_{\mu_y}}$", r"$\min{g_{\mu_z}}$",
                   r"$\sum g_{\mu_x}$", r"$\sum g_{\mu_y}$", r"$\sum g_{\mu_z}$",
                   r"$\max{g_{\sigma_x}}$", r"$\max{g_{\sigma_y}}$", r"$\max{g_{\sigma_z}}$",
                   r"$\min{g_{\sigma_x}}$", r"$\min{g_{\sigma_y}}$", r"$\min{g_{\sigma_z}}$",
                   r"$\sum g_{\sigma_x}$", r"$\sum g_{\sigma_y}$", r"$\sum g_{\sigma_z}$"]

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    set_ax_props(ax1)
    draw_gaussians(gmm, ax=ax1, display=False) # plot the Gaussian

    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], marker='.', c='black', s=10) # plot the points
    # plot the 3dmfvrepresentaion
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(pc_3dmfv, cmap=plt.get_cmap('seismic'), vmin=-1, vmax=1)
    tick_marks = np.arange(len(derivatives))
    ax2.set_xticks([])
    ax2.set_yticks(tick_marks)
    ax2.set_yticklabels(derivatives)
    for i, fv_val_g in enumerate(pc_3dmfv):
        for j, fv_val in enumerate(fv_val_g):
            plt.text(j, i, np.around(fv_val, 2),
                 horizontalalignment="center", fontsize=3,
                 color="black" if fv_val < 0.15 and fv_val > -0.15 else "white")

    if display:
        plt.show()
    if export:
        # mng = plt.get_current_fig_manager()
        # mng.frame.Maximize(True)
        # mng = plt.get_current_fig_manager()
        # mng.resize(*mng.window.maxsize())
        plt.savefig(export_path, format='png', dpi=150)

    fig.canvas.draw()
    # Now we can save it to a numpy array.
    fig_image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    fig_image = fig_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return fig_image


def draw_gaussians(gmm, ax='none', display=False):
    if ax == 'none':
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(0, 0)
        set_ax_props(ax)

    x, y, z = sphere(subdev=20)
    n_gaussians = len(gmm.weights_)
    for i in range(n_gaussians):
        X = x*np.sqrt(gmm.covariances_[i][0]) + gmm.means_[i][0]
        Y = y*np.sqrt(gmm.covariances_[i][1]) + gmm.means_[i][1]
        Z = z*np.sqrt(gmm.covariances_[i][2]) + gmm.means_[i][2]

        ax.plot_surface(X, Y, Z, color='w', alpha=0.2, linewidth=0.5, edgecolor=[0.3, 0.3, 0.3])

    if display:
        plt.show()
    return ax

def set_ax_props(ax, range=1):
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.set_xlim([-range, range])
    ax.set_ylim([-range, range])
    ax.set_zlim([-range, range])
    ax.view_init(elev=35.264, azim=45)
    axisEqual3D(ax)
    return ax

def sphere(subdev=10):
    #helper function to compute the coordinates of a unit sphere centered at 0,0,0
    # Create a sphere
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:complex(0,subdev), 0.0:2.0 * pi:complex(0,subdev)]
    x = r * sin(phi) * cos(theta)
    y = r * sin(phi) * sin(theta)
    z = r * cos(phi)
    return (x,y,z)

def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
