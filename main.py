import numpy as np
import os
import imageio
import utils
import visualization

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = 'images'
if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

def generate_2d_example():
    gmm = utils.get_2d_grid_gmm(subdivisions=[1, 1], variance=1)
    #spiral  : x=a*t*cos(t), y=a*t*sin(t)
    a = 1/(5*np.pi)
    t = np.linspace(0, 7*np.pi, 120, endpoint=True)
    points = np.transpose(np.array([a*t*np.cos(t), a*t*np.sin(t)]))
    points_flip_y = np.flipud(np.transpose(np.array([a*t*np.cos(t), a*t*np.sin(t)])))
    points_flip_y[:, 1] = -points_flip_y[:, 1]
    points = np.concatenate([points, points_flip_y], axis=0)
    # points = np.array([[[0, 0.3], [0, -0.3]]])
    fig_images = []
    for i, point in enumerate(points):
        if point.ndim == 1:
            point = np.expand_dims(point, axis=0)
        fv = utils.get_fisher_vectors(point, gmm, normalization=False)
        # fv = utils.get_3DmFV(points, gmm.weights_, gmm.means_, np.sqrt(gmm.covariances_), normalize=False)
        fv_per_point = utils.fisher_vector_per_point(point, gmm)
        fig_image = visualization.visualzie_fv_gaussian_and_points(gmm, point, fv_per_point, fv, display=False, export=True,
                                                       export_path=OUTPUT_PATH + '/fv_2d'+str(i)+'.png')
        print(fv)
        fig_images.append(fig_image)
    imageio.mimsave(OUTPUT_PATH + '/2d_fv.gif', fig_images)


def get_3d_spiral():
    # spiral  : x=a*t*cos(t), y=a*t*sin(t)
    a = 1 / (5 * np.pi)
    t = np.linspace(0, 7 * np.pi, 60, endpoint=True)
    R = 0.5
    points = np.transpose(np.array([R * np.cos(t), R * np.sin(t), a * t]))
    points_flip_y = np.flipud(np.transpose(np.array([R * np.cos(t), R * np.sin(t), a * t])))
    points_flip_y[:, 1] = -points_flip_y[:, 1]
    points = np.concatenate([points, points_flip_y], axis=0)
    return points


def generate_3d_example():
    n_gaussians = 1
    variance = np.square(1.0/n_gaussians)
    gmm = utils.get_grid_gmm(subdivisions=[n_gaussians, n_gaussians, n_gaussians], variance=variance)
    # points = np.array([[[0, 0, 0]]])
    points = get_3d_spiral()

    fig_images = []
    for i, point in enumerate(points):
        if point.ndim == 1:
            point = np.expand_dims(point, axis=0)
        pc_3dmfv = utils.get_3DmFV(point, gmm.weights_, gmm.means_, gmm.covariances_, normalize=True)
        fig_image = visualization.visualzie_3dmfv_gaussian_and_points(gmm, point, pc_3dmfv[0, :, :], display=False, export=False,
                                                                   export_path=OUTPUT_PATH + '/fv_3d' + str(i) + '.png')
        fig_images.append(fig_image)
    imageio.mimsave(OUTPUT_PATH + '/3d_fv.gif', fig_images)


def generate_3d_example_2():
    n_gaussians = 2
    variance = np.square(1.0/n_gaussians)
    gmm = utils.get_grid_gmm(subdivisions=[n_gaussians, n_gaussians, n_gaussians], variance=variance)
    points = utils.load_point_cloud_from_txt('car_130.txt')
    points = np.expand_dims(points, axis=0)
    pc_3dmfv = utils.get_3DmFV(points, gmm.weights_, gmm.means_, gmm.covariances_, normalize=True)
    fig_image = visualization.visualzie_3dmfv_gaussian_and_points(gmm, np.squeeze(points), pc_3dmfv[0, :, :], display=False,
                                                                  export=True,
                                                                  export_path=OUTPUT_PATH + '/fv_3d_model' + '.png')

if __name__ == "__main__":
    generate_3d_example_2()