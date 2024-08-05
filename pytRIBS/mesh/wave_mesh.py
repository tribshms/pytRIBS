import math

import numpy as np
import rasterio
import pywt
import pyvista as pv
from affine import Affine
import rasterio
import numpy as np
from scipy.interpolate import griddata
from rasterio.transform import from_origin
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import Delaunay


class WaveletTree:
    def __init__(self, path_to_raster, maxlevel=None):
        self.normalizing_coeff = None
        self.raster = path_to_raster
        self.maxlevel = maxlevel
        self.extract_raster_and_wavelet_info()

    def extract_raster_and_wavelet_info(self):
        with rasterio.open(self.raster) as src:
            self.data = src.read(1)  # Read the first band
            self.transform = src.transform
            self.bounds = src.bounds
            self.width = src.width
            self.height = src.height

        self.wavelet_packet = pywt.WaveletPacket2D(data=self.data, wavelet='db1',
                                                   maxlevel=self.maxlevel)
        # update maxlevel incase it's none
        self.maxlevel = self.wavelet_packet.maxlevel
        self.normalizing_coeff = self.find_max_average_coeffs()

    def get_extent(self):

        x_min = self.bounds.left  # this might be off
        x_max = self.bounds.right
        y_min = self.bounds.bottom
        y_max = self.bounds.top

        return x_min, x_max, y_min, y_max

    def find_max_average_coeffs(self):
        max_avg_coeffs_per_level = []

        for level in range(1, self.maxlevel + 1):
            # extract detail coeffs for given level
            v = self.wavelet_packet['v' * level].data
            h = self.wavelet_packet['h' * level].data
            d = self.wavelet_packet['d' * level].data

            r, c = np.shape(v)

            avg_coefs = [np.mean(np.abs([v[x, y], h[x, y], d[x, y]])) for y in range(0, c) for x in range(0, r)]
            max_avg_coeffs_per_level.append(max(avg_coefs))

        return max(max_avg_coeffs_per_level)

    def compute_normalized_coefficients(self, error_threshold):

        sig_masks_list = []

        thresh_exp = 0

        def normalize_detail_coefficients(h_ij, v_ij, d_ij):
            max_detail = np.max(np.abs([h_ij, v_ij, d_ij]))
            normalized_detail = max_detail / self.normalizing_coeff
            return normalized_detail

        for level in range(1, self.maxlevel + 1):
            v = self.wavelet_packet['v' * level].data
            h = self.wavelet_packet['h' * level].data
            d = self.wavelet_packet['d' * level].data

            threshold = 2 ** (thresh_exp) * error_threshold

            r, c = np.shape(h)
            norm_coeffs = np.array([[normalize_detail_coefficients(h[x, y], v[x, y], d[x, y]) for y in range(0, c)]
                                    for x in range(0, r)])

            sig_mask = np.array([[norm_coeffs[x, y] > threshold for y in range(0, c)]
                                 for x in range(0, r)])

            sig_masks_list.append(sig_mask)

            thresh_exp += -1

        return sig_masks_list

    def extract_points_from_significant_details(self, threshold):

        def rowcol_to_xy(row, col, dx, dy):
            x = (self.bounds.left + col * dx)  # + dx / 2  # center
            y = (self.bounds.top - row * dy)  # + dy / 2  # center

            return x, y

        sig_masks_list = self.compute_normalized_coefficients(threshold)
        centers = []

        for level in range(1, self.maxlevel + 1):
            sig_mask = sig_masks_list[level - 1]
            r, c = np.shape(self.wavelet_packet['a' * level].data)
            dx = (self.bounds.right - self.bounds.left) / c
            dy = (self.bounds.top - self.bounds.bottom) / r
            rows, cols = np.where(sig_mask)
            coords = [rowcol_to_xy(r, c, dx, dy) for r, c in zip(rows, cols)]
            centers.extend(coords)

        centers = np.array(list(set(centers)))  # ensure no duplicates

        elevations = self.interpolate_elevations(centers)
        # with rasterio.open(self.raster) as src:
        #     elevations = [val[0] * 3 for val in src.sample(centers)]

        return np.column_stack((centers, elevations))

    def convert_coords_to_mesh(self, coords):
        point_cloud = pv.PolyData(coords)
        mesh = point_cloud.delaunay_2d()
        return mesh

    def interpolate_elevations(self, points):
        # Extract original DEM data and its transform parameters
        original_data = self.data
        original_transform = self.transform
        original_height, original_width = original_data.shape

        # Create coordinate grids from the original DEM
        x = np.arange(original_width) * original_transform[0] + original_transform[2]
        y = np.arange(original_height) * original_transform[4] + original_transform[5]

        # Create meshgrid for interpolation
        grid_x, grid_y = np.meshgrid(x, y)
        grid_z = original_data.flatten()

        # Flatten the grid coordinates for griddata
        points_grid = np.vstack((grid_x.flatten(), grid_y.flatten())).T

        # Interpolate elevation for given points
        elevations = griddata(points_grid, grid_z, points, method='linear')

        return elevations

    @staticmethod
    def plot_mesh(mesh):
        plotter = pv.Plotter()
        plotter.add_mesh(mesh)
        plotter.show()

    # def fill_tree(self, sig_masks_list):
    #     xmin = self.bounds.left
    #     ymax = self.bounds.top
    #     self.root_cell.subdivide(sig_masks_list, xmin, ymax)
    #
    # def plot_cells(self, with_data=True):
    #     fig, ax = plt.subplots()
    #     if with_data:
    #         ax.imshow(self.data, extent=self.get_extent())
    #
    #     def plot_cell(ax, cell, color='blue'):
    #
    #         rect = patches.Rectangle(
    #             (cell.ul_x, cell.ul_y - cell.dy), cell.dx, cell.dy,
    #             linewidth=1, edgecolor=color, facecolor='none'
    #         )
    #         ax.add_patch(rect)
    #
    #         for child in cell.children:
    #             plot_cell(ax, child, color='red')
    #
    #     plot_cell(ax, self.root_cell)
    #
    #     ax.set_xlim(self.bounds.left, self.bounds.right)
    #     ax.set_ylim(self.bounds.bottom, self.bounds.top)
    #     ax.set_aspect('equal', 'box')
    #     ax.set_xlabel('X (UTM)')
    #     ax.set_ylabel('Y (UTM)')
    #     ax.set_title('Wavelet Cells')
    #
    #     plt.show()
    #
    # def convert_to_mesh(self, elevation_nan_value=-999999.0, exaggeration=3):
    #     points = self.extract_points(self.root_cell)
    #     points = np.array(list(set(points)))
    #
    #     mask = self.points_within_original_dem(points)
    #     points = points[mask]
    #
    #     elevations = self.extract_elevation_values(points)
    #
    #     mask = elevations != elevation_nan_value
    #     elevations = elevations[mask]
    #     points = points[mask]
    #
    #     points_3d = [(points[i, 0], points[i, 1], elevations[i] * exaggeration) for i in range(len(elevations))]
    #
    #     point_cloud = pv.PolyData(points_3d)
    #     mesh = point_cloud.delaunay_2d()
    #
    #     return mesh
    #
    # def extract_points(self, cell):
    #     points = []
    #
    #     if self.cell_intersects_original_dem(cell):
    #         points.extend([
    #             (cell.ul_x, cell.ul_y),
    #             (cell.ul_x + cell.dx, cell.ul_y),
    #             (cell.ul_x, cell.ul_y - cell.dy),
    #             (cell.ul_x + cell.dx, cell.ul_y - cell.dy)
    #         ])
    #
    #         #points.append((cell.ul_x + cell.dx / 2, cell.ul_y - cell.dy / 2))
    #
    #     # If this cell has children, recursively extract points from them
    #     if cell.children:
    #         for child in cell.children:
    #             points.extend(self.extract_points(child))
    #
    #     return points
    #
    # def cell_intersects_original_dem(self, cell):
    #     original_bounds = self.dem_original['bounds']
    #     return (cell.ul_x < original_bounds.right and
    #             cell.ul_x + cell.dx > original_bounds.left and
    #             cell.ul_y > original_bounds.bottom and
    #             cell.ul_y - cell.dy < original_bounds.top)
    #
    # def points_within_original_dem(self, points):
    #     original_bounds = self.dem_original['bounds']
    #     mask = ((points[:, 0] >= original_bounds.left) &
    #             (points[:, 0] <= original_bounds.right) &
    #             (points[:, 1] >= original_bounds.bottom) &
    #             (points[:, 1] <= original_bounds.top))
    #     return mask
    #
    # def extract_elevation_values(self, points):
    #     coords = [(points[row, 0], points[row, 1]) for row in range(0, len(points))]
    #
    #     with rasterio.open(self.raster) as src:
    #         elevations = [val[0] for val in src.sample(coords)]
    #
    #     return np.array(elevations)
    #
    # @staticmethod
    # def plot_mesh(mesh):
    #     plotter = pv.Plotter()
    #     plotter.add_mesh(mesh)
    #     plotter.show()
