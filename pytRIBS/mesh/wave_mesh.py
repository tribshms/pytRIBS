import pywt
import pyvista as pv
import rasterio
import numpy as np
from scipy.interpolate import griddata, RegularGridInterpolator


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
        max_avg_coeffs = []
        for level in range(1, self.maxlevel + 1):
            v = self.wavelet_packet['v' * level].data
            h = self.wavelet_packet['h' * level].data
            d = self.wavelet_packet['d' * level].data
            avg_coeffs = np.mean(np.abs([v, h, d]), axis=0)
            max_avg_coeffs.append(np.max(avg_coeffs))
        return max(max_avg_coeffs)
    def extract_points_from_significant_details(self, threshold):
        def process_level(level):
            v = self.wavelet_packet['v' * level].data
            h = self.wavelet_packet['h' * level].data
            d = self.wavelet_packet['d' * level].data

            r, c = v.shape
            dx = (self.bounds.right - self.bounds.left) / c
            dy = (self.bounds.top - self.bounds.bottom) / r

            norm_coeffs = np.maximum.reduce([np.abs(v), np.abs(h), np.abs(d)]) / self.normalizing_coeff
            sig_mask = norm_coeffs > (2 ** (-level + 1) * threshold)

            rows, cols = np.where(sig_mask)
            x_coords = self.bounds.left + cols * dx
            y_coords = self.bounds.top - rows * dy

            return zip(x_coords, y_coords)

        centers = set()
        for level in range(1, self.maxlevel + 1):
            centers.update(process_level(level))

        centers = np.array(list(centers))
        elevations = self.interpolate_elevations(centers)

        return np.column_stack((centers, elevations))

    def convert_coords_to_mesh(self, coords):
        point_cloud = pv.PolyData(coords)
        mesh = point_cloud.delaunay_2d()
        return mesh

    def interpolate_elevations(self, points):

        height, width = self.data.shape

        x = np.arange(width) * self.transform[0] + self.transform[2]
        y = np.arange(height) * self.transform[4] + self.transform[5]

        interpolator = RegularGridInterpolator((y, x), self.data, method='linear', bounds_error=False,
                                               fill_value=None)

        elevations = interpolator((points[:,1],points[:,0]))

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
