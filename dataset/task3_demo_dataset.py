"""Temp file for dataset generation and encoding

TODO(Ruhao Tian): Refactor this file with pyarrow.
"""

import os
import dataclasses
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from PIL import Image
import pandas as pd


@dataclasses.dataclass
class GenerationInfo:
    """The parameter set for generation a new dataset.

    Attributes:
        texture_file_path (str): The file path to the texture image.
        spline_control_points (list[tuple[float, float]] | np.ndarray): The control points
        for the spline, either as a list or a ndarray. Each element is a tuple where the first
        number is the X coordinate and the second number is the Y coordinate.
        spline_sample_number (int): The number of samples to generate along the spline.
        sample_grid_width (float): The width of the sample grid to map texture on. Default is 1.0.
        Grid width will be scaled to 1 after generation.
        sample_grid_height (float): The height of the sample grid to map texture on. Default is 1.0.
        Grid height will be scaled to 1 after generation.
    """

    texture_file_path: str
    spline_control_points: list[tuple[float, float]] | np.ndarray
    spline_sample_number: int
    sample_grid_width: float = 1.0
    sample_grid_height: float = 1.0


@dataclasses.dataclass
class MaskInfo:
    """The mask information for the dataset.

    Attributes:
        mask_type (str): The type of the mask. Options are "coordinates", "colors", "full", "none".
        mask_ratio (float): The ratio of the mask. Default is 1.0.
    """

    mask_type: str = "none"
    mask_ratio: float = 1.0


class MinimalBTSPDataset:
    """A minimal multi-modal dataset for the BTSP project."""

    def __init__(self):
        # persistent info
        self._file_schema = ["x", "y", "r", "g", "b"]
        self._mask_types = ["coordinates", "colors", "full", "none"]
        # dataset info
        self._precise_raw_data: pd.DataFrame = None
        self._generation_info: GenerationInfo = None
        self._coordinate_precision = 8
        self._color_precision = 8
        self._mask_info: MaskInfo = MaskInfo()

    def clone(self):
        """Clone a new dataset from existing dataset.

        Returns:
            MinimalBTSPDataset: New dataset instance.
        """

        # create a new instance
        new_instance = MinimalBTSPDataset()
        # copy the precise raw data
        new_instance._precise_raw_data = self._precise_raw_data.copy()
        # copy the generation info
        new_instance._generation_info = self._generation_info
        # copy the coordinate and color precision
        new_instance._coordinate_precision = self._coordinate_precision
        new_instance._color_precision = self._color_precision
        # copy the mask info
        new_instance._mask_info = self._mask_info
        return new_instance

    def update_precision(self, coordinate_precision: int, color_precision: int):
        """Update color and coordinate precision.

        Args:
            coordinate_precision (int): new coordinate precision
            color_precision (int): new color precision
        """
        self._coordinate_precision = coordinate_precision
        self._color_precision = color_precision

    def update_mask_info(self, mask_info: MaskInfo):
        """Update the mask info.

        Args:
            mask_info (MaskInfo): the new mask info.
        """
        self._mask_info = mask_info

    def input_dimensions(self) -> int:
        """Calculate the dataset input dimension.

        Returns:
            int: input dimension.
        """
        return self._coordinate_precision * 2 + self._color_precision * 3

    def from_raw_data(self, raw_data: pd.DataFrame):
        """Load precise raw data from pandas dataframe

        Args:
            raw_data (pd.DataFrame): the raw data to load

        Returns:
            self: the dataset itself
        """
        self._precise_raw_data = raw_data
        return self

    def from_file(self, dataset_name: str, input_path: str = "./"):
        """Load the dataset from files."""

        data_file = os.path.join(input_path, dataset_name + ".csv")
        precise_raw_data = pd.read_csv(data_file)

        # check the schema of the input file
        if not precise_raw_data.columns.equals(pd.Index(self._file_schema)):
            raise ValueError(
                (
                    "Input file schema is incorrect. Expected:"
                    f"{self._file_schema}"
                    ", got:"
                    f"{precise_raw_data.columns}"
                )
            )

        self._precise_raw_data = precise_raw_data

        param_file = os.path.join(input_path, dataset_name + "_params.json")
        # TODO(Ruhao Tian): Specify encoding option for loading and saving JSON files
        with open(param_file, "r") as f:
            # load json file
            json_obj = json.load(f)
            # load generation info
            self._generation_info = GenerationInfo(**json_obj["generation_info"])
            # load mask info
            self._mask_info = MaskInfo(**json_obj["mask_info"])
            # load precision info
            self._coordinate_precision = json_obj["coordinate_precision"]
            self._color_precision = json_obj["color_precision"]

        return self

    def create_dataset(
        self,
        generation_params: GenerationInfo,
    ):
        """Generate a new dataset from a texture file and control points.
            Will normalize the positions to 0-1.

        Args:
            generation_params (GenerationInfo): The generation parameters.
        """
        spline_control_points = generation_params.spline_control_points
        number_of_samples = generation_params.spline_sample_number
        texture_file = generation_params.texture_file_path
        sample_grid_width = generation_params.sample_grid_width
        sample_grid_height = generation_params.sample_grid_height

        # Generate the Catmull-Rom spline
        spline_sampled_coordinates = self._sample_catmull_rom_spline(
            spline_control_points, number_of_samples
        )

        # Load and resize the texture
        texture_image = Image.open(texture_file)
        texture_image_array = np.array(texture_image)

        # sample colors from the texture
        sampled_colors = self._map_grid_to_texture_colors(
            spline_sampled_coordinates,
            texture_image_array,
            sample_grid_width,
            sample_grid_height,
        )

        # finalizing the dataset
        combined_spline_dataset = np.hstack(
            (spline_sampled_coordinates, sampled_colors)
        )
        # scale the positions to 0-1
        combined_spline_dataset[:, :2] /= np.array(
            [sample_grid_width, sample_grid_height]
        )

        self._precise_raw_data = pd.DataFrame(
            combined_spline_dataset, columns=["x", "y", "r", "g", "b"]
        )

        self._generation_info = generation_params

        return self

    def save_dataset(self, dataset_name: str, output_file_path: str = "./"):
        """Save the dataset to file."""
        if self._precise_raw_data is None:
            raise ValueError("No dataset loaded.")

        data_file = os.path.join(output_file_path, dataset_name + ".csv")
        param_file = os.path.join(output_file_path, dataset_name + "_params.json")

        # save the dataset data to a CSV file
        self._precise_raw_data.to_csv(data_file, index=False)
        print(f"Dataset data saved to {data_file}")

        # save the generation info to a JSON file
        with open(param_file, "w") as f:
            json_obj = {}
            # save generation info
            json_obj["generation_info"] = dataclasses.asdict(self._generation_info)
            # save mask info
            json_obj["mask_info"] = dataclasses.asdict(self._mask_info)
            # save precision info
            json_obj.update(
                {
                    "coordinate_precision": self._coordinate_precision,
                    "color_precision": self._color_precision,
                }
            )
            json.dump(json_obj, f)
        print(f"Generation info saved to {param_file}")

    def _sample_catmull_rom_spline(
        self, spline_control_points: list | np.ndarray, num_samples: int = 100
    ) -> np.ndarray:
        """Sample points from a Catmull-Rom spline defined by control points.

        Args:
            spline_control_points: list | np.ndarray, each element is a
                tuple of (x, y) coordinates of the control points.
            num_samples: int, number of points to sample from the spline.
        """

        # Extract x and y coordinates from control points
        spline_control_points = np.array(spline_control_points)
        x = spline_control_points[:, 0]
        y = spline_control_points[:, 1]

        # Generate parameter t
        t = np.linspace(0, 1, len(spline_control_points))

        # Create cubic spline for x and y coordinates
        cs_x = CubicSpline(t, x, bc_type="clamped")
        cs_y = CubicSpline(t, y, bc_type="clamped")

        # Generate uniformly spaced points
        t_fine = np.linspace(0, 1, num_samples)
        x_fine = cs_x(t_fine)
        y_fine = cs_y(t_fine)

        # Combine x and y coordinates
        sampled_spline_points = np.vstack((x_fine, y_fine)).T
        return sampled_spline_points

    def _map_grid_to_texture_colors(
        self,
        grid_sample_points: np.ndarray,
        texture_image_array: np.ndarray,
        sample_grid_width,
        sample_grid_height,
    ):
        """Map the grid sample points to the texture colors.

        Args:
            grid_sample_points: np.ndarray, shape (num_points, 2), each row
                is a (x, y) coordinate of the sample point.
            texture_array: np.ndarray, shape (height, width, 3), the texture image.
            grid_width: int, the width of the grid.
            grid_height: int, the height of the grid.
        """
        # obtain the height and width of the texture
        texture_height, texture_width, _ = np.shape(texture_image_array)
        # print(texture_height, texture_width)
        # scale the spline points to the texture size
        scaled_spline_points = grid_sample_points * np.array(
            [texture_width / sample_grid_width, texture_height / sample_grid_height]
        )
        sampled_texture_colors = []
        for scaled_point in scaled_spline_points:
            x, y = scaled_point
            x = int(np.clip(x, 0, texture_width - 1))
            y = int(np.clip(y, 0, texture_height - 1))
            color = texture_image_array[y, x]  # Note: (y, x) indexing for image
            sampled_texture_colors.append(color)
        return np.array(sampled_texture_colors)

    def to_binary_tensors(self, masked: bool = False) -> np.ndarray:
        """Convert the dataset to binary tensors, with precision loss."""
        if self._precise_raw_data is None:
            raise ValueError("No dataset loaded.")
        binary_tensors = np.empty(
            (0, self._coordinate_precision * 2 + self._color_precision * 3), dtype=int
        )
        # load the dataset and convert to numpy array
        x = self._precise_raw_data["x"].to_numpy()
        y = self._precise_raw_data["y"].to_numpy()
        r = self._precise_raw_data["r"].to_numpy()
        g = self._precise_raw_data["g"].to_numpy()
        b = self._precise_raw_data["b"].to_numpy()
        # iterate all the rows in the input file
        for row_index in range(self._precise_raw_data.shape[0]):
            # convert float to binary np array
            x_binary = np.binary_repr(
                int(x[row_index] * pow(2, self._coordinate_precision - 1)),
            )[
                : self._coordinate_precision
            ]  # neglect the first sign bit
            y_binary = np.binary_repr(
                int(y[row_index] * pow(2, self._coordinate_precision - 1)),
            )[: self._coordinate_precision]
            r_binary = np.binary_repr(int(r[row_index]))[
                : self._color_precision
            ]  # neglect the first sign bit
            g_binary = np.binary_repr(
                int(g[row_index]),
            )[: self._color_precision]
            b_binary = np.binary_repr(int(b[row_index]))[: self._color_precision]
            # check length and pad zeros at right
            x_binary = x_binary.zfill(self._coordinate_precision)
            y_binary = y_binary.zfill(self._coordinate_precision)
            r_binary = r_binary.zfill(self._color_precision)
            g_binary = g_binary.zfill(self._color_precision)
            b_binary = b_binary.zfill(self._color_precision)
            # concatenate the binary arrays
            binary_array = np.array(
                list(map(int, (x_binary + y_binary + r_binary + g_binary + b_binary)))
            )
            binary_array = binary_array > 0
            binary_tensors = np.vstack((binary_tensors, binary_array))

        if masked:
            binary_mask = self.get_mask()
            # mask binary tensors
            binary_tensors[binary_mask] = 0

        # print("Binary conversion finished")

        return binary_tensors

    def to_float_tensors(self, masked: bool = False) -> np.ndarray:
        """Convert the binary tensors back to float tensors, with precision loss."""
        result = np.empty((0, 5))
        binary_tensors = self.to_binary_tensors(masked)
        for row_index in range(binary_tensors.shape[0]):
            row = binary_tensors[row_index]
            # print(row)
            x = int("".join(map(str, row[: self._coordinate_precision])), 2) / pow(
                2, self._coordinate_precision - 1
            )
            y = int(
                "".join(
                    map(
                        str,
                        row[
                            self._coordinate_precision : self._coordinate_precision * 2
                        ],
                    )
                ),
                2,
            ) / pow(2, self._coordinate_precision - 1)
            r = int(
                "".join(
                    map(
                        str,
                        row[
                            self._coordinate_precision
                            * 2 : self._coordinate_precision
                            * 2
                            + self._color_precision
                        ],
                    )
                ),
                2,
            )
            g = int(
                "".join(
                    map(
                        str,
                        row[
                            self._coordinate_precision * 2
                            + self._color_precision : self._coordinate_precision * 2
                            + self._color_precision * 2
                        ],
                    )
                ),
                2,
            )
            b = int(
                "".join(
                    map(
                        str,
                        row[
                            self._coordinate_precision * 2
                            + self._color_precision * 2 : self._coordinate_precision * 2
                            + self._color_precision * 3
                        ],
                    )
                ),
                2,
            )
            result = np.vstack((result, np.array([x, y, r, g, b])))

        # print a sample of the result
        # print(result[:5])
        return result

    def to_raw_data(self) -> pd.DataFrame:
        """Output the dataset as a pandas DataFrame, no precision loss."""
        if self._precise_raw_data is None:
            raise ValueError("No dataset loaded.")
        return self._precise_raw_data.copy()

    def calculate_sparsity(self):
        """Calculate the sparsity of the dataset's binary array."""
        binary_tensors = self.to_binary_tensors()
        total_ones = np.sum(binary_tensors)
        total_elements = binary_tensors.size

        return total_ones / total_elements

    def _binary_tensor_to_float(self, input_binary_tensors: np.ndarray) -> pd.DataFrame:
        """Load the dataset from binary tensors."""

        input_binary_tensors = input_binary_tensors

        # check the shape of the input binary tensors
        correct_length = self._coordinate_precision * 2 + self._color_precision * 3
        if input_binary_tensors.shape[1] != correct_length:
            raise ValueError(
                (
                    "Input binary tensors have incorrect shape. Expected:"
                    f"{correct_length}, got:"
                    f"{input_binary_tensors.shape[1]}"
                )
            )

        # convert the binary tensors to a pyarrow table
        x_binary = input_binary_tensors[:, : self._coordinate_precision]
        y_binary = input_binary_tensors[
            :, self._coordinate_precision : self._coordinate_precision * 2
        ]
        r_binary = input_binary_tensors[
            :,
            self._coordinate_precision * 2 : self._coordinate_precision * 2
            + self._color_precision,
        ]
        g_binary = input_binary_tensors[
            :,
            self._coordinate_precision * 2
            + self._color_precision : self._coordinate_precision * 2
            + self._color_precision * 2,
        ]
        b_binary = input_binary_tensors[
            :,
            self._coordinate_precision * 2
            + self._color_precision * 2 : self._coordinate_precision * 2
            + self._color_precision * 3,
        ]
        x = np.array(
            [
                int("".join(map(str, row)), 2) / pow(2, self._coordinate_precision - 1)
                for row in x_binary
            ]
        )
        y = np.array(
            [
                int("".join(map(str, row)), 2) / pow(2, self._coordinate_precision - 1)
                for row in y_binary
            ]
        )
        r = np.array([int("".join(map(str, row)), 2) for row in r_binary])
        g = np.array([int("".join(map(str, row)), 2) for row in g_binary])
        b = np.array([int("".join(map(str, row)), 2) for row in b_binary])

        float_data = pd.DataFrame.from_dict(
            {
                "x": x,
                "y": y,
                "r": r,
                "g": g,
                "b": b,
            }
        )

        return float_data

    def _scatter_dataframe(self, df: pd.DataFrame, ax: plt.Axes, alpha: float = 1.0, scatter_kwargs: dict = None):
        """Plot the dataset."""
        x = df["x"].to_numpy()
        y = df["y"].to_numpy()
        color_r = df["r"].to_numpy() / 255
        color_g = df["g"].to_numpy() / 255
        color_b = df["b"].to_numpy() / 255
        ax.scatter(x, y, c=np.vstack((color_r, color_g, color_b)).T, alpha=alpha, **scatter_kwargs)

    def plot_dataset(
        self,
        display: bool = True,
        save_as: str = None,
        ax: plt.Axes = None,
        external_tensors: np.ndarray = None,
        raw_data_alpha: float = 1,
        control_points: bool = True,
        selected_points: list[bool] = None,
        scatter_kwargs: dict = {},
    ):
        """Plot the dataset."""
        if ax is None:
            fig, ax = plt.subplots()

        if external_tensors is not None:
            external_data = self._binary_tensor_to_float(external_tensors)
            self._scatter_dataframe(external_data, ax, alpha=1)

        if selected_points is not None:
            # input check
            if len(selected_points) != self._precise_raw_data.shape[0]:
                raise ValueError(
                    (
                        f"The length of the selected points({len(selected_points)}) should be equal to the number of points({self._generation_info.spline_sample_number}) in the dataset."
                    )
                )
            # select dataframe rows
            selected_data = self._precise_raw_data[selected_points]
        else:
            selected_data = self._precise_raw_data

        # Plot the sampled points
        self._scatter_dataframe(selected_data, ax, alpha=raw_data_alpha, scatter_kwargs=scatter_kwargs)

        # Plot the control points
        if self._generation_info is None:
            print("No generation information found. Skipping.")
        elif control_points is False:
            pass
        else:
            control_x = [
                point[0] for point in self._generation_info.spline_control_points
            ]
            control_y = [
                point[1] for point in self._generation_info.spline_control_points
            ]
            # scale the control points to 0-1
            control_x = np.array(control_x) / self._generation_info.sample_grid_width
            control_y = np.array(control_y) / self._generation_info.sample_grid_height
            ax.plot(
                control_x,
                control_y,
                "ro--",
                label="Control Points",
                alpha=raw_data_alpha,
            )

        if save_as is not None:
            plt.savefig(save_as)

        if display:
            plt.show()

        return ax

    def _set_coordinates_mask(self):
        """Mask the coordinates of the dataset."""

        pattern_number = self._generation_info.spline_sample_number
        mask_ratio = self._mask_info.mask_ratio

        # mask the coordinates
        # True means masked
        binary_mask = np.random.choice(
            [False, True],
            size=(pattern_number, self._coordinate_precision * 2),
            p=[1 - mask_ratio, mask_ratio],
        )
        # extend the mask to match the dataset shape
        # the extended part should be False
        mask_extend = np.zeros((pattern_number, self._color_precision * 3), dtype=bool)
        # concatenate the mask
        binary_mask = np.concatenate((binary_mask, mask_extend), axis=1)
        return binary_mask

    def _set_colors_mask(self):
        """Mask the colors of the dataset."""

        pattern_number = self._generation_info.spline_sample_number
        mask_ratio = self._mask_info.mask_ratio

        # mask the colors
        # True means masked
        binary_mask = np.random.choice(
            [False, True],
            size=(pattern_number, self._color_precision * 3),
            p=[1 - mask_ratio, mask_ratio],
        )
        # extend the mask to match the dataset shape
        # the extended part should be False
        mask_extend = np.zeros(
            (pattern_number, self._coordinate_precision * 2), dtype=bool
        )
        # concatenate the mask
        binary_mask = np.concatenate((mask_extend, binary_mask), axis=1)
        return binary_mask

    def _set_full_mask(self):
        """Mask the full dataset."""

        pattern_number = self._generation_info.spline_sample_number
        mask_ratio = self._mask_info.mask_ratio

        # mask the full dataset
        # True means masked
        binary_mask = np.random.choice(
            [False, True],
            size=(
                pattern_number,
                self._coordinate_precision * 2 + self._color_precision * 3,
            ),
            p=[1 - mask_ratio, mask_ratio],
        )
        return binary_mask

    def get_mask(self):
        """Get the mask for the dataset."""
        # check mask info
        if self._mask_info is None:
            raise ValueError("No mask info found.")

        mask_type = self._mask_info.mask_type

        match mask_type:
            case "coordinates":
                return self._set_coordinates_mask()
            case "colors":
                return self._set_colors_mask()
            case "full":
                return self._set_full_mask()
            case "none":
                return np.zeros(
                    (
                        self._generation_info.spline_sample_number,
                        self.input_dimensions(),
                    ),
                    dtype=bool,
                )

    # DEPRECRATED METHOD
    # def from_float_tensors(self, input_float_tensors: np.ndarray):
    #     """Load the dataset from float tensors."""
    #     if self.binary_tensors is not None or self.precise_raw_data is not None:
    #         print("Warning: Overwriting existing dataset.")

    #     # check the shape of the input binary tensors
    #     correct_length = 5
    #     if input_float_tensors.shape[1] != correct_length:
    #         raise ValueError(
    #             (
    #                 "Input float tensors have incorrect shape. Expected:"
    #                 f"{correct_length}, got:"
    #                 f"{input_float_tensors.shape[1]}"
    #             )
    #         )

    #     self.precise_raw_data = pd.DataFrame.from_dict(
    #         {
    #             "x": input_float_tensors[:, 0],
    #             "y": input_float_tensors[:, 1],
    #             "r": input_float_tensors[:, 2],
    #             "g": input_float_tensors[:, 3],
    #             "b": input_float_tensors[:, 4],
    #         }
    #     )


if __name__ == "__main__":

    # Define the 2D grid limits
    GRID_WIDTH = 6
    GRID_HEIGHT = 6
    NUM_POINTS = 100

    # Define control points for the Catmull-Rom spline
    control_points = [(1, 1), (2, 3), (4, 4), (5, 1), (6, 3)]

    # Generate the dataset
    example = MinimalBTSPDataset()
    example.update_precision(8, 8)
    example_generation_info = GenerationInfo(
        texture_file_path="image.png",
        spline_control_points=control_points,
        spline_sample_number=NUM_POINTS,
        sample_grid_width=GRID_WIDTH,
        sample_grid_height=GRID_HEIGHT,
    )
    example.create_dataset(example_generation_info)

    dataset = example._precise_raw_data

    # print("Sampled dataset:")
    print(dataset[:5])  # Print the first 5 rows of the dataset

    # plot raw data
    example.plot_dataset(display=False, save_as="test_raw_data.png")

    # test loading and saving
    example.save_dataset("test_dataset")
    example.from_file("test_dataset")

    # test binary conversion
    for mask in example._mask_types:
        print(f"Testing mask type: {mask}")
        example_mask_info = MaskInfo(mask_type=mask, mask_ratio=1.0)
        example.update_mask_info(example_mask_info)
        tensors = example.to_binary_tensors(masked=True)
        print(tensors.shape)
        example.plot_dataset(
            display=False,
            save_as=f"test_{mask}_masked.png",
            external_tensors=tensors,
            raw_data_alpha=0.5,
        )
