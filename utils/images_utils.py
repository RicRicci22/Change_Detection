"""
Functions to handle images with respective ground truth
"""
from rasterio import features
from PIL import Image
import numpy as np
import shapely
import matplotlib.pyplot as plt
import rasterio
import cv2


def extract_blobs(mask_array):
    """
    Function that given a ground truth mask, use shapely to extract the blobs
    """
    # Find unique "colors" in the image
    temp_blobs = []

    unique_classes = np.unique(mask_array.reshape(-1, mask_array.shape[2]), axis=0)
    for class_ in unique_classes:
        if not (class_ == np.array([255, 255, 255])).all():
            all_blobs = []
            mask = (
                np.equal(mask_array.reshape(-1, mask_array.shape[2]), class_)
                .all(axis=1)
                .reshape(mask_array.shape[:2])
            )
            for shape, _ in features.shapes(
                mask.astype(np.int16),
                mask=(mask > 0),
                transform=rasterio.Affine(1.0, 0.0, 0.0, 0, 1.0, 0),
            ):
                all_blobs.append(shapely.geometry.shape(shape))

            all_blobs = shapely.geometry.MultiPolygon(all_blobs)
            if not all_blobs.is_valid:
                all_blobs = all_blobs.buffer(0)
                # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
                # need to keep it a Multi throughout
                if all_blobs.geom_type == "Polygon":
                    all_blobs = shapely.geometry.MultiPolygon([all_blobs])

            temp_blobs.append(all_blobs)

    final_blobs = []
    for polygon in temp_blobs:
        if polygon.geom_type == "MultiPolygon":
            for poly in list(polygon.geoms):
                final_blobs.append(poly)
        else:
            final_blobs.append(polygon)

    return final_blobs


def get_largest_poly(all_polygons: list):
    """
    Function that given a list of polygons, returns the largest one
    """
    # Get the polygon with the largest area
    if len(all_polygons) > 0:
        largest_polygon = sorted(all_polygons, key=lambda x: x.area, reverse=True)[0]
    else:
        return False

    return largest_polygon


def crop_image(image, polygon: shapely.geometry.Polygon = None):
    """
    Function that given an image and a shapely polygon, crop the image using the envelope of the polygon and return it
    """
    envelope = polygon.envelope
    coords = list(envelope.exterior.coords)[:-1]  # the last point is the first one
    # Crop the image
    crop = image.crop((coords[0][0], coords[0][1], coords[2][0], coords[2][1]))

    return crop


if __name__ == "__main__":
    image = Image.open(
        "D:/Riccardo/Datasets/segmentation/Semantic segmentation/second_dataset/test/im1/00005.png"
    )
    mask = Image.open(
        "D:/Riccardo/Datasets/segmentation/Semantic segmentation/second_dataset/test/label1/00005.png"
    )

    kernel_dim = 3
    kernel = np.ones((kernel_dim, kernel_dim), np.uint8)

    # Read the mask as a numpy array
    mask_array = np.array(mask)

    # Extract the polygon
    polygons = extract_blobs(mask_array)

    # Get the polygon with the largest area
    largest_polygon = get_largest_poly(polygons)

    crop = crop_image(image, largest_polygon)

    # Plot the image and the crop and save as figures
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image)
    ax[0].plot(*largest_polygon.envelope.exterior.xy, color="red")
    ax[0].axis("off")
    ax[1].imshow(crop)
    ax[1].axis("off")
    plt.savefig("crop.png")
