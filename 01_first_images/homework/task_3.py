import cv2
import numpy as np


def rotate(image: np.ndarray, point: tuple, angle: float) -> np.ndarray:
    """
    Повернуть изображение по часовой стрелке на угол от 0 до 360 градусов и преобразовать размер изображения.

    :param image: исходное изображение
    :param point: значение точки (x, y), вокруг которой повернуть изображение
    :param angle: угол поворота
    :return: повернутное изображение
    """

    h, w, _ = image.shape
    x, y = point

    r1 = np.array((0, 0, 1)).T
    r2 = np.array((w, 0, 1)).T
    r3 = np.array((0, h, 1)).T
    r4 = np.array((w, h, 1)).T

    M1 = cv2.getRotationMatrix2D((x, y), angle, scale=1.0)

    r1 = np.array(M1) @ r1
    r2 = np.array(M1) @ r2
    r3 = np.array(M1) @ r3
    r4 = np.array(M1) @ r4

    mx = np.max([r1, r2, r3, r4], axis=0)
    mn = np.min([r1, r2, r3, r4], axis=0)
    print(r1, r2, r3, r4, mn, mx)

    scale = mx - mn

    M1 = cv2.getRotationMatrix2D((x, y), angle, scale=1.0)
    M1[0, 2] -= mn[0]
    M1[1, 2] -= mn[1]

    image = cv2.warpAffine(image, M1, (int(scale[0]), int(scale[1])))

    return image


def normalize_image(image_rgb: np.ndarray) -> np.ndarray:
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    mask = get_mask(image_hsv)
    anchor = get_anchor_points(mask)

    size = np.linalg.norm(anchor[3] - anchor[0]), np.linalg.norm(anchor[1] - anchor[0])
    new_points = np.float32([[size[0], size[1]], [size[0], 0], [0, 0], [0, size[1]]])

    M1 = cv2.getPerspectiveTransform(anchor, new_points)

    normalized_image = cv2.warpPerspective(image_rgb, M1, (int(size[0]), int(size[1])))

    return normalized_image


def get_mask(image_hsv: np.ndarray) -> np.ndarray:
    notebook_low = (0, 50, 50)
    notebook_high = (5, 250, 255)

    return cv2.inRange(image_hsv, notebook_low, notebook_high)


def get_anchor_points(mask: np.ndarray) -> np.ndarray:
    indices = np.argwhere(mask == 255)
    anchor_ind = *np.argmax(indices, axis=0), *np.argmin(indices, axis=0)
    anchor = map(lambda x: indices[x][::-1], anchor_ind)
    
    return np.float32(list(anchor))
