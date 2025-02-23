import cv2
import numpy as np


def find_road_number(image: np.ndarray) -> int:
    """
    Найти номер дороги, на которой нет препятсвия в конце пути.

    :param image: исходное изображение
    :return: номер дороги, на котором нет препятсвия на дороге
    """

    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lanes_info = find_lanes(image_hsv)

    car_lane = find_car(lanes_info)
    free_lane = find_free_lane(image_hsv, lanes_info)

    return free_lane if car_lane != free_lane else -1


def find_lanes(image_hsv: np.ndarray) -> list[np.ndarray]:
    lane_low = (20, 50, 50)
    lane_high = (40, 250, 255)

    lane_mask = cv2.inRange(image_hsv, lane_low, lane_high)
    lane_indeces = []

    i = 0
    while i < lane_mask.shape[1]:
        elem = lane_mask[0][i]
        if elem: 
            i += 1
            continue

        lane_indeces.append(i)
        while not elem:
            i += 1
            elem = lane_mask[0][i]
        lane_indeces.append(i)
    
    return np.hsplit(image_hsv, lane_indeces)[1::2]


def find_car(lanes_info: list[np.ndarray]) -> int:
    car_low = (100, 50, 50)
    car_high = (120, 250, 255)

    for i in range(len(lanes_info)):
        lane = lanes_info[i]
        car_mask = cv2.inRange(lane, car_low, car_high)
        if np.any(car_mask):
            return i

    return -1


def find_free_lane(lanes_info: np.ndarray) -> int:
    barrier_low = (0, 50, 50)
    barrier_high = (10, 250, 255)

    for i in range(len(lanes_info)):
        lane = lanes_info[i]
        barrier_mask = cv2.inRange(lane, barrier_low, barrier_high)
        if np.any(barrier_mask):
            continue

        return i

    return -1
