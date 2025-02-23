import cv2
import numpy as np


def find_way_from_maze(image: np.ndarray) -> tuple:
    """
    Найти путь через лабиринт.

    :param image: изображение лабиринта
    :return: координаты пути из лабиринта в виде (x, y), где x и y - это массивы координат
    """
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cur_pos = (0, 0)
    while gray_image[cur_pos] != 255:
        cur_pos = (cur_pos[0], cur_pos[1] + 1)
    
    turn_matrix = np.full(gray_image.shape, np.inf)
    
    do_next_turn(gray_image, turn_matrix, cur_pos, -1)

    return repare_path(turn_matrix)


def do_next_turn(image: np.ndarray, turn_matrix: np.ndarray, cur_pos: tuple[int, int], cur_turn: int) -> None:
    size_y, size_x = turn_matrix.shape
    y, x = cur_pos

    if image[cur_pos] == 0:
        return
    
    if turn_matrix[cur_pos] != np.inf:
        return
    
    turn_matrix[cur_pos] = cur_turn + 1
    
    for i in range(-1, 2):
        for j in range(-1, 2):
            if ((0 <= (x + i) < size_x) and (0 <= (y + j) < size_y)):
                pos = (y + j, x + i)
                do_next_turn(image, turn_matrix, pos, cur_turn + 1)


def repare_path(turn_matrix: np.ndarray) -> list[list[int], list[int]]:
    size_y, size_x = turn_matrix.shape

    cur_pos = turn_matrix.shape
    cur_pos = (turn_matrix.shape[0] - 1, 0)
    while turn_matrix[cur_pos] == np.inf:
        cur_pos = (cur_pos[0], cur_pos[1] + 1)
    
    coords = [[cur_pos[0]], [cur_pos[1]]]
    while turn_matrix[cur_pos] != 0:
        y, x = cur_pos
        flag = False
        for i in range(-1, 2):
            for j in range(-1, 2):
                if ((0 <= (x + i) < size_x) and (0 <= (y + j) < size_y)):
                    pos = (y + j, x + i)
                    if turn_matrix[pos] < turn_matrix[cur_pos]:
                        cur_pos = pos
                        flag = True
                        break
            if flag:
                break
        
        coords[0].append(pos[0])
        coords[1].append(pos[1])
    
    return coords