import math
import traceback
from typing import List

import cv2
import d3dshot
import numpy as np
from cv2 import DMatch
from numpy import ndarray


class CustomErr(Exception):
    pass


d3dshot_mgr = d3dshot.create(capture_output='numpy')

color_board_img = cv2.imread('imgs/cb.png')
draw_board_img = cv2.imread('imgs/db.png')
pen_setting_img = cv2.imread('imgs/ps.png')


class Color:
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)


def to_int(data):
    if isinstance(data, (list, tuple, set, ndarray)):
        return [int(item) for item in data]
    elif isinstance(data, (str, float, int)):
        return int(data)
    else:
        raise NotImplementedError(f'{type(data)} is not supported')


def distance(p1, p2):
    return math.sqrt(
        ((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2)
    )


def draw_rectangle(img, pt1, pt2, color=Color.GREEN, thickness=2, text=None):
    cv2.rectangle(img, pt1, pt2, color=color, thickness=thickness)
    if text:
        cv2.putText(img, text, pt1, fontFace=3, fontScale=1, color=color)


def find_pos(sub_img, src_img, text, tolerance=0.5, method='sift', grey_mode=False, preview=1):
    if grey_mode:
        src = cv2.cvtColor(src_img.copy(), cv2.COLOR_RGB2GRAY)
        sub = cv2.cvtColor(sub_img.copy(), cv2.COLOR_RGB2GRAY)
    else:
        src = src_img.copy()
        sub = sub_img.copy()

    src_h, src_w, *_ = src.shape
    sub_h, sub_w, *_ = sub.shape

    if method == 'legacy':
        # Legacy template matching
        result = cv2.matchTemplate(sub, src, cv2.TM_CCOEFF_NORMED)
        y, x = np.unravel_index(result.argmax(), result.shape)
        cv2.rectangle(src, (x, y), (x + sub_w, y + sub_h), color=(0, 255, 0), thickness=5)
        preview = src_img
        pt1 = (x, y)
        pt2 = (x + src_w, y + src_h)
    elif method == 'sift':
        # SIFT
        sift = cv2.SIFT_create()
        src_kp, src_des = sift.detectAndCompute(src, None)
        sub_kp, sub_des = sift.detectAndCompute(sub, None)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(sub_des, src_des, k=2)
        good: List[List[DMatch]] = []
        for m, n in matches:
            if m.distance < tolerance * n.distance:
                good.append([m])
        try:
            # Calculate offset and Draw final point
            if len(good) < 2:
                raise CustomErr(f'{text} feature point not enough ({len(good)})')

            x, y = to_int(src_kp[good[0][0].trainIdx].pt)
            offset_x, offset_y = to_int(sub_kp[good[0][0].queryIdx].pt)
            factor = np.divide(
                distance(to_int(src_kp[good[0][0].trainIdx].pt), to_int(src_kp[good[1][0].trainIdx].pt)),
                distance(to_int(sub_kp[good[0][0].queryIdx].pt), to_int(sub_kp[good[1][0].queryIdx].pt))
            )
            if np.isnan(x) or np.isnan(y):
                raise CustomErr(f'{text}:Invalid raw position {x, y}')
            if np.isnan(offset_x) or np.isnan(offset_y):
                raise CustomErr(f'{text}:Invalid offset {offset_x, offset_y}')
            if np.isnan(factor):
                raise CustomErr(f'{text}:Invalid factor {factor}')

            pt1 = (x - offset_x * factor), (y - offset_y * factor)
            pt2 = np.add(pt1, [sub_w * factor, sub_h * factor])
            draw_rectangle(src_img, to_int(pt1), to_int(pt2), thickness=5, text=text)

            # Draw feature point
            if preview > 1:
                preview = cv2.drawMatchesKnn(sub_img, sub_kp, src_img, src_kp, good[:2], None, flags=2)
            else:
                preview = src_img
        except CustomErr as e:
            print(e)
            preview = src_img
            pt1 = pt2 = 0
        except Exception:
            print(traceback.format_exc())
            preview = src_img
            pt1 = pt2 = 0
    else:
        raise NotImplementedError

    # Show preview
    show(preview)
    return pt1, pt2


def show(img, width=1280, height=720, title='Image'):
    cv2.imshow(title, cv2.resize(img, (width, height)))


def screenshot():
    screen_shot = d3dshot_mgr.screenshot()
    screen_shot = cv2.cvtColor(screen_shot, cv2.COLOR_BGR2RGB)
    return screen_shot


if __name__ == '__main__':
    while True:
        screen = screenshot()
        find_pos(color_board_img, screen, text='Color', preview=2)
        # find_pos(draw_board_img, screen, tolerance=0.9, text='Draw', grey_mode=True)
        # find_pos(pen_setting_img, screen, tolerance=0.9, text='Pen', grey_mode=True)
        cv2.waitKey(1)
