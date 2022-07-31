import cv2
import numpy as np
from tensorflow.keras.models import load_model

# algorithm of sudoku solve
def solve_su(bo):
    find = find_empty(bo)
    if not find:
        return True
    else:
        row, col = find
    for i in range(1, 10):
        if valid(bo, i, (row, col)):
            bo[row][col] = i
            if solve_su(bo):
                return True
            bo[row][col] = 0
    return False


def valid(bo, num, pos):
    # Check row
    for i in range(len(bo[0])):
        if bo[pos[0]][i] == num and pos[1] != i:
            return False
    # Check column
    for i in range(len(bo)):
        if bo[i][pos[1]] == num and pos[0] != i:
            return False
    # Check box
    box_x = pos[1] // 3
    box_y = pos[0] // 3
    for i in range(box_y*3, box_y*3 + 3):
        for j in range(box_x * 3, box_x*3 + 3):
            if bo[i][j] == num and (i,j) != pos:
                return False
    return True


def print_board(bo):
    for i in range(len(bo)):
        if i % 3 == 0 and i != 0:
            print("- - - - - - - - - - - - - ")
        for j in range(len(bo[0])):
            if j % 3 == 0 and j != 0:
                print(" | ", end="")
            if j == 8:
                print(bo[i][j])
            else:
                print(str(bo[i][j]) + " ", end="")


def find_empty(bo):
    for i in range(len(bo)):
        for j in range(len(bo[0])):
            if bo[i][j] == 0:
                return i, j  # row, col
    return None


def replace_points(outline):
    """
    :param outline: 4 points of contour
    :return: replaced point of contour by
    """
    outline = outline.reshape((4, 2))
    outline = outline.reshape((4, 2))
    new_outline = np.zeros((4, 1, 2), dtype=np.int32)
    add_coord = outline.sum(axis=1)  # 좌표별 x,y 합
    new_outline[0], new_outline[3] = outline[np.argmin(add_coord)], outline[np.argmax(add_coord)]
    diff_coord = np.diff(outline, axis=1)
    new_outline[1], new_outline[2] = outline[np.argmin(diff_coord)], outline[np.argmax(diff_coord)]
    return new_outline


def big_contour(contours):
    """
    :param contours: all contours in img
    :return: the biggest contour in img
    """
    outline = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 60:  # 최소 60이상의 크기를 가질 경우 진행
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                outline = approx
                max_area = area
    return outline, max_area


def split_boxes(img_warp_gray):
    """
    9x9 스도쿠의 칸을 하나씩 나누어 81개의 박스로 저장
    :param img_warp_gray: 기하학적 변환을 통해 만들어낸 스도쿠 사진 -> gray scale
    :return: boxes: 9x9칸의 박스들을 순서대로 저장해놓은 리스트
    """
    rows = np.vsplit(img_warp_gray, 9)  # 세로축으로 나눔(열))
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)
        # 가로축으로 나눔(행)
        for box in cols:
            boxes.append(box)
    return boxes  # boxes --> 9x9칸의 박스


def predict_number(boxes, model):
    """
    model predict
    :param boxes: 9x9칸의 박스들을 순서대로 저장해놓은 리스트
    :param model: 학습된 mnist 데이터 cnn 모델
    :return: 예측된 결과를 digit 0~9에서 반환(1~9 -> digit, 0 -> None)
    """
    predicted_result = []
    for image in boxes:
        img = np.asarray(image)
        img = img[5:img.shape[0] - 5, 5:img.shape[1] -5]
        img = cv2.resize(img, (28, 28))  # img 각각의 사진을 resize하여 입력값에 맞게 저장
        img = img / 255
        predictions = model.predict(img.reshape(1, 28, 28, 1))
        class_index = model.predict_classes(img.reshape(1, 28, 28, 1))
        probability_value = np.argmax(predictions, axis=-1)
        if probability_value > 0.8:
            predicted_result.append(class_index[0])  # 1~9까지의 digit
        else:
            predicted_result.append(0)  # 1~9까지의 digit가 아닌 None
    return predicted_result


def display_numbers(img, numbers, color=(0, 255, 0)):
    """
    :param img: img of original
    :param numbers: founded number in advance
    :param color: display color -> Green
    :return: image with numbers
    """
    secW = int(img.shape[1] / 9)
    secH = int(img.shape[0] / 9)
    for x in range(0, 9):
        for y in range(0, 9):
            if numbers[(y * 9) + x] != 0:
                cv2.putText(img, str(numbers[(y * 9) + x]), (x * secW + int(secW / 2) - 10, int((y + 0.8) * secH)),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, color, 2, cv2.LINE_AA)
    return img