import cv2
import numpy as np
from sudoku_func import *
from tensorflow.keras.models import load_model

model = load_model('CNN_mnist_digitAI/mnist_cnn_model.h5')
img = cv2.imread("sudoku_resource/Sudoku_example.png")
img = cv2.resize(img, (450, 450))
img_blank = np.zeros((450, 450, 3), np.uint8)

# 전처리
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
img_Threshold = cv2.adaptiveThreshold(img_blur, 255, 1, 1, 11, 2)

# 컨투어 찾기
img_all_Contours = img.copy()
img_big_Contour = img.copy()
contours, hierarchy = cv2.findContours(img_Threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img_all_Contours, contours, -1, (0, 255, 0), 2)

# 가장 큰 컨투어 찾기
outline, max_area = big_contour(contours)
print(f"max_area = {max_area}, outline = {outline}")

# 기하학적 변환을 위한 좌표 정렬
if outline is not 0:
    new_outline = replace_points(outline)

    cv2.drawContours(img_big_Contour, new_outline, -1, (255, 0, 0), 5)
    pts1 = np.float32(outline)
    pts2 = np.float32([[0, 0], [0, 450], [450, 450], [450, 0]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp_Color = cv2.warpPerspective(img, matrix, (450, 450))
    img_digit_detected = img_blank.copy()
    img_warp_gray = cv2.cvtColor(img_warp_Color, cv2.COLOR_BGR2GRAY)

    img_digit_solved = img_blank.copy()
    boxes = split_boxes(img_warp_gray)

    predicted_numbersList = predict_number(boxes, model)
    print(predicted_numbersList)
    img_digit_detected = display_numbers(img_digit_detected, predicted_numbersList, color=(0, 0, 255))
    predicted_numbersList = np.asarray(predicted_numbersList)
    # 스도쿠에서 풀어야되는 빈칸을 1로 만들어 표시
    pos_array = np.where(predicted_numbersList > 0, 0, 1)
    print(f"pos_array = \n {pos_array}")

    board_array = np.array_split(predicted_numbersList, 9)
    print(board_array)

    try:
        solve_su(board_array)
    except:
        pass

    list_flatten = []
    for sub_list in board_array:
        for item in sub_list:
            list_flatten.append(item)
    solved_number = list_flatten * pos_array # 초기 숫자는 0으로 초기화
    img_digit_solved = display_numbers(img_digit_solved, solved_number)

cv2.imshow("img_warp_gray", img_warp_gray)
cv2.imshow("img_digit_detected", img_digit_detected)
cv2.imshow("img_digit_solved", img_digit_solved)
cv2.waitKey(0)
cv2.destroyAllWindows()
