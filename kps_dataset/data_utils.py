import cv2
import pdb
def draw_points(img, coord):
    point_size = 4
    point_color = (0, 0, 255)  # BGR
    points = []
    for one in coord:
        points.append((one[0], one[1]))
    for i, p in enumerate(points):
        if i < 5:
            color = (255, 0, 0)
        elif i >= 5 and i <= 8:
            color = (0, 0, 255)
        elif i >= 9 and i <= 12:
            color = (0, 255, 0)
        elif i >= 13 and i <= 16:
            color = (255, 255, 0)
        elif i >= 17 and i <= 20:
            color = (0, 255, 255)
        cv2.circle(img, (int(p[0]+0.5), int(p[1]+0.5)), point_size, color, -1, 4)
        # cv2.putText(crop_img, str(i), p, cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

        # cv2.imshow("1", img)
        # cv2.waitKey(0)

    # 画直线
    edges = [[0, 1], [1, 2], [2, 3], [3, 4],
             [0, 5], [5, 6], [6, 7], [7, 8],
             [0, 9], [9, 10], [10, 11], [11, 12],
             [0, 13], [13, 14], [14, 15], [15, 16],
             [0, 17], [17, 18], [18, 19], [19, 20]]
    for x, y in edges:
        if points[x][0] > 0 and points[x][1] > 0 and points[y][0] > 0 and points[y][1] > 0:
            cv2.line(img, points[x], points[y], point_color, 1)

    return img