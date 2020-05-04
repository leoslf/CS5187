from streetnumber.detection import * 
import itertools
import cv2

if __name__ == "__main__":
    img = cv2.imread("train/1.png")

    regions = Regions((32, 32, 1), img)

    img_with_bboxes = img.copy()

    colors = itertools.cycle(itertools.product([0, 255], repeat = 3))
    # print (colors)
    starts, ends = regions.bboxes_opencv_order
    for i, (a, b, c, r, color) in enumerate(zip(starts, ends, regions.bbox_centers, regions.bbox_radius, colors)):
        # color = next(colors)
        print (color)
        img_with_bboxes = cv2.rectangle(img_with_bboxes, tuple(map(int, a)), tuple(map(int, b)), color, 2)
        img_with_bboxes = cv2.circle(img_with_bboxes, tuple(map(int, c)), int(r), color, 2)
        img_with_bboxes = cv2.circle(img_with_bboxes, tuple(map(int, c)), 1, color, -1)
        img_with_bboxes = cv2.putText(img_with_bboxes, str(i), tuple(map(int, a)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=2)

    cv2.imshow("img with bboxes", img_with_bboxes)
    cv2.waitKey(0)

