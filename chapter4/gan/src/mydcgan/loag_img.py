import os
import glob

import cv2


def load_imgs(tol=100000):
    img_paths = glob.glob(
        os.path.dirname(
            os.path.abspath(__file__)
        ) + "\\dataset\\cartoonset100k\\*\\*.png"
    )

    c = 0
    for img_path in img_paths:
        if c > tol:
            break
        img = cv2.imread(img_path)
        img = img[80:420, 80:420]
        img = cv2.resize(img, (128, 128))
        cv2.imwrite(
            os.path.dirname(
                os.path.abspath(__file__)
            ) + "\\dataset64\\img_" + str(c) + ".png", img)

        if c % 1000 == 0:
            print(c)
        c += 1


if __name__ == '__main__':
    load_imgs()
