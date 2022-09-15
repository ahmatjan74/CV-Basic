"""
1.	打开图像，显示图像，灰度化，二值化，存储图像，缩放图像，观察其分辨率，降低灰度分辨率两种模式，观察图像变化；
"""
import cv2


def show_save(title, img, dest):
    cv2.imshow(title, img)
    cv2.imwrite(dest, img)
    cv2.waitKey(0)


def reduce_intensity_levels(img, level):
    img = cv2.copyTo(img, None)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            si = img[x, y]
            ni = int(level * si / 255 + 0.5) * (255 / level)
            img[x, y] = ni
    return img


class DigitalImageProcessing:
    def __init__(self, task_name, img_path):
        self.task_name = task_name
        self.img_path = img_path

    def process_img(self):
        # open
        im_data = cv2.imread(self.img_path)
        # show
        cv2.imshow(self.task_name, im_data)
        cv2.waitKey(0)
        # resize ->
        width, height = im_data.shape[:2]
        img_small = cv2.resize(im_data, (width // 3, height // 2), interpolation=cv2.INTER_CUBIC)
        show_save(title='small', img=img_small, dest='./image/small.jpg')
        img_shrink = cv2.resize(im_data, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        show_save(title='shrink', img=img_shrink, dest='./image/shrink.jpg')
        _, img_thresh = cv2.threshold(im_data, 127, 255, cv2.THRESH_BINARY)
        show_save(title='thresh', img=img_thresh, dest='./image/thresh.jpg')

    def show_save_gray_img(self):
        im_data = cv2.imread(self.img_path)
        gray = cv2.cvtColor(im_data, cv2.COLOR_BGR2GRAY)
        show_save(title='gray', img=gray, dest='./image/gray.jpg')
        gray8 = reduce_intensity_levels(gray, 7)
        show_save(title='gray8', img=gray8, dest='./image/gray8.jpg')
        gray2 = reduce_intensity_levels(gray, 1)
        show_save(title='gray2', img=gray2, dest='./image/gray2.jpg')


image_path = './image/img2.jpg'
dip = DigitalImageProcessing(task_name='digital-image-processing', img_path=image_path)
dip.process_img()
dip.show_save_gray_img()
