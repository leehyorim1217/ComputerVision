import cv2 as cv
import matplotlib.pyplot as plt

# 이미지 읽기
img = cv.imread(r'C:\Users\CIE2_11\Pictures\Saved Pictures\raichu.jpg')

# 이미지가 제대로 읽혔는지 확인
# R G B 하나씩 떼서 히스토그램 평활화 해보기
# 이후 각각 평활화된 3채널 병합
if img is None:
    print("이미지를 불러올 수 없습니다.")
    exit()

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

'''
plt.imshow(gray, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.show()
# 01. gray scale 이미지


h = cv.calcHist([gray], [0], None, [256], [0, 256])
plt.plot(h, color='r', linewidth=1)
plt.title("Original Histogram")
plt.show()
# 01. gray scale된 원본 이미지의 그래프
'''

equal = cv.equalizeHist(gray)

'''
plt.imshow(equal, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.show()
# 02. 히스토그램 평활화된 이미지



h = cv.calcHist([equal], [0], None, [256], [0, 256])
plt.plot(h, color='r', linewidth=1)
plt.title("Equalized Histogram")
plt.show()
# 02. 히스토그램 평활화된 이미지의 그래프
'''

img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.title("original")
plt.xticks([]), plt.yticks([])
plt.show()

#1. 각 채널 RGB 분리
b_channel = img[:, :, 0] #B채널 시각화
g_channel = img[:, :, 1] #G채널 시각화
r_channel = img[:, :, 2] #R채널 시각화

# 2. 각 채널 히스토그램 평활화
b_eq = cv.equalizeHist(b_channel)
g_eq = cv.equalizeHist(g_channel)
r_eq = cv.equalizeHist(r_channel)


# 평활화 +분리된 각 채널 시각화
plt.imshow(r_eq)
plt.title('Equalized R Channel')
plt.xticks([]), plt.yticks([])
plt.show()

plt.imshow(g_eq)
plt.title('Equalized G Channel')
plt.xticks([]), plt.yticks([])
plt.show()

plt.imshow(b_eq)
plt.title('Equalized B Channel')
plt.xticks([]), plt.yticks([])
plt.show()

# 3. 평활화된 채널 병합
merged_eq = cv.merge([b_eq, g_eq, r_eq])

# 4. 병합 이미지 BGR → RGB 변환 후 시각화
merged_eq_rgb = cv.cvtColor(merged_eq, cv.COLOR_BGR2RGB)

plt.imshow(merged_eq_rgb)
plt.title('Merged Equalized Image')
plt.axis('off')
plt.show()