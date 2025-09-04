import cv2
import numpy as np

def find_max_connected_region(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) ## _, 是阈值，binary 是二值化后的图像

    visited = np.zeros_like(binary, dtype=bool) ## visited 是一个和 binary 大小相同的数组，用来记录是否访问过，zreo_like 是生成和 binary 大小相同的数组，元素都是 0
    max_area = 0
    max_box = None

    for y in range(binary.shape[0]):
        for x in range(binary.shape[1]):
            if binary[y, x] == 255 and not visited[y, x]:
                stack = [(y, x)]
                region = []

                while stack:
                    cy, cx = stack.pop()
                    if not visited[cy, cx]:
                        visited[cy, cx] = True
                        region.append((cy, cx))

                        for dy, dx in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                            ny, nx = cy + dy, cx + dx
                            if (0 <= ny < binary.shape[0] and 
                                0 <= nx < binary.shape[1] and 
                                binary[ny, nx] == 255 and 
                                not visited[ny, nx]):
                                stack.append((ny, nx))
                if len(region) > max_area:
                    max_area = len(region)
                    points = np.array(region)
                    rect = cv2.minAreaRect(points)
                    max_box = cv2.boxPoints(rect).astype(int)

    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(result, [max_box], 0, (0, 0, 255), 2)
    return result

result = find_max_connected_region(r'C:\Users\34946\Desktop\perception-interview\111.png')
cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()