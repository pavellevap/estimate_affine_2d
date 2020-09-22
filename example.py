import numpy as np
from affine_estimation import estimate_affine, estimate_affine_ransac


noise_power = 0.05
outlier_fraction = 0.8
points_count = 1000
outliers_count = int(points_count * outlier_fraction)
inliers_count = points_count - outliers_count
labels_count = 750

angle = np.random.uniform(-np.pi, np.pi)
scale_x, scale_y = np.random.lognormal(0, 1, 2)
trans_x, trans_y = np.random.normal(0, 5, 2)
model_true = np.array([[np.cos(angle) * scale_x,  -np.sin(angle) * scale_y, trans_x],
                       [np.sin(angle) * scale_x,   np.cos(angle) * scale_y, trans_y]]).astype(np.float32)

outliers_src = np.random.uniform(-1, 1, size=(outliers_count, 2)).astype(np.float32)
points_src = np.random.uniform(-1, 1, size=(points_count, 2)).astype(np.float32)
noise = np.random.normal(0, noise_power, size=(points_count, 2)).astype(np.float32)
points_corrupted_src = np.concatenate([points_src[:inliers_count], outliers_src])
points_corrupted_src = np.concatenate([points_corrupted_src.T, np.ones((1, points_count)).astype(np.float32)])
points_dst = np.matmul(model_true, points_corrupted_src).T + noise

outlier_labels = np.random.randint(0, labels_count, outliers_count).astype(np.int32)
labels_src = np.random.randint(0, labels_count, points_count).astype(np.int32)
labels_dst = np.concatenate([labels_src[:inliers_count], outlier_labels])

matches = [[], []]
for i in range(len(labels_src)):
    label = labels_src[i]
    p1 = points_src[i]
    for p2 in points_dst[labels_dst == label]:
        matches[0].append(p1)
        matches[1].append(p2)
matches = (np.array(matches[0]), np.array(matches[1]))

def reprojection_error(model_pred):
    inliers_src = points_src[:inliers_count]
    inliers_dst = points_dst[:inliers_count]
    inliers_src = np.concatenate([inliers_src.T, np.ones((1, inliers_count)).astype(np.float32)])
    return np.sqrt(np.sum(np.square(np.matmul(model_pred, inliers_src).T - inliers_dst)))

print(f'True model:\n{model_true}')

model_pred = estimate_affine(matches[0], matches[1])
print(f'Least squares reprojection error: {reprojection_error(model_pred)}')
print(f'Least squares model:\n{model_pred}')

model_pred, inliers = estimate_affine_ransac(points_src, labels_src, points_dst, labels_dst, max_iter=10000, 
                                    stop_prob=0.9999, stop_match=10000, threshold=0.15)
print(f'Ransac reprojection error: {reprojection_error(model_pred)}')

print(f'Ransac model:\n{model_pred}')
print(f'Inliers count: {len(inliers)}')

