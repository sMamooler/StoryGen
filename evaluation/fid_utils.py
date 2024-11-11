import numpy as np
from scipy import linalg

import torch
import torch.nn.functional as F
from torchvision import transforms

from model.inception import InceptionV3


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print(
            "fid calculation produces singular product; adding %s to diagonal of cov estimates"
            % eps
        )
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)


def calculate_fid_given_features(feature1, feature2):
    mu1 = np.mean(feature1, axis=0)
    sigma1 = np.cov(feature1, rowvar=False)
    mu2 = np.mean(feature2, axis=0)
    sigma2 = np.cov(feature2, rowvar=False)
    fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

    return fid_value


def inception_feature(images, device):

    fid_augment = transforms.Compose(
        [
            # transforms.ToPILImage(mode="RGB"),
            transforms.Resize([28, 64]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block_idx])

    images = torch.stack([fid_augment(image) for image in images])
    images = images.type(torch.FloatTensor)  # .to(device)
    images = (images + 1) / 2
    images = F.interpolate(
        images, size=(299, 299), mode="bilinear", align_corners=False
    )
    pred = inception(images)[0]

    if pred.shape[2] != 1 or pred.shape[3] != 1:
        pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
    return pred.reshape(-1, 2048)
