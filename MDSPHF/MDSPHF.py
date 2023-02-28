import numpy as np
import scipy.io as sio
import scipy
import cv2
from utils.quality_measure import quality_reference_accessment

def mdsphf(Xh, Y, Z, B, R, k=31, ratio=8, lamb=1e-6, mu=1e-6):
    '''
    MDSPHF method
    :param X:
    :param Y:
    :param Z:
    :param B:
    :param R:
    :param k:
    :param ratio:
    :param lamb:
    :param mu:
    :return:
    '''
    h, w, c = Xh.shape
    Xh = np.reshape(Xh, [h * w, -1], order='F').T
    Z = np.reshape(Z, [h * w, -1], order='F').T

    Y = np.reshape(Y, [(h // ratio) * (w // ratio), -1], order='F').T
    _, _, V = scipy.linalg.svd(Y.T, full_matrices=False)
    P = V.T[:, :k]
    # _, _, V = scipy.linalg.svd(X, full_matrices=False)
    # P = V[:, :k]
    RP = np.dot(R, P)
    H1 = np.dot(RP.T, RP) + lamb * np.dot(P.T, P)
    H2 = np.dot(RP.T, Z) + lamb * np.dot(P.T, Xh)
    A = np.linalg.solve(H1, H2)

    A_tensor = np.reshape(A.T, [h, w, -1], order='F')
    B = np.expand_dims(B, axis=-1)
    ABD = np.real(np.fft.ifftn(np.fft.fftn(A_tensor) * B))
    ABD = ABD[::ratio, ::ratio, :]
    ABD = np.reshape(ABD, [(h // ratio) * (w // ratio), k], order='F')
    ABD = ABD.T
    # ABD = blur_downsample(A, ratio, B, h, w)

    H3 = np.dot(ABD, ABD.T) + mu * np.dot(A, A.T)
    H4 = np.dot(Y, ABD.T) + mu * np.dot(Xh, A.T)
    P = np.linalg.solve(H3.T, H4.T)
    P = P.T

    target = np.dot(P, A)
    target = np.reshape(target.T, [h, w, -1], order='F')
    target = np.float32(target)

    return target


if __name__ == '__main__':

    arr = ['CAVE']
    for num in arr:
        if num == 'CAVE':
            data_path = 'CAVEMAT/'
            test_start = 21
            test_end = 32
            h = 512
            w = 512
            k = 11
            ratio = 8

        elif num == 'Harvard':
            data_path = 'D:/Fairy/HARVARDMAT/'
            test_start = 31
            test_end = 50
            h = 1040
            w = 1392
            k = 6
            ratio = 8
        elif num == 'PU':
            data_path = 'F:/Fairy/PUMAT/'
            test_start = 3
            test_end = 3
            h = 128
            w = 128
            k = 4
            ratio = 8
        else:
            print('invalid dataset')
            break
        mat = sio.loadmat('saved_B_R/%s/R_B.mat' % num)
        B = mat['B']
        R = mat['R']
        # R = getSpectralResponse()
        # B = get_kernal(8,2,512,512)

        out = {}
        average_out = {'cc': 0, 'psnr': 0, 'sam': 0, 'ssim': 0, 'rmse': 0, 'egras': 0, 'uiqi': 0}

        for i in range(test_start, test_end + 1):
            mat = sio.loadmat(data_path + '%d.mat' % i)
            Y = mat['Y']
            Z = mat['Z']
            X = mat['label']
            Xh = cv2.resize(Y, (w, h), interpolation=cv2.INTER_CUBIC)

            F = mdsphf(Xh, Y, Z, B, R, k=k, ratio=ratio)
            F = np.clip(F, 0, 1)

            quality_reference_accessment(out, X, F, ratio)
            for key in out.keys():
                average_out[key] += out[key]
            print('%d has finished' % i)

        for key in average_out.keys():
            average_out[key] /= (test_end - test_start + 1)
        print(average_out)
