import numpy as np
from skimage.measure import compare_psnr, compare_ssim
import scipy.io as sio
import scipy.signal as sg
from sewar.full_ref import uqi,sam
from sewar.no_ref import qnr


def quality_reference_accessment(out:dict,reference, target, ratio):
    '''
    有参照融合质量评价
    :param references:参照图像
    :param target: 融合图像
    :param ratio:边界大小
    :return:
    '''
    # rows, cols, bands = reference.shape
    # 去除边界
    # removed_reference = reference[ratio:rows - ratio, ratio:cols - ratio, :]
    # removed_target = target[ratio:rows - ratio, ratio:cols - ratio, :]
    out['cc'] = CC(reference, target)
    out['sam'] = SAM(reference, target)[0]
    out['rmse'] = RMSE(reference, target)
    out['egras'] = ERGAS(reference, target, ratio)
    out['psnr'] = PSNR(reference, target)
    out['ssim'] = SSIM(reference, target)
    out['uiqi'] = UIQI(reference, target)
    return out

def quality_no_reference_accessment(out:dict, lrhs, hrms, target, ratio):
    '''
    无参照融合质量评价
    :param references:参照图像
    :param target: 融合图像
    :param ratio:边界大小
    :return:
    '''
    # rows, cols, bands = reference.shape
    # 去除边界
    # removed_reference = reference[ratio:rows - ratio, ratio:cols - ratio, :]
    # removed_target = target[ratio:rows - ratio, ratio:cols - ratio, :]
    out['qnr'] = QNR(hrms,lrhs,target,ratio)
    return out

def QNR(hrms,lrhs,target,ratio):
    '''
    没有参照图像的融合质量评价
    :param D_lambd: 光谱失真
    :param D_s: 空间失真
    :return:
    '''
    return qnr(hrms,lrhs,target,r=ratio)

def CC(reference, target):
    '''
    相关性评价(按通道求两者相关系数，再取均值，理想值为1)
    :param references: 参照图像
    :param target: 融合图像
    :return:
    '''
    bands = reference.shape[2]
    out = np.zeros([bands])
    for i in range(bands):
        ref_temp = reference[:, :, i].flatten(order='F')  # 展开成向量
        target_temp = target[:, :, i].flatten(order='F')  # 展开成向量
        cc = np.corrcoef(ref_temp, target_temp)  # 求取相关系数矩阵
        out[i] = cc[0, 1]
    return np.mean(out)


def dot(m1, m2):
    '''
    两个三维图像求相同位置不同通道构成的向量内积
    :param m1: 图像1
    :param m2: 图像2
    :return:
    '''
    r, c, b = m1.shape
    p = r * c
    temp_m1 = np.reshape(m1, [p, b], order='F')
    temp_m2 = np.reshape(m2, [p, b], order='F')
    out = np.zeros([p])
    for i in range(p):
        out[i] = np.inner(temp_m1[i, :], temp_m2[i, :])
    out = np.reshape(out, [r, c], order='F')
    return out


def SAM(reference, target):
    '''
    光谱角度映射器评价（求取平均光谱映射角度，理想值为0）
    :param reference: 参照图像
    :param target: 融合图像
    :return:
    '''
    rows, cols, bands = reference.shape
    pixels = rows * cols
    eps = 2.2204e-16  # 浮点精度
    prod_scal = dot(reference, target)  # 取各通道相同位置组成的向量进行内积运算
    norm_ref = dot(reference, reference)
    norm_tar = dot(target, target)
    prod_norm = np.sqrt(norm_ref * norm_tar)  # 二范数乘积矩阵
    prod_map = prod_norm
    prod_map[prod_map == 0] = eps  # 除法避免除数为0

    cos_value = prod_scal / prod_map
    cos_value[cos_value > 1] = 1
    cos_value[cos_value < -1] = -1  ## 余弦相似度值域为[-1,1]

    map = np.arccos(cos_value)  # 求得映射矩阵,光谱角弧度为[0,pi]
    # 求取平均光谱角度
    angolo = np.mean(map)
    # 转换为度数
    angle_sam = np.real(angolo) * 180 / np.pi
    return angle_sam, map

def SSIM_BAND(reference, target):
    return compare_ssim(reference,target,data_range=1.0)

def UIQI_BAND(reference, target, block_size = 32):
    '''
    单通道图像UIQI 评价
    :param reference:
    :param target:
    :param block_size:
    :return:
    '''
    N = block_size ** 2

    blocks = np.ones([block_size, block_size], dtype=np.float32)

    img1_sq = reference * reference
    img2_sq = target * target
    img12 = reference * target

    img1_sum = sg.correlate2d(reference, blocks, mode='valid')
    img2_sum = sg.correlate2d(target, blocks, mode='valid')
    img1_sq_sum = sg.correlate2d(img1_sq, blocks, mode='valid')
    img2_sq_sum = sg.correlate2d(img2_sq, blocks, mode='valid')
    img12_sum = sg.correlate2d(img12, blocks, mode='valid')


    img12_sum_mul = img1_sum * img2_sum
    img12_sq_sum_mul = img1_sum * img1_sum + img2_sum * img2_sum
    numerator = 4 * (N * img12_sum - img12_sum_mul) * img12_sum_mul
    denominator1 = N * (img1_sq_sum + img2_sq_sum) - img12_sq_sum_mul
    denominator = denominator1 * img12_sq_sum_mul

    quality_map = np.ones_like(denominator)

    index1 = np.where(denominator1 == 0, 1, 0)
    index2 = np.where(img12_sq_sum_mul != 0, 1, 0)

    index = np.logical_and(index1, index2)

    quality_map[index] = 2 * img12_sum_mul[index] / img12_sq_sum_mul[index]

    index = np.where(denominator != 0, True, False)

    quality_map[index] = numerator[index] / denominator[index]

    return np.mean(quality_map)

def UIQI(reference, target):
    '''
    平均通用图像质量指标（UIQI）
    :param reference:
    :param target:
    :return:
    '''
    # rows, cols, bands = reference.shape
    # muiqi = 0
    # for i in range(bands):
    #     muiqi += UIQI_BAND(reference[:, :, i], target[:, :, i])
    # muiqi /= bands
    # return muiqi
    return uqi(reference,target)

def SSIM(reference, target):
    '''
    平均结构相似性
    :param reference:
    :param target:
    :return:
    '''
    rows,cols,bands = reference.shape
    mssim = 0
    for i in range(bands):
        mssim += SSIM_BAND(reference[:,:,i],target[:,:,i])
    mssim /= bands
    return mssim
    # return compare_ssim(reference, target, multichannel=True)


def PSNR(reference, target):
    '''
    峰值信噪比
    :param reference: 参照图像
    :param target: 融合图像
    :return:
    '''
    max_pixel = 1.0
    return 10.0 * np.log10((max_pixel ** 2) / np.mean(np.square(reference - target)))
    # return compare_psnr(reference, target)


def RMSE(reference, target):
    '''
    根均方误差评价（两图像各位置像素值差的F范数除以总像素个数的平方根，理想值为0）
    :param reference: 参照图像
    :param target: 融合图像
    :return:
    '''
    rows, cols, bands = reference.shape
    pixels = rows * cols * bands
    out = np.sqrt(np.sum((reference - target) ** 2) / pixels)
    return out


def ERGAS(references, target, ratio):
    '''
    总体相对误差评价（各通道求取相对均方误差取根均值，再乘以相应系数，理想值为0）
    :param references: 参照图像
    :param target: 融合图像
    :return:
    '''
    rows, cols, bands = references.shape
    d = 1 / ratio  # 全色图像与高光谱图像空间分辨率之比
    pixels = rows * cols
    ref_temp = np.reshape(references, [pixels, bands], order='F')
    tar_temp = np.reshape(target, [pixels, bands], order='F')
    err = ref_temp - tar_temp
    rmse2 = np.sum(err ** 2, axis=0) / pixels  # 均方误差
    uk = np.mean(tar_temp, axis=0)  # 各通道像素均值
    relative_rmse2 = rmse2 / uk ** 2  # 相对均方误差
    total_relative_rmse = np.sum(relative_rmse2)  # 求和
    out = 100 * d * np.sqrt(1 / bands * total_relative_rmse)  # 总体相对误差
    return out


if __name__ == '__main__':

    # target_path = r'e:/Fairy/CAVEMAT3/outputs/mhf-net/'
    # target_path = r'e:/Fairy/CAVEMAT3/outputs/CNN-Fus/'
    # target_path = r'e:/Fairy/CAVEMAT2/outputs/OTD2/'
    # target_path = r'e:/Fairy/CAVEMAT3/outputs/dhsis/'
    # target_path = r'e:/Fairy/CAVEMAT3/outputs/dbin/'
    # target_path = r'e:/Fairy/CAVEMAT3/outputs/ADMM-HFNetk8/'
    # target_path = r'e:/Fairy/CAVEMAT3/outputs/hysure/'
    # target_path = r'e:/Fairy/CAVEMAT3/outputs/darn/'
    # target_path = r'e:/Fairy/CAVEMAT3/outputs/ADMM-HFNet_noR/'
    # target_path =r'e:/Fairy/models/Unsupervised_HARVARD/outputs/LTTR/'
    # # target_path = 'e:/Fairy/WV2MAT/outputs/CNN-Fus/'
    # # target_path = r'd:/Fairy/HARVARDMAT/outputs/HARVARDMAT_ADMMk8/'
    # # target_path = r'f:/Fairy/PUMAT/outputs/ADMM-HFNet/'
    target_path = r'f:\target_detection\\'
    #
    # # target_path = r'e:/Fairy/HARVARDMAT/outputs/mhf-net/'
    # # target_path = r'e:/Fairy/HARVARDMAT/outputs/dbin/'
    # # target_path = r'e:/Fairy/HARVARDMAT/outputs/dhsis/'
    # # target_path = r'e:/Fairy/HARVARDMAT/outputs/hysure/'
    # # target_path = r'e:/Fairy/HARVARDMAT/outputs/OTD/'
    # # target_path = r'e:/Fairy/HARVARDMAT/outputs/CNN-Fus/'
    # # target_path = r'e:/Fairy/HARVARDMAT/outputs/dual_attenation/'
    # # target_path = r'e:/Fairy/outputs/cave_dbin/'
    # # target_path = r'e:/Fairy/CAVEMAT3/outputs/dual_attenation/'
    # # target_path = r'd:/Fairy/HARVARDMAT/outputs/HARVARDMAT_ADMM/'
    # # target_path = r'e:/Fairy/CAVEMAT/outputs/ADMM_noR/'
    reference_path = r'e:/Fairy/CAVEMAT3/'
    # # reference_path = r'd:/Fairy/HARVARDMAT/'
    # # reference_path = r'd:/Fairy/WV2MAT/'
    # # reference_path = r'f:/Fairy/PUMAT/'
    num_start = 8
    num_end = 8
    ratio = 8
    out = {}
    average_out = {'cc':0,'sam':0,'psnr':0,'rmse':0,'egras':0,'ssim':0,'uiqi':0}
    # average_out = {'qnr': 0}
    for i in range(num_start,num_end + 1):
        mat = sio.loadmat(reference_path+'%d.mat'%i)
        reference = mat['label']
        # lrhs = mat['Y']
        # hrms = mat['Z']
        # hrms = np.mean(hrms,axis=-1)
        # hrms = hrms.T
        target = sio.loadmat(target_path + '%d.mat'%i)['b']
        target = np.squeeze(target)
        # target = mat['UP']
        target = np.float32(target)
        target[target<0] = 0.0
        target[target>1] = 1.0
        quality_reference_accessment(out,reference,target,ratio)
        # quality_no_reference_accessment(out, lrhs, hrms, target,ratio=ratio)
        for key in out.keys():
            average_out[key] += out[key]
        print('image %d has finished'%i)
    for key in average_out.keys():
        average_out[key] /= (num_end-num_start+1)
    print(average_out)


    # list = ['HySure','OTD','LTTR','CNN-Fus','DBIN']
    # # list = ['ICCV15']
    # # list = ['MSDCNN','3DCNN','RSIFNN','DHSIS','PNN','DSPHF']
    #
    # # original = r'f:/Fairy/outputs/'
    # original = r'F:/Fairy/PUMAT5/outputs/'
    #
    # reference_path = r'F:/Fairy/PUMAT5/'
    # num_start = 1
    # num_end = 1
    # ratio = 4
    # out = {}
    #
    # for method in list:
    #     target_path = original + method +'/'
    #     average_out = {'cc': 0, 'sam': 0, 'psnr': 0, 'rmse': 0, 'egras': 0, 'ssim': 0, 'uiqi': 0}
    #     # average_out = {'qnr': 0}
    #     for i in range(num_start, num_end + 1):
    #         mat = sio.loadmat(reference_path + '%d.mat' % i)
    #         reference = mat['label']
    #         # lrhs = mat['Y']
    #         # hrms = mat['Z']
    #         # hrms = np.mean(hrms,axis=-1)
    #         # hrms = hrms.T
    #         target = sio.loadmat(target_path + '%d.mat' % i)['F']
    #         target = np.squeeze(target)
    #         # target = mat['UP']
    #         target = np.float32(target)
    #         target[target < 0] = 0.0
    #         target[target > 1] = 1.0
    #         quality_reference_accessment(out, reference, target, ratio)
    #         # quality_no_reference_accessment(out, lrhs, hrms, target,ratio=ratio)
    #         for key in out.keys():
    #             average_out[key] += out[key]
    #         print('image %d has finished' % i)
    #     for key in average_out.keys():
    #         average_out[key] /= (num_end - num_start + 1)
    #     print(average_out)
    #
    #     print('%s has finished'%method)





