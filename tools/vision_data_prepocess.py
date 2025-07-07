#可视化数据预处理过程


import numpy as np
import pandas as pd
from tqdm import tqdm
from astropy.io import fits
from scipy.signal import medfilt
from scipy.interpolate import interp1d


param = {'poly_global_order': 5,
         'nor': 1,
         'poly_lowerlimit': 3,
         'poly_upperlimit': 4,
         'median_radius': 3,
         'poly_SM': 0,
         'poly_del_filled': 2
         }


# read fits
def get_flux_data(wavelength, flux,rv_ku0, wavelength_scope,RV_correct=True, ver="dr7"):
    c = 299792.458
    wavelength_fixed = np.arange(wavelength_scope[0], wavelength_scope[1], 1)
    if RV_correct:
        # wave0=wave/(1+RV/c)
        wavelength = wavelength / (1 + rv_ku0 / c)


        # interpolation
        f = interp1d(wavelength, flux, kind='linear')
        flux_data = f(wavelength_fixed)


    return wavelength_fixed,flux_data.astype(np.float32)


def csp_polyfit(sp, angs, param):
    # standardize flux
    sp_c = np.mean(sp)
    sp = sp - sp_c
    sp_s = np.std(sp)
    sp = sp / sp_s

    # standardize wavelength
    angs_c = np.mean(angs)
    angs = angs - angs_c
    angs_s = np.std(angs)
    angs = angs / angs_s

    param['poly_sp_c'] = sp_c
    param['poly_sp_s'] = sp_s
    param['poly_angs_c'] = angs_c
    param['poly_angs_s'] = angs_s

    data_flag = np.full(sp.shape, 1)

    i = 0
    con = True
    while (con):
        P_g = np.polyfit(angs, sp, param['poly_global_order'])
        param['poly_P_g'] = P_g
        fitval_1 = np.polyval(P_g, angs)
        dev = fitval_1 - sp
        sig_g = np.std(dev)

        data_flag_new = (dev > (-param['poly_upperlimit'] * sig_g)) * (dev < (param['poly_lowerlimit'] * sig_g))

        if sum(abs(data_flag_new - data_flag)) > 0:
            if param['poly_del_filled'] == 1:
                data_flag = data_flag_new
            else:
                fill_flag = data_flag - data_flag_new
                index_1 = np.where(fill_flag != 0)
                sp[index_1] = fitval_1[index_1]
        else:
            con = False
        i += 1

    index_2 = np.where(data_flag != 0)
    param['poly_sp_filtered'] = sp[index_2]
    param['poly_angs_filtered'] = angs[index_2]

    return param


def sp_median_polyfit1stage(flux, lambda_log, param):
    flux1 = flux
    lambda1 = lambda_log

    flux_median1 = medfilt(flux1, param['median_radius'])

    dev1 = flux_median1 - flux1
    sigma = np.std(dev1)
    data_flag1 = (dev1 < (param['poly_lowerlimit'] * sigma)) * (dev1 > (-param['poly_upperlimit'] * sigma))

    fill_flag1 = 1 - data_flag1

    if param['poly_del_filled'] == 1:
        index_1 = np.where(data_flag1)
        flux1 = flux1[index_1]
        lambda1 = lambda1[index_1]
    elif param['poly_del_filled'] == 2:
        index_2 = np.where(fill_flag1)
        flux1[index_2] = flux_median1[index_2]

    param = csp_polyfit(flux1, lambda1, param)

    angs = lambda1 - param['poly_angs_c']
    angs = angs / param['poly_angs_s']

    fitval_g = np.polyval(param['poly_P_g'], angs)
    continum_fitted = fitval_g * param['poly_sp_s'] + param['poly_sp_c']
    if param['poly_SM'] == 1:
        angss = lambda1
    else:
        angss = 10 ** lambda1
    return continum_fitted


def flux_transform(wavelength_fixed, flux):
    # 创建保存目录（与wave_process函数保持一致）
    save_dir = r'D:\Notebook_workdir\thesis\fig\data_preprocessing'
    os.makedirs(save_dir, exist_ok=True)
    
    # 拟合连续谱
    continum_fitted = sp_median_polyfit1stage(flux[:len(wavelength_fixed)], np.log10(wavelength_fixed), param)
    
    # 创建图形
    plt.figure(figsize=(10, 6))
    
    # 绘制原始曲线（灰色半透明，突出拟合曲线）
    plt.plot(wavelength_fixed, flux, color='green', alpha=0.5, label='Processed Data after interp1d', zorder=1)
    
    # 绘制拟合曲线（红色，加粗）
    plt.plot(wavelength_fixed, continum_fitted, color='crimson', linewidth=2, label='Fitted Continuum', zorder=2)
    
    # 图形设置
    plt.xlabel('Wavelength')
    plt.ylabel('Flux')
    plt.title('Flux Curve vs Fitted Continuum')
    plt.legend()
    # plt.grid(True, linestyle='--', alpha=0.7, zorder=0)  # 添加网格线在底层
    
    # 保存图片
    save_path = os.path.join(save_dir, 'flux_fitting_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 高分辨率保存
    plt.close()  # 关闭图形释放内存
    
    # 数据处理
    flux_data_fitted = flux / continum_fitted
    return wavelength_fixed, flux_data_fitted.astype(np.float32)



import matplotlib.pyplot as plt
import os

def wave_process(wavelength, flux, rv_ku0, wavelength_scope):
    # 创建保存图片的目录（如果不存在）
    save_dir = r'D:\Notebook_workdir\thesis\fig\data_preprocessing'
    os.makedirs(save_dir, exist_ok=True)
    
    # 第一步：原始数据绘图
    plt.figure(figsize=(10, 6))
    plt.plot(wavelength, flux, color='blue', label='Original Data')
    plt.xlabel('Wavelength')
    plt.ylabel('Flux')
    plt.title('Original Wavelength and Flux')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'original_data.png'))
    plt.close()
    
    # 第一步处理：获取flux数据
    wavelength_fixed, flux_data = get_flux_data(wavelength, flux, rv_ku0, wavelength_scope)
    
    # 第二步：处理后数据绘图
    plt.figure(figsize=(10, 6))
    plt.plot(wavelength_fixed, flux_data, color='green', label='Processed Data after interp1d')
    plt.xlabel('Wavelength (Fixed)')
    plt.ylabel('Flux')
    plt.title('Data after get_flux_data Processing')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'step1_processed_data.png'))
    plt.close()
    
    # 第二步处理：flux变换
    wavelength_fixed, flux_data_fitted = flux_transform(wavelength_fixed, flux_data)
    
    # 第三步：拟合后数据绘图
    plt.figure(figsize=(10, 6))
    plt.plot(wavelength_fixed, flux_data_fitted, color='red', label='Final Data')
    plt.xlabel('Wavelength (Fixed)')
    plt.ylabel('Flux (Fitted)')
    plt.title('final Data')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'step2_fitted_data.png'))
    plt.close()
    
    return wavelength_fixed, flux_data_fitted


def get_EGP(wavelength, flux, numerator_range=(4200, 4400), denominator_range=(4425, 4520)):
    # 创建插值函数
    f = interp1d(wavelength, flux)

    # 确保积分范围在有效范围内
    numerator_range = (max(numerator_range[0], min(wavelength)), min(numerator_range[1], max(wavelength)))
    denominator_range = (max(denominator_range[0], min(wavelength)), min(denominator_range[1], max(wavelength)))

    # 计算积分值
    wavelength1 = np.arange(numerator_range[0], numerator_range[1])
    wavelength2 = np.arange(denominator_range[0], denominator_range[1])
    result1 = np.trapz(f(wavelength1), wavelength1)
    result2 = np.trapz(f(wavelength2), wavelength2)

    # 计算EGP值
    EGP = -2.5 * np.log10(result1 / result2)
    
    return EGP


data_dir="./data/spectra/"

csv=pd.read_csv(r'D:\Notebook_workdir\thesis\data\train\y_train.csv')
p=csv.iloc[0]
fname = p['combined_file']
fname = fname.split('/', 2)[-1]

wave_range=(3900, 8800)
filename=data_dir + fname
hdu = fits.open(filename)
origin_wavelength=hdu[1].data["WAVELENGTH"][0]
origin_flux=hdu[1].data["FLUX"][0]
z=p['combined_z']
print(123)
# range_index=(origin_wavelength>wave_range[0])&(origin_wavelength<wave_range[1])
# EGP_value=get_EGP(origin_wavelength,origin_flux)
wavelength,flux=wave_process(origin_wavelength,origin_flux,z,wave_range)

print(flux)