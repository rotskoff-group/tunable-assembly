import cupy as cp
import cupyx

# This file contains a modification of the code from
# https://stackoverflow.com/questions/34222272/computing-mean-square-displacement-using-python-and-fft

# Convention is thing being advanced in time is x, stationary quantity is y


def corrFFTAll_gpu(x, y, N_len, N_batch):
    Fx = cp.zeros((2 * N_len, N_batch), dtype=cp.complex64)
    Fy = cp.zeros((2 * N_len, N_batch), dtype=cp.complex64)
    PSD = cp.zeros((2 * N_len, N_batch), dtype=cp.complex64)
    res_xx = cp.zeros((2 * N_len, N_batch), dtype=cp.complex64)
    res_xx_real = cp.zeros((N_len, N_batch), dtype=cp.float32)
    res_yy = cp.zeros((2 * N_len, N_batch), dtype=cp.complex64)
    res_yy_real = cp.zeros((N_len, N_batch), dtype=cp.float32)
    res_xy = cp.zeros((2 * N_len, N_batch), dtype=cp.complex64)
    res_xy_real = cp.zeros((N_len, N_batch), dtype=cp.float32)
    res_yx = cp.zeros((2 * N_len, N_batch), dtype=cp.complex64)
    res_yx_real = cp.zeros((N_len, N_batch), dtype=cp.float32)
    Fx = cupyx.scipy.fft.fft(x, n=2 * N_len, axis=0)
    Fy = cupyx.scipy.fft.fft(y, n=2 * N_len, axis=0)
    Fx_conj = Fx.conjugate()
    Fy_conj = Fy.conjugate()
    PSD = Fx * Fx_conj
    res_xx = cupyx.scipy.fft.ifft(PSD, axis=0)
    PSD = Fy * Fy_conj
    res_yy = cupyx.scipy.fft.ifft(PSD, axis=0)
    PSD = Fy * Fx_conj
    res_xy = cupyx.scipy.fft.ifft(PSD, axis=0)
    PSD = Fx * Fy_conj
    res_yx = cupyx.scipy.fft.ifft(PSD, axis=0)
    res_xx_real = (res_xx[:N_len]).real
    res_yy_real = (res_yy[:N_len]).real
    res_xy_real = (res_xy[:N_len]).real
    res_yx_real = (res_yx[:N_len]).real
    n = N_len * cp.ones(N_len) - cp.arange(0, N_len)  # divide res(m) by (N-m)
    n = n.astype(cp.float32)
    res_xx_real = res_xx_real / n[:, cp.newaxis]
    res_yy_real = res_yy_real / n[:, cp.newaxis]
    res_xy_real = res_xy_real / n[:, cp.newaxis]
    res_yx_real = res_yx_real / n[:, cp.newaxis]
    return res_xx_real, res_yy_real, res_xy_real, res_yx_real


def vacf_parallel_gpu(velocities, cycle, batch_size=16):
    vacf_xx_mean = cp.zeros((velocities.shape[0]), dtype=cp.float32)
    vacf_yy_mean = cp.zeros((velocities.shape[0]), dtype=cp.float32)
    vacf_xy_mean = cp.zeros((velocities.shape[0]), dtype=cp.float32)
    vacf_yx_mean = cp.zeros((velocities.shape[0]), dtype=cp.float32)
    # Generate batches from velocities
    for i in range(0, velocities.shape[1], batch_size):
        if i + batch_size < velocities.shape[1]:
            velocities_gpu = cp.asarray(velocities[:, i:i + batch_size, :])
        else:
            velocities_gpu = cp.asarray(velocities[:, i:, :])
        vacf_xx_, vacf_yy_, vacf_xy_, vacf_yx_ = corrFFTAll_gpu(
            velocities_gpu[:, :, 0],
            velocities_gpu[:, :, 1],
            velocities_gpu.shape[0],
            velocities_gpu.shape[1],
        )
        vacf_xx_mean += cp.sum(vacf_xx_, axis=1)
        vacf_yy_mean += cp.sum(vacf_yy_, axis=1)
        vacf_xy_mean += cp.sum(vacf_xy_, axis=1)
        vacf_yx_mean += cp.sum(vacf_yx_, axis=1)
    vacf_xx_mean /= velocities.shape[1]
    vacf_yy_mean /= velocities.shape[1]
    vacf_xy_mean /= velocities.shape[1]
    vacf_yx_mean /= velocities.shape[1]
    # Save to file
    cp.savez_compressed(
        f"vacf_gpu_{cycle}.npz",
        vacf_xx_mean=vacf_xx_mean,
        vacf_yy_mean=vacf_yy_mean,
        vacf_xy_mean=vacf_xy_mean,
        vacf_yx_mean=vacf_yx_mean,
    )
