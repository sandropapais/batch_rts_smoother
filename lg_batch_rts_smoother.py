from os.path import dirname, join
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from statsmodels.graphics.gofplots import qqplot


def read_data_mat():
    # Load .mat file
    data_dir = join(dirname(__file__), 'data')
    mat_file_name = join(data_dir, 'dataset1.mat')
    mat_contents = sio.loadmat(mat_file_name)
    return mat_contents


def compute_measurement_stats(mat_contents):
    # Unpack data and flatten to 1D
    range_meas = mat_contents['r'].flatten()
    pos_true = mat_contents['x_true'].flatten()
    pos_true = pos_true.flatten()
    t = mat_contents['t'].flatten()
    vel_meas = mat_contents['v'].flatten()
    pos_cyl = mat_contents['l'].flatten()
    # range_meas_var = mat_contents['r_var'].flatten()  # unused
    # vel_meas_var = mat_contents['v_var'].flatten()  # unused
    samples_count = pos_true.size
    t_step = t[1]

    # Measurement model and motion model noise
    pos_meas = pos_cyl - range_meas
    pos_meas_err = pos_meas - pos_true  # measurement model noise
    pos_prop_err = np.zeros(samples_count)  # motion model noise
    for i in range(1, samples_count):
        pos_prop_err[i] = pos_true[i] - pos_true[0] - t_step * np.sum(vel_meas[:i + 1]) - np.sum(pos_prop_err[:i])

    # Fit data and compute statistics
    pos_true_mean = np.mean(pos_true)
    meas_err_mean, meas_err_std = norm.fit(pos_meas_err)
    prop_err_mean, prop_err_std = norm.fit(pos_prop_err)
    q_proc_noise_cov = prop_err_std ** 2
    r_meas_noise_cov = meas_err_std ** 2
    print('*** PREPROCESS ***\n'
          f'Position mean = {pos_true_mean:1.3f} m\n'
          f'Measurement noise: mean={meas_err_mean:1.3e} m, std={meas_err_std:1.3f} m,'
          f' 3-sigma={3 * meas_err_std:1.3f}m\n'
          f'Propagation noise: mean={prop_err_mean:1.3e} m, std={prop_err_std:1.3f} m, '
          f'3-sigma={3 * prop_err_std:1.3f}m')

    # PLOT01: Measurements
    fig1, axs1 = plt.subplots(2, 1)
    axs1[0].plot(t, range_meas)
    axs1[0].set_ylabel('Laser Rangefinder (m)')
    axs1[0].set_title("Measurements")
    axs1[1].plot(t, vel_meas)
    axs1[1].set_xlabel('Time (s)')
    axs1[1].set_ylabel('Odometry (m/s)')
    fig1.savefig('out/meas_vs_t.png')

    # PLOT02: Model Noise
    fig2, axs2 = plt.subplots(2, 1)
    axs2[0].plot(t, pos_meas_err)
    axs2[0].set_ylabel('Measurement\nModel Noise (m)')
    axs2[0].set_title("Model Noise")
    axs2[1].plot(t, pos_prop_err)
    axs2[1].set_xlabel('Time (s)')
    axs2[1].set_ylabel('Motion\nModel Noise (m)')
    fig2.tight_layout(pad=0.2)
    fig2.savefig('out/model_noise_vs_t.png')

    # PLOT03: Model Noise Histogram
    fig3, axs3 = plt.subplots(2, 1)
    n_pts_plot = 100
    bins = axs3[0].hist(pos_meas_err, 20, density=1)[1]
    meas_err_fit_bins = np.linspace(bins[0], bins[-1], n_pts_plot)
    meas_err_fit = norm.pdf(meas_err_fit_bins, meas_err_mean, meas_err_std)
    axs3[0].plot(meas_err_fit_bins, meas_err_fit)
    axs3[0].set_xlabel('Measurement Model Noise (m)')
    axs3[0].text(0.95, 0.95, fr'$\mu={meas_err_mean:1.3e}, \sigma={meas_err_std:1.3f}$',
                 horizontalalignment='right', verticalalignment='top', transform=axs3[0].transAxes)
    axs3[0].set_title("Model Noise Histogram")
    bins = axs3[1].hist(pos_prop_err, 20, density=1)[1]
    prop_err_fit_bins = np.linspace(bins[0], bins[-1], n_pts_plot)
    prop_err_fit = norm.pdf(prop_err_fit_bins, prop_err_mean, prop_err_std)
    axs3[1].plot(prop_err_fit_bins, prop_err_fit)
    axs3[1].set_xlabel('Motion Model Noise (m)')
    axs3[1].text(0.95, 0.95, fr'$\mu={prop_err_mean:1.3e}, \sigma={prop_err_std:1.3f}$',
                 horizontalalignment='right', verticalalignment='top', transform=axs3[1].transAxes)
    fig3.tight_layout(pad=0.2)
    fig3.savefig('out/hist_model_noise.png')

    # PLOT04: Model Noise Q-Q
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(2, 1, 1)
    qqplot(pos_meas_err, ax=ax4, line='s')
    ax4.set_ylabel('Measurement\nModel Noise')
    ax4.set_title("Model Noise Q-Q")
    ax4 = fig4.add_subplot(2, 1, 2)
    qqplot(pos_prop_err, ax=ax4, line='s')
    ax4.set_ylabel('Motion\nModel Noise')
    fig4.tight_layout(pad=0.2)
    fig4.savefig('out/qq_model_noise.png')

    return pos_meas, vel_meas, pos_true, t, samples_count, t_step, q_proc_noise_cov, r_meas_noise_cov


def rts_smoother(pos_meas, vel_meas, pos_true, t, samples_count, t_step, q_proc_noise_cov, r_meas_noise_cov,
                 flg_debug_prop_only, flg_debug_fwd_only, update_interval_indices, a_trans_mat, c_obs_mat):
    # RTS smoother initialization
    pos_est_ini = pos_meas[0]
    pos_var_ini = q_proc_noise_cov
    pos_est_pred_fwd = np.zeros(samples_count)
    pos_est_corr_fwd = np.zeros(samples_count)
    pos_est_corr = np.zeros(samples_count)
    pos_var_pred_fwd = np.zeros(samples_count)
    pos_var_corr_fwd = np.zeros(samples_count)
    pos_var_corr = np.zeros(samples_count)

    # RTS smoother forward pass
    for i in range(0, samples_count):
        if i == 0:
            pos_est_pred_fwd[0] = pos_est_ini
            pos_var_pred_fwd[0] = pos_var_ini
        else:
            pos_var_pred_fwd[i] = a_trans_mat * pos_var_corr_fwd[i - 1] * a_trans_mat + q_proc_noise_cov
            pos_est_pred_fwd[i] = a_trans_mat * pos_est_corr_fwd[i - 1] + t_step * vel_meas[i]
        if flg_debug_prop_only == 1:
            kalman_gain = 0
        elif (i % update_interval_indices) == 0:
            kalman_gain = pos_var_pred_fwd[i] * c_obs_mat / (
                    c_obs_mat * pos_var_pred_fwd[i] * c_obs_mat + r_meas_noise_cov)
        else:
            kalman_gain = 0
        pos_var_corr_fwd[i] = (1 - kalman_gain * c_obs_mat) * pos_var_pred_fwd[i]
        pos_est_corr_fwd[i] = pos_est_pred_fwd[i] + kalman_gain * (pos_meas[i] - c_obs_mat * pos_est_pred_fwd[i])

    # RTS smoother backward pass
    if flg_debug_fwd_only == 1:
        pos_var_corr = pos_var_corr_fwd
        pos_est_corr = pos_est_corr_fwd
    else:
        pos_est_corr[-1] = pos_est_corr_fwd[-1]
        pos_var_corr[-1] = pos_var_corr_fwd[-1]
        for i in range(samples_count - 1, 0, -1):
            pos_est_corr[i - 1] = \
                pos_est_corr_fwd[i - 1] + (pos_var_corr_fwd[i - 1] * a_trans_mat / pos_var_pred_fwd[i]) * \
                (pos_est_corr[i] - pos_est_pred_fwd[i])
            pos_var_corr[i - 1] = \
                pos_var_corr_fwd[i - 1] + (pos_var_corr_fwd[i - 1] * a_trans_mat / pos_var_pred_fwd[i]) * \
                (pos_var_corr[i] - pos_var_pred_fwd[i]) * (pos_var_corr_fwd[i - 1] * a_trans_mat / pos_var_pred_fwd[i])

    # Post process results
    pos_est_err = pos_est_corr - pos_true
    pos_est_err_mean = np.mean(pos_est_err)
    pos_est_err_mod_avg = np.mean(np.abs(pos_est_err))
    pos_est_rmse = np.sqrt(np.sum(pos_est_err ** 2) / samples_count)
    pos_est_err_std = np.std(pos_est_err)
    pos_3sigma = 3 * np.sqrt(pos_var_corr)
    print('*** POST-PROCESS ****\n'
          f'Process model variance = {q_proc_noise_cov:1.5f}\n'
          f'Measurement model variance = {r_meas_noise_cov:1.5f}\n'
          f'Average error magnitude = {pos_est_err_mod_avg:1.3f} m\n'
          f'Root mean square error = {pos_est_rmse:1.3f} m\n'
          f'3 sigma error = {3 * pos_est_err_std:1.3f} m')

    # PLOT05: Estimation Error and Uncertainty
    fig5, ax5 = plt.subplots()
    ax5.plot(t, pos_est_err, label=r'$\hat x_k - x_k$')
    ax5.plot(t, pos_3sigma, 'r--', label=r'$\pm3\hat\sigma_{x_k}$')
    ax5.plot(t, -pos_3sigma, 'r--')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Position (m)')
    ax5.set_title("Estimation Error and Uncertainty")
    ax5.legend()
    fig5.savefig(f'out/est_err_and_3std_{update_interval_indices}steps.png')

    # PLOT06: Estimation Error Histogram
    fig6, ax6 = plt.subplots()
    n_pts_plot = 1000
    bins = ax6.hist(pos_est_err, 20, density=1)[1]
    pos_est_err_fit_bins = np.linspace(bins[0], bins[-1], n_pts_plot)
    pos_est_err_fit = norm.pdf(pos_est_err_fit_bins, 0, pos_est_err_std)
    ax6.plot(pos_est_err_fit_bins, pos_est_err_fit)
    ax6.set_xlabel('Position Estimate Error (m)')
    ax6.set_title("Estimation Error Histogram")
    plt.text(0.95, 0.95, fr'$\mu={pos_est_err_mean:1.3e}, \sigma={pos_est_err_std:1.3f}$',
             horizontalalignment='right', verticalalignment='top', transform=ax6.transAxes)
    fig6.savefig(f'out/est_err_hist_{update_interval_indices}steps.png')

    # PLOT07: Estimation and Uncertainty
    fig7, ax7 = plt.subplots()
    ax7.plot(t, pos_est_corr, label=r'$\hat{x_k}$')
    ax7.plot(t, pos_true, label=r'$x_k$')
    ax7.plot(t, pos_est_corr + pos_3sigma, 'r--', label=r'$\hat{x_k}\pm3\hat\sigma_{x_k}$')
    ax7.plot(t, pos_est_corr - pos_3sigma, 'r--')
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Position (m)')
    ax7.set_title("Estimation and Uncertainty")
    ax7.legend()
    fig7.savefig(f'out/est_and_tru_{update_interval_indices}steps.png')


def main():
    # Define flags
    flg_plot_show = 0  # 0, 1
    flg_debug_prop_only = 0  # 0, 1
    flg_debug_fwd_only = 0  # 0, 1
    # Define RTS smoother parameters
    update_interval_indices = 1000  # 1, 10, 100, 1000
    a_trans_mat = 1
    c_obs_mat = 1
    # Load data file
    mat_contents = read_data_mat()
    # Q1: Preprocess data for statistics
    pos_meas, vel_meas, pos_true, t, samples_count, t_step, q_proc_noise_cov, r_meas_noise_cov = \
        compute_measurement_stats(mat_contents)
    # Q5: Call RTS smoother on data
    q_proc_noise_cov = 0.002  # instead of using prop_err_std ** 2, we inflate the process noise
    rts_smoother(pos_meas, vel_meas, pos_true, t, samples_count, t_step, q_proc_noise_cov, r_meas_noise_cov,
                 flg_debug_prop_only, flg_debug_fwd_only, update_interval_indices, a_trans_mat, c_obs_mat)
    # Plotting
    if flg_plot_show == 1:
        plt.show()


if __name__ == "__main__":
    main()
