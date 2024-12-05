import numpy as np
from scipy.signal import butter, lfilter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class Filters:

    @staticmethod
    def hampel_filter(data, window_size, n_sigmas):
        #esse n_sigma= 3 não vai dar BO n
        """
        Aplica o filtro de Hampel para remover outliers.
        - data: Numpy array 1D ou 2D.
        - window_size: Tamanho da janela ao redor de cada ponto.
        - n_sigmas: Multiplicador para identificar outliers baseado no desvio padrão.
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        filtered_data = data.copy()

        for col in range(data.shape[1]):
            for i in range(data.shape[0]):
                start = max(0, i - window_size)
                end = min(data.shape[0], i + window_size + 1)
                window = data[start:end, col]

                median = np.median(window)
                mad = np.median(np.abs(window - median))
                threshold = mad * n_sigmas
                if abs(data[i, col] - median) > threshold:
                    filtered_data[i, col] = median
        return filtered_data
    
    @staticmethod
    def hampel_filter_forloop_numba(input_series, window_size, n_subport, n_sigmas=3):
        new_series = input_series.copy()
        k = 1.4826  # scale factor for Gaussian distribution
        n = len(new_series[0, :])
        for subportadora in range(n_subport):
            amps_subportadora = new_series[subportadora, :].copy()

            for i in range(window_size, n - window_size + 1):
                x0 = np.nanmedian(amps_subportadora[i - window_size:i + window_size])
                S0 = k * np.nanmedian(np.abs(amps_subportadora[i - window_size:i + window_size] - x0))

                if i - window_size == 0:  # tanto os primeiros quantos os ultimos valores não estão sendo pegos
                    for j in range(window_size+1):
                        if np.abs(amps_subportadora[j] - x0) > n_sigmas * S0:
                            new_series[subportadora, j] = x0
                elif i + window_size == n - 1:
                    for j in range(n - window_size, n):
                        if np.abs(amps_subportadora[j] - x0) > n_sigmas * S0:
                            new_series[subportadora, j] = x0
                else:
                    if np.abs(amps_subportadora[i] - x0) > n_sigmas * S0:
                        new_series[subportadora, i] = x0

        return new_series

    def moving_average(data, window_size, min_periods=1):
        """
        Aplica a média móvel aos dados.
        - data: Numpy array 1D ou 2D.
        - window_size: Tamanho da janela para a média.
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        smoothed_data = np.zeros_like(data)

        half_window = window_size // 2
        for col in range(data.shape[1]):
            for i in range(data.shape[0]):
                start = max(0, i - half_window)
                end = min(data.shape[0], i + half_window + 1)
                window = data[start:end, col]
            
            # Aplica a média apenas se houver observações suficientes
                if len(window) >= min_periods:
                    smoothed_data[i, col] = np.mean(window)
                else:
                    smoothed_data[i, col] = np.nan
        return smoothed_data
    

    def bandpass_filter(data, lowcut, highcut, fs, order=5):
        nyquist = fs / 2
        b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype='band', analog=False, output='ba')
        return lfilter(b, a, data)


    def apply_pca_to_csi(csi_data):
        real_part = np.real(csi_data)
        imaginary_part = np.imag(csi_data)
        csi_data_processed = np.hstack((real_part, imaginary_part))
        scaler = StandardScaler()
        csi_scaled = scaler.fit_transform(csi_data_processed)

        pca = PCA(n_components=1)
        csi_pca = pca.fit_transform(csi_scaled)

        return csi_pca