import time
import importlib
import config
import numpy as np
from plotters.AmpPhaPlotter import Plotter # Amplitude and Phase plotter
decoder = importlib.import_module(f'decoders.{config.decoder}') # This is also an import
from filters import Filters
import matplotlib.pyplot as plt




# def hampel_filter(data, window_size=5, n_sigmas=3):
#     """
#     Aplica o filtro de Hampel para remover outliers.
#     - data: Numpy array 1D ou 2D.
#     - window_size: Tamanho da janela ao redor de cada ponto.
#     - n_sigmas: Multiplicador para identificar outliers baseado no desvio padrão.
#     """
#     if data.ndim == 1:
#         data = data.reshape(-1, 1)
#     filtered_data = data.copy()

#     for col in range(data.shape[1]):
#         for i in range(data.shape[0]):
#             start = max(0, i - window_size)
#             end = min(data.shape[0], i + window_size + 1)
#             window = data[start:end, col]

#             median = np.median(window)
#             mad = np.median(np.abs(window - median))
#             threshold = mad * n_sigmas
#             if abs(data[i, col] - median) > threshold:
#                 filtered_data[i, col] = median
#     return filtered_data



def string_is_int(s):
    '''
    Check if a string is an integer
    '''
    try:
        int(s)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    pcap_filename = input('Pcap file name: ')

    if '.pcap' not in pcap_filename:
        pcap_filename += '.pcap'
    pcap_filepath = '/'.join([config.pcap_fileroot, pcap_filename])

    try:
        samples = decoder.read_pcap(pcap_filepath)
    except FileNotFoundError:
        print(f'File {pcap_filepath} not found.')
        exit(-1)

    if config.plot_samples:
        plotter = Plotter(samples.bandwidth)

    while True:
        command = input('> ')

        if 'help' in command:
            print(config.help_str)
        
        elif 'exit' in command:
            break

        elif ('-' in command) and \
            string_is_int(command.split('-')[0]) and \
            string_is_int(command.split('-')[1]):

            start = int(command.split('-')[0])
            end = int(command.split('-')[1])

            for index in range(start, end+1):
                if config.print_samples:
                    samples.print(index)
                if config.plot_samples:
                    csi = samples.get_csi(
                        index,
                        config.remove_null_subcarriers,
                        config.remove_pilot_subcarriers
                    )

                    csi_filtered = Filters.hampel_filter(csi, window_size=31, n_sigmas=3)
                    csi_smoothed = Filters.moving_average(csi_filtered, window_size=5)
                    csi_bandpass = Filters.bandpass_filter(csi_smoothed, lowcut=0.6, highcut=3.67, fs=1000)
                   
                    csi_pca = Filters.apply_pca_to_csi(csi_bandpass)
                    
                    plotter.update(csi_bandpass)

                elif string_is_int(command):
                    index = int(command)

                    if config.print_samples:
                        samples.print(index)
                    if config.plot_samples:
                        csi = samples.get_csi(
                            index,
                            config.remove_null_subcarriers,
                            config.remove_pilot_subcarriers
                        )

                        # Aplicar o filtro de Hampel e a média móvel (se necessário)
                        csi_filtered = Filters.hampel_filter(csi, window_size=31, n_sigmas=3)
                        csi_smoothed = Filters.moving_average(csi_filtered, window_size=10,min_periods=1)
                        csi_bandpass = Filters.bandpass_filter(csi_smoothed, lowcut=0.6, highcut=3.67, fs=1000)
                        csi_pca = Filters.apply_pca_to_csi(csi_bandpass)


                        # csi_pca = Filters.apply_pca_to_csi(csi_bandpass)

                        plotter.update(csi_bandpass)


        else:
            print('Unknown command. Type help.')