# labse-ales123
#Laboratorio 3 de Señales
  
  ## Configuración del Sistema DAQ

Primero, configuramos el sistema de adquisición de datos (DAQ) utilizando la API de NI-DAQmx para Python. Esto incluye la creación de una tarea, la adición de un canal de voltaje y la configuración de la frecuencia de muestreo.

		import nidaqmx
		import numpy as np
		import matplotlib.pyplot as plt
		from scipy.signal import butter, filtfilt, hamming

		# Configuración del sistema DAQ
		task = nidaqmx.Task()
		task.ai_channels.add_ai_voltage_chan("Dev1/ai0")  # Ajusta según tu configuración
		task.timing.cfg_samp_clk_timing(1000)  # Frecuencia de muestreo en Hz
  ## Adquisición de Datos
Leemos los datos de la señal EMG desde el sistema DAQ. Aquí, especificamos el número de muestras que queremos adquirir.
		# Adquisición de datos
		data = task.read(number_of_samples_per_channel=10000)
		task.stop()
		task.close()

		# Convertir datos a un array de numpy
		emg_signal = np.array(data)

### Filtrado de la Señal
Aplicamos filtros pasa altas y pasa bajas para eliminar el ruido no deseado de la señal EMG. Utilizamos la biblioteca scipy para crear y aplicar estos filtros.

		# Filtro pasa altas
		def highpass_filter(signal, cutoff, fs, order=5):
			nyquist = 0.5 * fs
			normal_cutoff = cutoff / nyquist
			b, a = butter(order, normal_cutoff, btype='high', analog=False)
			filtered_signal = filtfilt(b, a, signal)
			return filtered_signal

		# Filtro pasa bajas
		def lowpass_filter(signal, cutoff, fs, order=5):
			nyquist = 0.5 * fs
			normal_cutoff = cutoff / nyquist
			b, a = butter(order, normal_cutoff, btype='low', analog=False)
			filtered_signal = filtfilt(b, a, signal)
			return filtered_signal

		# Aplicar filtros
		fs = 1000  # Frecuencia de muestreo
		high_cutoff = 20  # Frecuencia de corte para el filtro pasa altas
		low_cutoff = 450  # Frecuencia de corte para el filtro pasa bajas
		filtered_signal = highpass_filter(emg_signal, high_cutoff, fs)
		filtered_signal = lowpass_filter(filtered_signal, low_cutoff, fs)

##Aventanamiento y Transformada de Fourier

Dividimos la señal filtrada en ventanas de tiempo y aplicamos la Transformada de Fourier (FFT) a cada ventana para obtener el espectro de frecuencias.
		# Función para aventanamiento y FFT
		def windowed_fft(signal, window_size, overlap):
			step = window_size - overlap
			windows = []
			for start in range(0, len(signal) - window_size + 1, step):
				window = signal[start:start + window_size] * hamming(window_size)
				windows.append(np.fft.fft(window))
			return np.array(windows)

		# Parámetros de aventanamiento
		window_size = 256
		overlap = 128
		spectral_windows = windowed_fft(filtered_signal, window_size, overlap)
## Análisis Espectral
Calculamos la frecuencia mediana de cada ventana y visualizamos cómo cambia esta frecuencia a lo largo del tiempo, lo cual puede ser un indicador de fatiga muscular.

# Calcular la frecuencia mediana
def median_frequency(spectrum):
    power_spectrum = np.abs(spectrum) ** 2
    cumulative_power = np.cumsum(power_spectrum)
    total_power = cumulative_power[-1]
    median_freq = np.where(cumulative_power >= total_power / 2)[0][0]
    return median_freq

median_frequencies = [median_frequency(window) for window in spectral_windows]

		# Visualizar la disminución de la frecuencia mediana
		plt.plot(median_frequencies)
		plt.xlabel('Ventana')
		plt.ylabel('Frecuencia Mediana')
		plt.title('Disminución de la Frecuencia Mediana')
		plt.show()
