import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Membaca data dari file CSV
data = pd.read_csv('Student_Performance.csv')

# Mengambil kolom yang relevan
NL = data['Sample Question Papers Practiced'].values.reshape(-1, 1)
NT = data['Performance Index'].values

# Membuat model regresi linear
model = LinearRegression()
model.fit(NL, NT)

# Memprediksi nilai NT berdasarkan model regresi
NT_pred = model.predict(NL)

# Menghitung galat RMS
rms_error = np.sqrt(mean_squared_error(NT, NT_pred))

# Plot grafik titik data dan hasil regresinya
plt.scatter(NL, NT, color='blue', label='Data Sebenarnya')
plt.plot(NL, NT_pred, color='red', label='Regresi Linear')
plt.xlabel('Jumlah Latihan Soal (NL)')
plt.ylabel('Nilai Ujian Siswa (NT)')
plt.title('Regresi Linear antara Jumlah Latihan Soal dan Nilai Ujian Siswa')
plt.legend()
plt.show()

print(f'Galat RMS: {rms_error:.2f}')
