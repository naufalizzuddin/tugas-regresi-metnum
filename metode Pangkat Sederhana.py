import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Membaca data dari file CSV
data = pd.read_csv('Student_Performance.csv')

# Mengambil kolom yang relevan
NL = data['Sample Question Papers Practiced'].values
NT = data['Performance Index'].values

# Menghindari nilai nol atau negatif
NL = NL[NL > 0]
NT = NT[:len(NL)]  # Sesuaikan panjang NT dengan NL setelah filtering

# Memisahkan data menjadi data pelatihan dan pengujian
NL_train, NL_test, NT_train, NT_test = train_test_split(NL, NT, test_size=0.2, random_state=42)

# Model pangkat sederhana: NT = a * (NL^b)
# Mengubah NL menjadi log(NL) dan NT menjadi log(NT) untuk mendapatkan model linier
log_NL_train = np.log(NL_train)
log_NT_train = np.log(NT_train)

# Melakukan regresi linier pada log(NL) dan log(NT)
coefficients = np.polyfit(log_NL_train, log_NT_train, 1)
b, log_a = coefficients
a = np.exp(log_a)

# Menampilkan koefisien regresi
print(f"Koefisien regresi: a = {a}, b = {b}")

# Fungsi untuk menghitung NT berdasarkan model pangkat sederhana
def predict_NT(NL, a, b):
    return a * (NL ** b)

# Memprediksi nilai NT pada data pengujian
NT_pred = predict_NT(NL_test, a, b)

# Menghitung galat RMS
rms_error = np.sqrt(mean_squared_error(NT_test, NT_pred))
print(f"Galat RMS: {rms_error:.2f}")

# Plot grafik titik data dan hasil regresinya
plt.scatter(NL, NT, label='Data Aktual', color='blue')
NL_line = np.linspace(min(NL), max(NL), 100)
NT_line = predict_NT(NL_line, a, b)
plt.plot(NL_line, NT_line, label='Model Pangkat Sederhana', color='red')
plt.title('Regresi Pangkat Sederhana antara Jumlah Latihan Soal dan Nilai Ujian Siswa')
plt.xlabel('Jumlah Latihan Soal (NL)')
plt.ylabel('Nilai Ujian Siswa (NT)')
plt.legend()
plt.show()
