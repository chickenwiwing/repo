
import pandas as pd
import numpy as np
import streamlit as st

# Judul aplikasi
st.title(":red[DO DETECTOR]")
st.title("Prediksi Dropout dengan Model Markov")

# Load data
try:
    data = pd.read_csv("student_dataset_500.csv")
except FileNotFoundError:
    st.error("File student_dataset_500.csv tidak ditemukan. Pastikan file ada di direktori yang sama.")
    st.stop()

# Kategorisasi (fungsi yang sama)
def kategorisasi(nilai, kategori):
    if nilai >= kategori[2]:
        return "tinggi"
    elif nilai >= kategori[1]:
        return "sedang"
    else:
        return "rendah"

data['Attendance_Cat'] = data['Attendance'].apply(lambda x: kategorisasi(x, [0, 60, 80]))
data['Grades_Cat'] = data['Grades'].apply(lambda x: kategorisasi(x, [0, 60, 80]))
data['State'] = data['Attendance_Cat'] + "_" + data['Grades_Cat'] + "_" + data['Behavior']

# Matriks transisi (sama)
states = data['State'].unique()
n_states = len(states)
transition_matrix = pd.DataFrame(np.zeros((n_states, n_states)), index=states, columns=states)

for i in range(len(data) - 1):
    current_state = data['State'][i]
    next_state = data['State'][i+1]
    transition_matrix.loc[current_state, next_state] += 1

transition_matrix = transition_matrix.div(transition_matrix.sum(axis=1), axis=0)

# Fungsi prediksi (sama)
def prediksi_dropout(state_awal,transition_matrix):
    if "rendah" in state_awal:
        return "Potensi Dropout Tinggi"
    elif "sedang" in state_awal:
        return 'Potensi Dropout Sedang'
    else:
        return "Potensi Dropout Rendah"

# Input terpisah dengan Streamlit
attendance_input = st.number_input("Masukkan Nilai Kehadiran (0-100)", min_value=0, max_value=100, value=70) #ditambahkan validasi input
grades_input = st.number_input("Masukkan Nilai (0-100)", min_value=0, max_value=100, value=65) #ditambahkan validasi input
behavior_input = st.selectbox("Pilih Perilaku", ["good", "average", "poor"])

# Proses input dan prediksi
if st.button("Prediksi"):
    attendance_cat = kategorisasi(attendance_input, [0, 60, 80])
    grades_cat = kategorisasi(grades_input, [0, 60, 80])
    state_input = attendance_cat + "_" + grades_cat + "_" + behavior_input

    if state_input in transition_matrix.index:
        prediksi = prediksi_dropout(state_input, transition_matrix)
        st.write(f"Prediksi untuk siswa dengan state {state_input}: **{prediksi}**")

        st.subheader("Matriks Transisi:")
        st.dataframe(transition_matrix.style.format("{:.2f}"))
        
    else:
      st.error("Kombinasi input tidak valid, kemungkinan data training tidak memiliki kombinasi ini.")
      st.write("Berikut state yang ada di data training : ",list(transition_matrix.index)) #menampilkan state yang ada didata training

# Informasi tambahan
st.write("Data yang digunakan untuk model ini berasal dari file `synthetic_student_dropout_datad.csv`.")
