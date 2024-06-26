# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 08:31:44 2023

@author: pc
"""

# Kumpulan Library yang digunakan
import pandas as pd
import numpy as np
import calendar
import streamlit as st
import base64
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from streamlit_gsheets import GSheetsConnection
from st_on_hover_tabs import on_hover_tabs
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
st.set_page_config(layout="wide")

# Library untuk mengabaikan warnings
import warnings
warnings.filterwarnings('ignore')


# Mengkoneksikan dengan database spreadsheet
conn = st.connection("gsheets", type=GSheetsConnection)
existing_data = conn.read(
    worksheet="DB_PE", 
    ttl=60,
    usecols=list(range(17)))
existing_data = existing_data.dropna(how='all')

# Ubah tampilan dataframe menjadi string
existing_data = existing_data.astype(str)

# Menghilangkan angka 0 dibelakang koma pada variabel tertentu
existing_data['Nomor Induk Keluarga'] = existing_data['Nomor Induk Keluarga'].apply(lambda x: x.split('.')[0])
existing_data['Tahun Pengukuran'] = existing_data['Tahun Pengukuran'].apply(lambda x: x.split('.')[0])
existing_data['Nomor Telepon'] = existing_data['Nomor Telepon'].apply(lambda x: x.split('.')[0])
existing_data['Tekanan Darah Sistolik (mmHg)'] = existing_data['Tekanan Darah Sistolik (mmHg)'].apply(lambda x: x.split('.')[0])
existing_data['Tekanan Darah Diastolik (mmHg)'] = existing_data['Tekanan Darah Diastolik (mmHg)'].apply(lambda x: x.split('.')[0])
existing_data['Usia (tahun)'] = existing_data['Usia (tahun)'].apply(lambda x: x.split('.')[0])
existing_data['Jumlah Kelahiran Hidup'] = existing_data['Jumlah Kelahiran Hidup'].apply(lambda x: x.split('.')[0])
existing_data['Pernah mengalami tekanan darah tinggi ?'] = existing_data['Pernah mengalami tekanan darah tinggi ?'].apply(lambda x: x.split('.')[0])
existing_data['Pernah mengalami preeklamsia ?'] = existing_data['Pernah mengalami preeklamsia ?'].apply(lambda x: x.split('.')[0]) 



# Load saved model
df = pd.read_csv('dataset_versi_normal.csv')
df['level_risiko'].replace({"High": "3", "Moderate": "2", "Low" : "1"}, inplace=True)
df["level_risiko"] = df["level_risiko"].astype("int64")

# Memisahkan variabel independen dan dependen
x = df.drop (columns="level_risiko", axis=1)
y = df['level_risiko']

# Lakukan standarisasi untuk normalisasi data
scaler = StandardScaler()
scaler.fit(x)         
standarized_data = scaler.transform(x)

#Masukan hasil standarisasi data ke variabel X
X = standarized_data
Y = df['level_risiko']

# Buat model
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=42, stratify=Y)

forest = RandomForestClassifier(criterion='gini', max_depth=20, n_estimators=100, min_samples_leaf=1, min_samples_split=2)
forest.fit(X_train, Y_train)

# Membuat data tahun, bulan dan puskesmas
daftar_puskesmas = ['Ajung', 'Ambulu', 'Andongsari', 'Arjasa', 'Balung', 'Bangsalsari',
                    'Banjarsengon', 'Cakru', 'Curahnongko', 'Gladakpakem', 'Gumukmas',
                    'Jelbuk', 'Jember Kidul', 'Jenggawah', 'Jombang', 'Kalisat', 'Kaliwates',
                    'Karangduren', 'Kasiyan', 'Kemuningsari Kidul', 'Kencong', 'Kalisat', 'Klatakan',
                    'Ledokombo', 'Lojejer', 'Mangli', 'Mayang', 'Mumbulsari', 'Nogosari',
                    'Pakusari', 'Paleran', 'Panti', 'Patrang', 'Puger', 'Rambipuji',
                    'Rowotengah', 'Sabrang', 'Semboro', 'Silo I', 'Silo II', 'Sukorambi',
                    'Sukowono', 'Sumberbaru', 'Sumberbaru', 'Sumberjambe', 'Sumbersari', 'Tanggul',
                    'Tembokrejo', 'Tempurejo', 'Umbulsari', 'Wuluhan']

tahun = [datetime.today().year, datetime.today().year + 1]
tahun_def = [t for t in range(2022, datetime.today().year + 1)]
bulan = list(calendar.month_name[1:])


# Membuat fungsi untuk melakukan klasifikasi level risiko
def get_value(val,my_dict):

          for key ,value in my_dict.items():

            if val == key:

              return value
          
def preeklamsia_risk_level(input_data):  
    # Ubah data yang diinput menjadi array
    input_data_as_numpy_array = np.array(input_data)

    input_data_reshape = input_data_as_numpy_array.reshape(1, -1)
    std_data = scaler.transform(input_data_reshape)

    prediction = forest.predict(std_data)

    # Ubah class hasil prediksi menjadi integer agar dapat dibaca oleh model
    predicted_class = int(prediction[0])

    if predicted_class == 1:
        risk_level = "Anda memiliki risiko rendah untuk terkena preeklamsia"
        color = "green"
    elif predicted_class == 2:
        risk_level = "Anda memiliki risiko sedang untuk terkena preeklamsia"
        color = "orange"
    else:
        risk_level = "Anda memiliki risiko tinggi untuk terkena preeklamsia"
        color = "red"
    
    # Construct the message with the specified color
    risk_level_message = f"<span style='color:{color}'>{risk_level}</span>"

    # Get the probabilities for each class
    probabilities = forest.predict_proba(std_data)[0]

    return risk_level_message, probabilities



with st.sidebar:
    with st.sidebar:
        tabs = on_hover_tabs(tabName=['Dashboard', 'Deteksi Dini', 'Database'], 
                             iconName=['analytics', 'monitor heart', 'economy'],
                             styles = {'navtab': {'background-color':'#E75480',
                                                  'color': '#FFFFFF',
                                                  'font-size': '18px',
                                                  'transition': '.2s',
                                                  'white-space': 'nowrap',
                                                  'font-weight': 'bold'},
                                       'tabOptionsStyle': {':hover :hover': {'color': 'red',
                                                                      'cursor': 'pointer'}},
                                       'iconStyle':{'position':'fixed',
                                                    'left':'9px',
                                                    'text-align': 'left',
                                                    'font-size': '30px'},
                                       'tabStyle' : {'list-style-type': 'None',
                                                     'margin-bottom': '30px',
                                                     'padding-left': '30px'}},
                             key="1")

with open("logo-project.png", "rb") as img_file:
    img_byte = img_file.read()

# Mengubah byte menjadi base64 string
img_str = base64.b64encode(img_byte).decode()

# Membuat URL untuk gambar
img_url = f"data:image/png;base64,{img_str}"

# Menampilkan gambar di bagian bawah sidebar
st.sidebar.image(img_url, use_column_width=True)





# Halaman dashboard
if tabs == 'Dashboard':
    
    # Judul Halaman
    st.markdown('<h1 style="color:#E75480;">Dashboard Deteksi Dini Preeklamsia Kabupaten Jember, Jawa Timur</h1>', unsafe_allow_html=True)
    
    #def get_latest_year():
    def get_latest_year():
        return existing_data['Tahun Pengukuran'].max()

    def main():
        # Mendapatkan tahun terbaru
        tahun_terbaru = get_latest_year()
        tahun_sebelumnya = str(int(tahun_terbaru) - 1)  # Konversi ke integer lalu kembali ke string

        # Filter data untuk tahun terbaru dan tahun sebelumnya
        db_data_tahun_terbaru = existing_data[existing_data['Tahun Pengukuran'] == tahun_terbaru]
        db_data_tahun_sebelumnya = existing_data[existing_data['Tahun Pengukuran'] == tahun_sebelumnya]

        # Menghitung jumlah data risiko preeklamsia yang masuk untuk tahun terbaru dan tahun sebelumnya
        total_masuk_tahun_terbaru = db_data_tahun_terbaru.shape[0]
        total_masuk_tahun_sebelumnya = db_data_tahun_sebelumnya.shape[0]

        # Mendapatkan perbandingan antara total data risiko preeklamsia tahun terbaru dan tahun sebelumnya
        perbandingan_total_masuk = total_masuk_tahun_terbaru - total_masuk_tahun_sebelumnya

        # Menghitung jumlah data untuk setiap tingkat risiko preeklamsia
        tidak_berisiko_tahun_terbaru = db_data_tahun_terbaru[db_data_tahun_terbaru['Risiko Preeklamsia'] == 'Tidak Berisiko'].shape[0]
        tidak_berisiko_tahun_sebelumnya = db_data_tahun_sebelumnya[db_data_tahun_sebelumnya['Risiko Preeklamsia'] == 'Tidak Berisiko'].shape[0]

        sedang_tahun_terbaru = db_data_tahun_terbaru[db_data_tahun_terbaru['Risiko Preeklamsia'] == 'Sedang'].shape[0]
        sedang_tahun_sebelumnya = db_data_tahun_sebelumnya[db_data_tahun_sebelumnya['Risiko Preeklamsia'] == 'Sedang'].shape[0]

        tinggi_tahun_terbaru = db_data_tahun_terbaru[db_data_tahun_terbaru['Risiko Preeklamsia'] == 'Tinggi'].shape[0]
        tinggi_tahun_sebelumnya = db_data_tahun_sebelumnya[db_data_tahun_sebelumnya['Risiko Preeklamsia'] == 'Tinggi'].shape[0]

        # Row A
        st.markdown('### Metrik dibandingkan tahun sebelumnya')
        col1, col2, col3, col4 = st.columns(4)
        
        # Menambahkan informasi tentang data risiko preeklamsia pada kolom pertama
        col1.metric("Data Masuk", f"{total_masuk_tahun_terbaru}", f"{perbandingan_total_masuk} Ibu Hamil")
        
        # Menambahkan informasi tentang tingkat risiko preeklamsia pada kolom kedua hingga keempat
        col2.metric("Tidak Berisiko", f"{tidak_berisiko_tahun_terbaru}", f"{tidak_berisiko_tahun_terbaru - tidak_berisiko_tahun_sebelumnya} Ibu Hamil")
        col3.metric("Sedang", f"{sedang_tahun_terbaru}", f"{sedang_tahun_terbaru - sedang_tahun_sebelumnya} Ibu Hamil")
        col4.metric("Tinggi", f"{tinggi_tahun_terbaru}", f"{tinggi_tahun_terbaru - tinggi_tahun_sebelumnya} Ibu Hamil")
    
    if __name__ == "__main__":
        main()
        
    
    def load_data():
       return existing_data

    df = load_data()
    
    # Subheader 1
    st.subheader('Jumlah Deteksi Dini Preeklamsia')
    # Ambil opsi tahun dan bulan yang tersedia dalam dataset
    tahun_filter_options = ['Semua'] + sorted(existing_data['Tahun Pengukuran'].unique())
    bulan_filter_options = ['Semua'] + sorted(existing_data['Bulan Pengukuran'].unique())

    col1, col2 = st.columns(2)

    with col1:
        tahun_filter = st.selectbox('Filter Tahun', tahun_filter_options, key="tahun_filter_2022")

        # Tambahkan filter untuk tahun
        if tahun_filter != 'Semua':
            filtered_data = existing_data[existing_data['Tahun Pengukuran'] == tahun_filter]
        else:
            filtered_data = existing_data

    with col2:
        bulan_filter = st.selectbox('Filter Bulan', bulan_filter_options, key="bulan_filter")

        # Tambahkan filter untuk bulan
        if bulan_filter != 'Semua':
            filtered_data = filtered_data[filtered_data['Bulan Pengukuran'] == bulan_filter]
    
    # Tombol reset filter
    if st.button('Reset Filter'):
        tahun_filter = 'Semua'
        bulan_filter = 'Semua'
        filtered_data = existing_data.copy()

    # Melakukan penghitungan jumlah risiko preeklamsia untuk setiap wilayah puskesmas
    df_counts = filtered_data.groupby(['Wilayah Puskesmas', 'Risiko Preeklamsia']).size().reset_index(name='Jumlah')

    # Mendefinisikan urutan kategori pada sumbu y
    category_order = ["Tidak Berisiko", "Sedang", "Tinggi"]

    # Membuat diagram batang menggunakan Plotly Express
    fig = px.bar(df_counts, x='Wilayah Puskesmas', y='Jumlah', color='Risiko Preeklamsia',
                 color_discrete_map={
                     "Tidak Berisiko": "#adf7b6",
                     "Sedang": "#ffee93",
                     "Tinggi": "#ffc09f"
                 },
                 category_orders={"Risiko Preeklamsia": category_order},
                 labels={"Wilayah Puskesmas": "Wilayah Puskesmas", "Jumlah": "Jumlah", "Risiko Preeklamsia": "Risiko Preeklamsia"})

    # Mengatur ukuran diagram
    fig.update_layout(height=500, width=1100)

    # Menampilkan diagram menggunakan Streamlit
    st.plotly_chart(fig, use_container_width=True)  # Menggunakan use_container_width=True untuk menyesuaikan lebar dengan container
        
    
    #Subheader 2
    st.subheader('Visualisasi Data')
    
    puskesmas_filter_options = ['Semua'] + sorted(existing_data['Wilayah Puskesmas'].unique())
    tahun_filter_options = ['Semua'] + sorted(existing_data['Tahun Pengukuran'].unique())
    
    
    # Buat Kolom dengan lebar yang lebih besar untuk diagram garis
    col1, col2 = st.columns([2, 1])

    with col1:
        # Melakukan penghitungan jumlah pengamatan untuk setiap kategori risiko preeklamsia dalam setiap tahun pengukuran
        df_counts = df.groupby(['Tahun Pengukuran', 'Risiko Preeklamsia']).size().reset_index(name='Count')

        # Membuat pivot agar data dalam format yang diharapkan oleh Plotly Express
        df_pivot = df_counts.pivot(index='Tahun Pengukuran', columns='Risiko Preeklamsia', values='Count').fillna(0)

        # Mendefinisikan urutan kategori pada sumbu y
        category_order = ["Tidak Berisiko", "Sedang", "Tinggi"]

        # Membuat diagram garis menggunakan Plotly Express dengan mengatur urutan kategori pada sumbu y
        fig = px.line(df_pivot, x=df_pivot.index, y=df_pivot.columns, title="Jumlah Berdasarkan Tahun",
                     color_discrete_map={
                         "Tidak Berisiko": "#adf7b6",
                         "Sedang": "#ffee93",
                         "Tinggi": "#ffc09f"
                     },
                     category_orders={"Risiko Preeklamsia": category_order},
                     labels={"value": "Jumlah"})  # Menambah label pada sumbu y

        # Mengatur ukuran dan warna marker
        fig.update_traces(marker=dict(size=10, line=dict(width=2, color='DarkSlateGrey')))

        # Menghilangkan angka setelah koma pada label sumbu x
        fig.update_xaxes(tickmode='linear')

        # Menambahkan mode teks untuk menampilkan jumlah pada diagram
        fig.update_traces(mode='lines+markers+text', textposition='top center')

        # Mengatur ukuran plot
        fig.update_layout(width=600, height=500)

        # Menampilkan diagram menggunakan Streamlit
        st.plotly_chart(fig, theme="streamlit")

    # Atur posisi diagram pie di kolom kedua
    with col2:
        col1, col2 = st.columns([2, 1])
        

        # Filter berdasarkan wilayah puskesmas
        with col1:
            puskesmas_filter = st.selectbox('Filter Puskesmas', puskesmas_filter_options)

        # Filter berdasarkan tahun
        with col2:
            tahun_filter = st.selectbox('Filter Tahun', tahun_filter_options)

        # Filter data berdasarkan pilihan pengguna
        filtered_data = existing_data.copy()  # Copy existing_data untuk mencegah perubahan di tempat
        if puskesmas_filter != 'Semua':
            filtered_data = filtered_data[filtered_data['Wilayah Puskesmas'] == puskesmas_filter]
        if tahun_filter != 'Semua':
            filtered_data = filtered_data[filtered_data['Tahun Pengukuran'] == tahun_filter]

        # Menghitung jumlah risiko preeklamsia
        risiko_counts = filtered_data['Risiko Preeklamsia'].value_counts()
        labels = risiko_counts.index.tolist()
        values = risiko_counts.values.tolist()

        # Membuat plot pie
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3,
                                     marker=dict(colors=["#adf7b6", "#ffee93", "#ffc09f"]))])

        # Mengatur ukuran plot
        fig.update_layout(width=400, height=400)
        
        # Menambahkan judul pada diagram pie
        fig.update_layout(title="Jumlah Berdasarkan Risiko")

        st.plotly_chart(fig, theme="streamlit")
    
    
    
    
    
# Halaman klasifikasi level risiko
if tabs == 'Deteksi Dini':
    
    # Judul Halaman
    st.markdown('<h1 style="color:#E75480;">Deteksi Dini Risiko Preeklamsia</h1>', unsafe_allow_html=True)
    
    #Kolom Data Pribadi
    st.subheader('Profil Ibu')
    col1, col2 = st.columns(2)
    
    with col1:
        nama_ibu = st.text_input('Nama Ibu')
        
    with col2:
        nama_suami = st.text_input('Nama Suami')
        
    with col1:
        alamat = st.text_input('Alamat')
        
    with col2:
        puskesmas = st.selectbox('Wilayah Puskesmas', daftar_puskesmas)
    
    with col1:
        nomor_nik = st.text_input('Nomor Induk Keluarga', max_chars=16)
        
    with col2:
        nomor_telp = st.text_input('Nomor Telepon', max_chars=12)

    # Input validation for telephone number
        try:
            # Attempt to convert the input to an integer
            nomor_telp = int(nomor_telp)
        except ValueError:
            # If conversion fails, display a warning message
            st.warning("Nomor telepon harus berupa angka.")
    
    #Kolom Data Ibu
    st.subheader('Data Pemeriksaan Ibu')
    col1, col2 = st.columns(2)
    
    with col1:
        tahun_pengukuran = st.selectbox('Tahun Pengukuran', tahun, key="tahun")
        
    with col2:
        bulan_pengukuran = st.selectbox('Bulan Pengukuran', bulan, key="bulan")
    
    with col1:
        tinggi_badan = st.number_input('Tinggi Badan (cm)', min_value=100.0, max_value=240.0, value=None)
    
    with col2:
        berat_badan = st.number_input('Berat Badan (kg)', min_value=30.0, max_value=200.0, value=None)
    
    with col1:
        tekanan_darah_sistolik = st.number_input('Tekanan Darah Sistolik (mmHg)', min_value=50, max_value=240, value=None)
    
    with col2:
        tekanan_darah_diastolik = st.number_input('Tekanan Darah Diastolik (mmHg)', min_value=30, max_value=240, value=None)
    
    with col1:
        usia = st.number_input('Usia (tahun)', min_value=15, max_value=70, value=None)
    
    with col2:
        paritas = st.number_input('Jumlah Kelahiran Hidup', min_value=0, max_value=10, value=None)
    
    with col1:
        hipertensi_options = {'Pernah': 1, 'Tidak Pernah': 0}
        hipertensi = st.selectbox('Pernah mengalami tekanan darah tinggi ?', tuple(hipertensi_options.keys()))
        riwayat_hipertensi = get_value(hipertensi,hipertensi_options)
    
    with col2:
        preeklamsia_options = {'Pernah': 1, 'Tidak Pernah': 0}
        preeklamsia = st.selectbox('Pernah mengalami preeklamsia ?', tuple(preeklamsia_options.keys()))
        riwayat_preeklamsia = get_value(preeklamsia,preeklamsia_options)
        
    
    # Kode untuk prediksi
    risiko = ''
    prediction_percentage = ''
    
    # Membuat tombol untuk klasifikasi level risiko
    if st.button('**Klasifikasi Level Risiko** :heart:' ):
        if nama_ibu is None or nama_suami is None or alamat is None or nomor_nik is None or nomor_telp is None or tinggi_badan is None or berat_badan is None or tekanan_darah_sistolik is None or tekanan_darah_diastolik is None or usia is None or paritas is None or riwayat_hipertensi is None or riwayat_preeklamsia is None:
            st.warning("**Mohon lengkapi semua isian terlebih dahulu**")
        else:
            risiko, probabilities = preeklamsia_risk_level([tinggi_badan, berat_badan, tekanan_darah_sistolik, tekanan_darah_diastolik, usia, paritas, riwayat_hipertensi, riwayat_preeklamsia])

            # Get the index of the maximum probability
            max_prob_index = np.argmax(probabilities)
            # Get the corresponding probability
            max_prob = probabilities[max_prob_index]
            # Convert the probability to percentage
            prediction_percentage = f"Confidence: {max_prob * 100:.2f}%"
        st.markdown(risiko, unsafe_allow_html=True)  # Render HTML directly
        st.info(prediction_percentage)

        
        if risiko == 'Anda **risiko rendah** untuk terkena preeklamsia':
            risiko_pe = "Rendah"
        elif risiko == 'Anda memiliki **risiko sedang** untuk terkena preeklamsia':
            risiko_pe = "Sedang"
        else:
            risiko_pe = "Tinggi"
    
    
    # Memasukan hasil input kedalam database spreadsheet yang telah dibuat
        db_data = pd.DataFrame(
            [
                {
                    'Nama Ibu': nama_ibu,
                    'Nama Suami': nama_suami,
                    'Alamat': alamat,
                    'Wilayah Puskesmas': puskesmas,
                    'Nomor Induk Keluarga': nomor_nik,
                    'Nomor Telepon': nomor_telp,
                    'Tahun Pengukuran': tahun_pengukuran,
                    'Bulan Pengukuran': bulan_pengukuran,
                    'Tinggi Badan (cm)': tinggi_badan,
                    'Berat Badan (kg)': berat_badan,
                    'Tekanan Darah Sistolik (mmHg)': tekanan_darah_sistolik,
                    'Tekanan Darah Diastolik (mmHg)': tekanan_darah_diastolik,
                    'Usia (tahun)': usia,
                    'Jumlah Kelahiran Hidup': paritas,
                    'Pernah mengalami tekanan darah tinggi ?': riwayat_hipertensi,
                    'Pernah mengalami preeklamsia ?': riwayat_preeklamsia,
                    'Risiko Preeklamsia': risiko_pe
                    }
                ]
            )
       
        if not db_data.isnull().values.any():
            updated_df = pd.concat([existing_data, db_data], ignore_index=True)
            conn.update(worksheet='DB_PE', data=updated_df)
        
        
    
    
# Halaman database
if tabs == 'Database':
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['pre-authorized']
    )
    
    authenticator.login()
    
    if st.session_state["authentication_status"]:
        authenticator.logout()
        st.markdown(f'<h2><span style="font-size: px;">Halo,</span> <strong>{st.session_state["name"]}</strong></h2>', unsafe_allow_html=True)
        # Judul Halaman
        st.markdown('<h1 style="color:#E75480;">Database Hasil Skrining Preeklamsia</h1>', unsafe_allow_html=True)
        tahun_filter_options = ['Semua'] + sorted(existing_data['Tahun Pengukuran'].unique())
        bulan_filter_options = ['Semua'] + sorted(existing_data['Bulan Pengukuran'].unique())
        risiko_filter_options = ['Semua'] + sorted(existing_data['Risiko Preeklamsia'].unique())
        puskesmas_filter_options = ['Semua'] + sorted(existing_data['Wilayah Puskesmas'].unique())
        
        
        def warna_risiko(risiko_pe):
            if risiko_pe == 'Tinggi':
                color = 'rgba(255, 0, 0, 0.3)'
            elif risiko_pe == 'Sedang':
                color = 'rgba(255, 255, 0, 0.3)'
            else:
                color = 'rgba(0, 128, 0, 0.3)'
            return f'background-color: {color}'
        

        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            puskesmas_filter = st.selectbox('Filter Puskesmas', puskesmas_filter_options)
            
        with col2:
            bulan_filter = st.selectbox('Filter Bulan', bulan_filter_options, key="bulan_filter")
            
        with col3:
            tahun_filter = st.selectbox('Filter Tahun', tahun_filter_options, key="tahun_filter_2022")
        
        with col4:
            risiko_filter = st.selectbox('Filter Risiko Preeklamsia', risiko_filter_options)
        
        # Tombol reset filter
        if st.button('Reset Filter'):
            puskesmas_filter = 'Semua'
            bulan_filter = 'Semua'
            tahun_filter = 'Semua'
            risiko_filter = 'Semua'
        
        # Filter data berdasarkan pilihan pengguna
        filtered_data = existing_data.copy()  # Copy existing_data untuk mencegah perubahan di tempat
        if puskesmas_filter != 'Semua':
            filtered_data = filtered_data[filtered_data['Wilayah Puskesmas'] == puskesmas_filter]
        if bulan_filter != 'Semua':
            filtered_data = filtered_data[filtered_data['Bulan Pengukuran'] == bulan_filter]
        if tahun_filter != 'Semua':
            filtered_data = filtered_data[filtered_data['Tahun Pengukuran'] == tahun_filter]  # Perbaikan di sini
        if risiko_filter != 'Semua':
            filtered_data = filtered_data[filtered_data['Risiko Preeklamsia'] == risiko_filter]

        # Apply background color based on risk
        st.dataframe(filtered_data.style.applymap(warna_risiko, subset=['Risiko Preeklamsia']))
    elif st.session_state["authentication_status"] is False:
        st.error('Username atau password salah')
    elif st.session_state["authentication_status"] is None:
        st.warning('Silahkan masukan username dan password')