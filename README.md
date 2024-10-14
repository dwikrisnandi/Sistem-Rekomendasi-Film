# Laporan Poroyek Machine Learning - Dwi Krisnandi

Submission 2 Machine Learning Terapan - Sistem Rekomendasi Film

## Project Overview

Proyek pengembangan sistem rekomendasi film sangat penting untuk diselesaikan karena dapat meningkatkan pengalaman pengguna dengan memberikan rekomendasi yang lebih sesuai dengan minat, mengatasi masalah informasi berlebih yang sering dihadapi pengguna, serta memberikan manfaat ekonomi yang signifikan bagi perusahaan streaming. Sistem rekomendasi yang efektif dapat membantu pengguna menemukan film yang sesuai dengan preferensi mereka, sehingga mengurangi waktu pencarian dan meningkatkan kepuasan pengguna secara keseluruhan. Penelitian menunjukkan bahwa algoritma rekomendasi berbasis machine learning dapat meningkatkan akurasi dan kinerja rekomendasi, sementara sistem yang baik dapat menghemat biaya dan meningkatkan retensi pelanggan, seperti yang dialami oleh Netflix yang mengklaim menghemat sekitar $1 miliar per tahun berkat algoritma rekomendasi mereka. Selain itu, pemahaman yang lebih dalam tentang interaksi pengguna dengan algoritma rekomendasi dapat membantu pengembang menciptakan sistem yang lebih adaptif dan responsif terhadap kebutuhan pengguna, menjadikan proyek ini tidak hanya bermanfaat bagi pengguna tetapi juga memberikan keuntungan yang substansial bagi industri hiburan secara keseluruhan.

Sumber : 
1. Zhang, "Movie Recommendation System Based on Machine Learning," *Highlights in Business Economics and Management* (2023). doi:10.54097/hbem.v21i.14740. Penelitian ini menunjukkan bagaimana sistem rekomendasi berbasis machine learning dapat meningkatkan akurasi dan kinerja rekomendasi dengan mengintegrasikan tag fitur. [link](https://drpress.org/ojs/index.php/HBEM/article/view/14740)

2. Singh, A., & Soundarabai, "Collaborative filtering in movie Recommendation System based on Rating and Genre," *Ijarcce* (2017). doi:10.17148/ijarcce.2017.63107. Artikel ini membahas bagaimana sistem rekomendasi membantu merekomendasikan item berdasarkan kebutuhan pengguna yang bervariasi, mengurangi kebingungan dalam pencarian konten. [link](https://www.researchgate.net/publication/318293298_Collaborative_filtering_in_movie_Recommendation_System_based_on_Rating_and_Genre)

## Bussiness Understanding

**Problem Statements**

Dalam konteks proyek rekomendasi film, pernyataan masalah yang dihadapi adalah kesulitan pengguna dalam menemukan film yang sesuai dengan preferensi diantara banyaknya pilihan yang tersedia.

**Goals**

Tujuan dari proyek ini adalah untuk mengembangkan sistem rekomendasi film yang dapat memberikan saran yang relevan dan personal kepada pengguna. Diharapkan pengguna dapat dengan mudah menemukan film yang sesuai dengan selera merek.

**Solution Approach**

Untuk mencapai tujuan tersebut, dua pendekatan solusi yang dapat diterapkan adalah:

1. **Content-Based Filtering**: Pendekatan ini berfokus pada analisis karakteristik film yang telah ditonton oleh pengguna sebelumnya.

2. **Collaborative Filtering**: Pendekatan ini memanfaatkan data interaksi pengguna dengan film, seperti rating yang diberikan, untuk menemukan pola dan kesamaan antara pengguna.

## Data Understanding
### Informasi Umum
Dataset didownload dari Kaggle dengan sumber data [Movie Recommendation Data](https://www.kaggle.com/datasets/rohan4050/movie-recommendation-data)

Berisikan 5 file dengan jumlah data setiap file
1. movies_metadata.csv = 45466
2. links.csv = 9742
3. movies.csv = 610
4. ratings.csv = 9742
5. tags.csv = 1572

Dimana infromasi dari file tersebut adalah
1. links yaitu daftar link pada setiap film
2. movies merupakan daftar film yang tersedia untuk sistem ini
3. ratings yaitu daftar penilaian dari film yang derikan pengguna
5. tags adalah daftar kata kunci dari film yang ada

### Visualisasi Data
Distribusi Rating
![Visualisasi Distribusi Rating](https://github.com/user-attachments/assets/6b7fcb9c-8fa7-4b15-8eff-d04775cd9848)
Dari hasil visualisi dapat diketahui bahwa rating berada diantara range 0-5 dengan frekuaensi paling banyak di kisaran 26.000

Distribusi Genre
![download](https://github.com/user-attachments/assets/96f4b03e-0d0b-4f8d-bf44-b955126406a8)
Dari hasil visualisasi di atas dapat diketahui 10 genre terbanyak dalam dataset ini dan genre Drama paling banyak dengan kisaran data 1000 lebih

## Data Preparation
### Data Preprocessing
1. Menggabungkan data berdasarkan movieId

Menggabungkan semua data dari beberapa file csv dengan fungsi concate berdasarkan movieId kedalam variabel movie_all. Tujuan utama dari kode ini adalah untuk mendapatkan kumpulan movieId yang unik dari berbagai sumber data (misalnya data link, movie, rating, dan tags), serta menghapus duplikasi sehingga hanya movieId yang benar-benar unik yang tersisa, lalu mencetak jumlahnya.
```
Python
# Menggabungkan seluruh movieID pada kategori movie
movie_all = np.concatenate((
    df_links.movieId.unique(),
    df_movies.movieId.unique(),
    df_rating.movieId.unique(),
    df_tags.movieId.unique(),
))
 
# Mengurutkan data dan menghapus data yang sama
movie_all = np.sort(np.unique(movie_all))
 
print('Jumlah seluruh data movie berdasarkan movieID: ', len(movie_all))
```
hasil dari penggabungan movie didapat sebanyak 9742 data

2. Menggabungkan data berdasarkan userId

Langkah ini serupa dengan penggabungan movieId, tetapi kali ini menggabungkan userId dari df_rating dan df_tags:
```
python
Salin kode
user_all = np.concatenate((
    df_rating.userId.unique(),
    df_tags.userId.unique(),
))
user_all = np.sort(np.unique(user_all))
print('Jumlah seluruh user: ', len(user_all))
```
Ditemukan 610 data userId unik.

3. Menggabungkan Dataframe

Bagian ini bertujuan untuk menggabungkan beberapa dataframe berdasarkan movieId dan memasukkan informasi tambahan dari genre dan tag film.

* Menggabungkan rating dan movie
Menggabungkan dataframe df_rating dengan df_movies berdasarkan movieId dan mengambil kolom 'title' dan 'genres' dari df_movies:
```
python
all_movie_name = pd.merge(df_rating, df_movies[['movieId','title','genres']], on='movieId', how='left')
```
Ini bertujuan untuk menggabungkan rating dengan detail film, seperti judul dan genre.

* Menggabungkan tags dengan movie
Setelah mendapatkan hasil penggabungan all_movie_name, kode ini menggabungkan dataframe df_tags untuk menambahkan tag yang relevan:
```
python
all_movie = pd.merge(all_movie_name, df_tags[['movieId','tag']], on='movieId', how='left')
```
Ini memungkinkan data film mencakup informasi tag yang tersedia untuk setiap movieId.

### Mengatasi Missing Value
Setelah penggabungan, terdapat nilai kosong (missing value) yang perlu ditangani. Kode ini menghitung jumlah missing value:
```
python
all_movie.isnull().sum()
```
Dari hasil ini, ada 52.549 data kosong di kolom tag. Untuk mengatasi masalah ini, data yang memiliki nilai kosong dihapus menggunakan dropna():
```
python
all_movie_clean = all_movie.dropna()
```
Langkah ini mengurangi dataset dari 285.762 baris menjadi 233.213 baris.

### Mengurutkan dan Menghapus Duplikasi
Setelah pembersihan, kode mengurutkan dataset berdasarkan movieId:
```
python
fix_movie = all_movie_clean.sort_values('movieId', ascending=True)
```
Selanjutnya, kode menghapus duplikasi movieId dengan drop_duplicates():
```
python
preparation = fix_movie.drop_duplicates('movieId')
```
Langkah ini memastikan bahwa setiap movieId hanya muncul sekali di dataset.

### Konversi ke List
Data hasil pembersihan kemudian dikonversi dari bentuk dataframe menjadi list untuk memudahkan manipulasi dalam bentuk lain:
```
python
movie_id = preparation['movieId'].tolist()
movie_name = preparation['title'].tolist()
movie_genre = preparation['genres'].tolist()
```
Langkah ini menyiapkan data movieId, title, dan genres dalam bentuk list.

### Membuat Dataframe Baru
Setelah semua data dikonversi ke list, kode membuat dataframe baru dengan movieId, movie_name, dan genre:
```
python
movie_new = pd.DataFrame({
    'id': movie_id,
    'movie_name': movie_name,
    'genre': movie_genre
})
```
Ini menghasilkan dataframe yang bersih dan siap untuk proses selanjutnya, seperti pemodelan atau analisis lebih lanjut.

## Modeling and Result
Proses modeling yang lakukan pada data ini adalah dengan membuat algoritma machine learning, yaitu content based filtering dan collabrative filtering. untuk algoritma content based filtering DIbuat dengan apa yang disukai pengguna, sedangkan untuk content based filtering DIbuat dengan memanfaatkan tingkat rating dari film tersebut.

### Model Development dengan Content-Based Filtering
#### Menggunakan TF-IDF (Term Frequency-Inverse Document Frequency)
TF-IDF adalah teknik untuk mengubah data teks menjadi representasi numerik yang mencerminkan pentingnya kata dalam dokumen (dalam hal ini, genre film) dibandingkan dengan seluruh dokumen dalam dataset.
```
python
tf = TfidfVectorizer()
tf.fit(movie_new['genre'])
```
Inisialisasi TfidfVectorizer: digunakan untuk menghitung skor TF-IDF berdasarkan kolom genre pada data film.
```
python
tfidf_matrix = tf.fit_transform(movie_new['genre'])
```
Transformasi genre menjadi matriks TF-IDF, di mana setiap elemen dalam matriks mewakili skor TF-IDF untuk genre tertentu pada film tertentu.

#### Menghitung Kesamaan Kosinus (Cosine Similarity)
Setelah mendapatkan representasi TF-IDF, langkah selanjutnya adalah menghitung cosine similarity antara film untuk melihat seberapa mirip mereka berdasarkan genre:
```
python
cosine_sim = cosine_similarity(tfidf_matrix)
```
Matriks cosine similarity mengukur kesamaan antar film berdasarkan representasi TF-IDF dari genre mereka. Semakin tinggi nilai cosine similarity, semakin mirip dua film.

#### Membuat Fungsi Rekomendasi
Kode selanjutnya mendefinisikan fungsi movie_recommendations untuk memberikan rekomendasi film berdasarkan tingkat kesamaan (cosine similarity):
```
python
def movie_recommendations(nama_movie, similarity_data=cosine_sim_df, items=movie_new[['movie_name', 'genre']], k=10)
```
Fungsi ini menerima judul film dan mencari film lain yang paling mirip dengan film tersebut. Langkah-langkahnya:
* Mengambil film yang memiliki kesamaan tertinggi.
* Menghindari merekomendasikan film yang sama.
* Mengembalikan rekomendasi dalam bentuk DataFrame yang mencakup nama film dan genre.

#### Contoh Penggunaan Rekomendasi untuk "Deadpool 2 (2018)"
Dengan menjalankan kode:
```
python
movie_recommendations('Deadpool 2 (2018)')
```
Sistem akan merekomendasikan film lain yang mirip dengan Deadpool 2 berdasarkan genre Action, Comedy, Sci-Fi. Kode ini menghasilkan rekomendasi yang sesuai dengan genre yang ada di film tersebut.


### Model Development dengan Collaborative Filtering
#### Data Preparation untuk Collaborative Filtering
Collaborative filtering menggunakan data interaksi pengguna (seperti rating film) untuk memberikan rekomendasi. Dalam kode ini, dilakukan persiapan data sebagai berikut:

1.Encoding userId dan movieId

Encoding dilakukan untuk mengonversi userId dan movieId menjadi bentuk numerik agar bisa digunakan dalam model machine learning
```
python
user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
movie_to_movie_encoded = {x: i for i, x in enumerate(movie_ids)}
```
2. Membagi dataset menjadi train (80%) dan validation (20%)

Data dibagi menjadi data latih dan validasi untuk menguji performa model.
```
python
train_indices = int(0.8 * df.shape[0])
```


### Membangun Model Collaborative Filtering dengan TensorFlow
Model Collaborative Filtering dibuat menggunakan Neural Network dengan embedding untuk representasi user dan movie

#### Class RecommenderNet

Model ini memiliki embedding untuk user dan movie. Proses embedding adalah teknik untuk merepresentasikan user dan movie ke dalam vektor berdimensi rendah, yang kemudian digunakan untuk menghitung tingkat kesukaan user terhadap movie berdasarkan vektor-vektor ini.
```python
self.user_embedding = layers.Embedding(num_users, embedding_size)
self.movie_embedding = layers.Embedding(num_movie, embedding_size)
```
Operasi dot product antara embedding user dan movie digunakan untuk memprediksi tingkat rating yang mungkin diberikan user terhadap movie.

## Evaluasi Model
### RMSE (Root Mean Squared Error)
Root Mean Squared Error (RMSE) adalah metrik yang digunakan untuk mengukur seberapa baik prediksi dari sebuah model cocok dengan data aktual. RMSE menghitung akar dari nilai rata-rata kuadrat kesalahan antara nilai prediksi dengan nilai aktual. Metrik ini sering digunakan untuk model regresi atau ketika kita ingin mengetahui seberapa besar rata-rata kesalahan dalam satuan aslinya (misalnya, harga dalam satuan dolar, jarak dalam kilometer).

Secara sederhana, RMSE memberitahu seberapa jauh prediksi model dari nilai aktualnya. Semakin kecil nilai RMSE, semakin baik model dalam memprediksi nilai dengan akurat.

### Rumus RMSE
Rumus RMSE didefinisikan sebagai berikut:

```
RMSE = sqrt((1/n) * Σ(y_i - ŷ_i)^2)
```

### Penjelasan dari Rumus:
- **n**: Jumlah total data (observasi).
- **y_i**: Nilai aktual dari data ke-i.
- **ŷ_i**: Nilai prediksi dari model untuk data ke-i.
- **(y_i - ŷ_i)^2**: Ini adalah **error** atau kesalahan kuadrat antara nilai aktual dan prediksi pada data ke-i.

### Tahapan Perhitungan RMSE:
1. Hitung **error** (selisih) antara nilai aktual dan prediksi untuk setiap data.
2. Kuadratkan setiap error untuk menghilangkan nilai negatif.
3. Hitung rata-rata dari semua nilai error kuadrat.
4. Ambil akar kuadrat dari nilai rata-rata error tersebut untuk mengembalikan kesalahan ke skala aslinya.

Model dikompilasi menggunakan BinaryCrossentropy sebagai fungsi loss, optimizer Adam, dan metrik Root Mean Squared Error (RMSE):
```
python
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=10e-6),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)
```
EarlyStopping dan ModelCheckpoint digunakan untuk menghindari overfitting dan menyimpan model terbaik selama pelatihan.

#### Visualisasi RMSE
![download](https://github.com/user-attachments/assets/841f574d-9319-4a25-b3f0-fbb4169c0041)
##### Analisis Grafik
1. Di awal pelatihan (di epoch rendah), RMSE untuk data pelatihan dan pengujian mulai dari nilai yang lebih tinggi (sekitar 0.27–0.28). Ini menunjukkan bahwa pada awalnya, model belum mempelajari pola data dengan baik.

2. Kedua garis (train dan test) turun dengan cepat dalam beberapa epoch pertama (sekitar epoch 0–20). Ini adalah tahap di mana model belajar secara signifikan dari data, sehingga kesalahannya menurun dengan cepat.

3. Setelah epoch ke-20, garis biru (train) terus menurun lebih tajam dibandingkan garis oranye (test). Ini menunjukkan bahwa model menjadi lebih baik pada data pelatihan.

#### Menguji Model dan Memberikan Rekomendasi
Setelah model dilatih, kode berikut digunakan untuk memberikan rekomendasi kepada user berdasarkan preferensi film mereka

Mengambil user dan film yang belum ditonton
```
python
movie_not_watched = movie_df[~movie_df['id'].isin(movie_watched_by_user.movieId.values)]
```
Model memberikan rekomendasi berdasarkan prediksi rating tertinggi untuk film yang belum ditonton oleh user.

## Kesimpulan
Content-Based Filtering memberikan rekomendasi berdasarkan genre film, menggunakan TF-IDF dan cosine similarity untuk menemukan film serupa.
Collaborative Filtering memberikan rekomendasi berdasarkan interaksi pengguna, menggunakan embedding user dan film untuk memprediksi preferensi rating.
Kedua pendekatan ini sangat berguna untuk memberikan rekomendasi film yang relevan kepada pengguna, berdasarkan genre (Content-Based) dan preferensi pengguna (Collaborative Filtering).
