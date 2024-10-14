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
2. links.csv = 9742 baris dan semuanya unik tidak duplikat
3. movies.csv = 9742 basris dan semanua unik tidak duplikat
4. ratings.csv = 100836 baris dan data unik berdasarkan userId sebanyak 610 dan data unik berdasarkan movieId sebanyak 9742
5. tags.csv = 3683 baris dan memiliki data unik sebanyak 1572

Dimana infromasi dari file tersebut adalah
1. links yaitu daftar link pada setiap film terdapat 3 kolom, yaitu movieId, imdbId, dan tmdbId. Kolom movieId dan imdbId lengkap (tidak ada data yang hilang), sedangkan kolom tmdbId memiliki 8 nilai yang hilang. Tipe data untuk movieId dan imdbId adalah integer, sementara tmdbId adalah float (bilangan desimal). Total memori yang digunakan adalah sekitar 228.5 KB
2. movies merupakan daftar film yang tersedia untuk sistem ini memiliki 3 kolom, yaitu movieId, title, dan genres. Semua kolom memiliki data yang lengkap (tidak ada nilai yang hilang). Tipe data untuk movieId adalah integer, sedangkan title dan genres disimpan dalam format string (object). Total memori yang digunakan oleh DataFrame ini adalah lebih dari 228.5 KB.
3. ratings yaitu daftar penilaian dari film yang derikan pengguna ada 4 kolom, yaitu userId, movieId, rating, dan timestamp. Semua kolom memiliki data yang lengkap (tidak ada nilai yang hilang). Tipe data untuk userId, movieId, dan timestamp adalah integer, sementara rating adalah float (bilangan desimal). Total memori yang digunakan oleh DataFrame ini adalah sekitar 3.1 MB
4. tags adalah daftar kata kunci dari film yang ada 4 kolom, yaitu userId, movieId, tag, dan timestamp. Semua kolom memiliki data yang lengkap (tidak ada nilai yang hilang). Tipe data untuk userId, movieId, dan timestamp adalah integer, sementara tag disimpan dalam format string (object). Total memori yang digunakan oleh DataFrame ini adalah lebih dari 115.2 KB.

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

Pemilihan kolom movieId, title, dan genres sangat strategis karena:
* Kolom menyediakan identifikasi yang diperlukan untuk setiap film.
* Menyediakan informasi yang dibutuhkan untuk analisis dan rekomendasi.
* Memastikan bahwa sistem rekomendasi dapat berfungsi secara efektif, dengan memberikan hasil yang relevan dan informatif kepada pengguna.

### Menggunakan TF-IDF (Term Frequency-Inverse Document Frequency)
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

### Data Preparation untuk Collaborative Filtering
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

## Modeling and Result
Proses modeling yang lakukan pada data ini adalah dengan membuat algoritma machine learning, yaitu content based filtering dan collabrative filtering. untuk algoritma content based filtering DIbuat dengan apa yang disukai pengguna, sedangkan untuk content based filtering DIbuat dengan memanfaatkan tingkat rating dari film tersebut.

### Model Development dengan Content-Based Filtering
#### Menghitung Kesamaan Kosinus (Cosine Similarity)
Setelah mendapatkan representasi TF-IDF, langkah selanjutnya adalah menghitung cosine similarity antara film untuk melihat seberapa mirip mereka berdasarkan genre:
```
python
cosine_sim = cosine_similarity(tfidf_matrix)
```
Matriks cosine similarity mengukur kesamaan antar film berdasarkan representasi TF-IDF dari genre mereka. Semakin tinggi nilai cosine similarity, semakin mirip dua film.
Kode ini bertujuan untuk membangun dan menampilkan matriks cosine similarity antar film berdasarkan representasi TF-IDF mereka. Dengan menggunakan DataFrame, informasi kesamaan antara film dapat diorganisir dengan cara yang mudah dipahami dan digunakan dalam sistem rekomendasi. Dengan mencetak bentuk DataFrame dan menampilkan sampel acak, Anda dapat dengan cepat memeriksa struktur data dan memvalidasi hasil yang diperoleh dari perhitungan cosine similarity.

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
  
Fungsi `movie_recommendations` dirancang untuk memberikan rekomendasi film berdasarkan nama film yang dimasukkan oleh pengguna, menggunakan matriks cosine similarity yang sudah dihitung sebelumnya. Fungsi ini mengambil nama film yang dicari dan menemukan indeks film lain yang memiliki kesamaan tertinggi dengan film tersebut menggunakan metode `argpartition`, yang memungkinkan pengambilan elemen teratas tanpa perlu mengurutkan seluruh data. Setelah itu, film yang dicari dihapus dari daftar rekomendasi untuk menghindari duplikasi, dan hasilnya digabungkan dengan DataFrame yang berisi informasi film seperti nama dan genre. Akhirnya, fungsi ini mengembalikan sejumlah `k` rekomendasi teratas dalam bentuk DataFrame, memberikan pengguna alternatif yang relevan berdasarkan preferensi mereka.

#### Contoh Penggunaan Rekomendasi untuk "Deadpool 2 (2018)"
Dengan menjalankan kode:
```
python
movie_recommendations('Deadpool 2 (2018)')
```
Sistem akan merekomendasikan film lain yang mirip dengan Deadpool 2 berdasarkan genre Action, Comedy, Sci-Fi. Kode ini menghasilkan rekomendasi yang sesuai dengan genre yang ada di film tersebut.

#### Top-N rekomendasi CBF dari judul Deadpool 2 (2018)
| No | Movie Name                                             | Genre                   |
|----|-------------------------------------------------------|-------------------------|
| 0  | Men in Black (a.k.a. MIB) (1997)                      | Action|Comedy|Sci-Fi    |
| 1  | Men in Black II (a.k.a. MIIB) (a.k.a. MIB 2) (2002) | Action|Comedy|Sci-Fi    |
| 2  | Ghostbusters (a.k.a. Ghost Busters) (1984)           | Action|Comedy|Sci-Fi    |
| 3  | Logan (2017)                                         | Action|Sci-Fi          |
| 4  | Superman II (1980)                                   | Action|Sci-Fi          |
| 5  | Terminator 2: Judgment Day (1991)                    | Action|Sci-Fi          |
| 6  | Planet of the Apes (1968)                            | Action|Drama|Sci-Fi    |
| 7  | Snowpiercer (2013)                                   | Action|Drama|Sci-Fi    |
| 8  | War of the Worlds, The (1953)                        | Action|Drama|Sci-Fi    |
| 9  | Mystery Science Theater 3000: The Movie (1996)       | Comedy|Sci-Fi          |


### Membangun Model Collaborative Filtering dengan TensorFlow
Model Collaborative Filtering dibuat menggunakan Neural Network dengan embedding untuk representasi user dan movie

#### Class RecommenderNet

Model ini memiliki embedding untuk user dan movie. Proses embedding adalah teknik untuk merepresentasikan user dan movie ke dalam vektor berdimensi rendah, yang kemudian digunakan untuk menghitung tingkat kesukaan user terhadap movie berdasarkan vektor-vektor ini.
```python
self.user_embedding = layers.Embedding(num_users, embedding_size)
self.movie_embedding = layers.Embedding(num_movie, embedding_size)
```
Operasi dot product antara embedding user dan movie digunakan untuk memprediksi tingkat rating yang mungkin diberikan user terhadap movie.

Kode kelas `RecommenderNet`, yang merupakan model neural network yang dibangun menggunakan TensorFlow dan Keras untuk sistem rekomendasi:

1. Definisi Kelas
```
class RecommenderNet(tf.keras.Model):
```
Kelas ini diturunkan dari tf.keras.Model, yang memungkinkan Anda untuk membangun model neural network kustom menggunakan TensorFlow. Ini memberikan struktur dan fungsi yang diperlukan untuk membuat dan melatih model.
2. Inisialisasi Fungsi
```
def __init__(self, num_users, num_movie, embedding_size, **kwargs):
```
* __init__: Merupakan konstruktor yang diubah untuk menginisialisasi objek RecommenderNet.
* Parameter:
  * num_users: Jumlah pengguna dalam dataset.
  * num_movie: Jumlah film dalam dataset.
  * embedding_size: Ukuran vektor embedding yang akan digunakan untuk pengguna dan film.
  * **kwargs: Parameter tambahan yang dapat diteruskan ke superclass.
3. Super Constructor
```
super(RecommenderNet, self).__init__(**kwargs)
```
Memanggil konstruktor superclass untuk memastikan bahwa semua pengaturan dari kelas dasar (tf.keras.Model) juga diterapkan.
4. Layer Embedding
```
self.user_embedding = layers.Embedding(num_users, embedding_size, embeddings_initializer='he_normal', embeddings_regularizer=keras.regularizers.l2(1e-6))
self.user_bias = layers.Embedding(num_users, 1)
self.movie_embedding = layers.Embedding(num_movie, embedding_size, embeddings_initializer='he_normal', embeddings_regularizer=keras.regularizers.l2(1e-6))
self.movie_bias = layers.Embedding(num_movie, 1)
```
* Layer Embedding:
  * user_embedding: Membuat layer embedding untuk pengguna, yang mengonversi ID pengguna menjadi vektor berdimensi embedding_size. Menggunakan he_normal sebagai initializer untuk distribusi normal, serta menambahkan regularisasi L2 untuk mencegah overfitting.

  * user_bias: Membuat layer embedding untuk bias pengguna, dengan ukuran output 1, yang menambahkan bias untuk setiap pengguna.

  * movie_embedding: Mirip dengan user_embedding, tetapi untuk film, mengonversi ID film menjadi vektor berdimensi embedding_size.

  * movie_bias: Membuat layer embedding untuk bias film, juga dengan ukuran output 1.

5. Metode Call
```
def call(self, inputs):
```
Metode ini mendefinisikan bagaimana model akan menghitung output ketika diberikan input.

6. Mengambil Vektor dan Bias
```
user_vector = self.user_embedding(inputs[:, 0]) 
user_bias = self.user_bias(inputs[:, 0]) 
movie_vector = self.movie_embedding(inputs[:, 1]) 
movie_bias = self.movie_bias(inputs[:, 1]) 
```
* inputs: Diasumsikan sebagai tensor dengan dua kolom, di mana kolom pertama adalah ID pengguna dan kolom kedua adalah ID film.
* Mengambil vektor embedding dan bias untuk pengguna dan film berdasarkan input yang diberikan.

7. Menghitung Produk Dot
```
dot_user_movie = tf.tensordot(user_vector, movie_vector, 2)
```
tf.tensordot: Menghitung produk dot antara vektor pengguna dan film. Angka 2 menunjukkan bahwa kedua vektor memiliki dua dimensi, sehingga hasilnya adalah skalar.

8. Menjumlahkan Vektor dan Bias
```
x = dot_user_movie + user_bias + movie_bias
```
Menambahkan hasil produk dot dengan bias pengguna dan bias film untuk mendapatkan nilai akhir.

9. Fungsi Aktivasi
```
return tf.nn.sigmoid(x)
```
Menggunakan fungsi aktivasi sigmoid untuk mengubah nilai output menjadi rentang antara 0 dan 1. Ini berguna untuk memprediksi rating, di mana 0 berarti tidak suka dan 1 berarti suka.

#### Top-N Rekomendasi dari user: 607

##### Film dengan rating tinggi dari penggunaan
| Judul Film                   | Genre                                   |
|------------------------------|-----------------------------------------|
| The Silence of the Lambs      | Crime, Horror, Thriller                 |
| Twister                       | Action, Adventure, Romance, Thriller    |
| Saving Private Ryan           | Action, Drama, War                      |
| Planet of the Apes            | Action, Drama, Sci-Fi                   |
| The Matrix                    | Action, Sci-Fi, Thriller                |

##### 10 Teratas Rekomendasi
| Judul Film                    | Genre                                   |
|-------------------------------|-----------------------------------------|
| The Usual Suspects             | Crime, Mystery, Thriller                |
| Forrest Gump                   | Comedy, Drama, Romance, War             |
| Reservoir Dogs                 | Crime, Mystery, Thriller                |
| Monty Python and the Holy Grail| Adventure, Comedy, Fantasy              |
| One Flew Over the Cuckoo's Nest| Drama                                   |
| The Princess Bride             | Action, Adventure, Comedy, Fantasy, Romance |
| The Graduate                   | Comedy, Drama, Romance                  |
| Cool Hand Luke                 | Drama                                   |
| Fight Club                     | Action, Crime, Drama, Thriller          |
| The Dark Knight                | Action, Crime, Drama, IMAX              |



## Evaluasi Model
### COLABORATIVE FILTERING
#### RMSE (Root Mean Squared Error)
Root Mean Squared Error (RMSE) adalah metrik yang digunakan untuk mengukur seberapa baik prediksi dari sebuah model cocok dengan data aktual. RMSE menghitung akar dari nilai rata-rata kuadrat kesalahan antara nilai prediksi dengan nilai aktual. Metrik ini sering digunakan untuk model regresi atau ketika kita ingin mengetahui seberapa besar rata-rata kesalahan dalam satuan aslinya (misalnya, harga dalam satuan dolar, jarak dalam kilometer).

Secara sederhana, RMSE memberitahu seberapa jauh prediksi model dari nilai aktualnya. Semakin kecil nilai RMSE, semakin baik model dalam memprediksi nilai dengan akurat.

#### Rumus RMSE
Rumus RMSE didefinisikan sebagai berikut:

```
RMSE = sqrt((1/n) * Σ(y_i - ŷ_i)^2)
```

#### Penjelasan dari Rumus:
- **n**: Jumlah total data (observasi).
- **y_i**: Nilai aktual dari data ke-i.
- **ŷ_i**: Nilai prediksi dari model untuk data ke-i.
- **(y_i - ŷ_i)^2**: Ini adalah **error** atau kesalahan kuadrat antara nilai aktual dan prediksi pada data ke-i.

#### Tahapan Perhitungan RMSE:
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

##### Visualisasi RMSE

![download](https://github.com/user-attachments/assets/9a01a2e3-44fb-4645-9dbc-656641f0bd5e)

##### Analisis Grafik
1. Di awal pelatihan (di epoch rendah), RMSE untuk data pelatihan dan pengujian mulai dari nilai yang lebih tinggi (sekitar 0.28–0.29). Ini menunjukkan bahwa pada awalnya, model belum mempelajari pola data dengan baik.

2. Kedua garis (train dan test) turun dengan cepat dalam beberapa epoch pertama (sekitar epoch 0–20). Ini adalah tahap di mana model belajar secara signifikan dari data, sehingga kesalahannya menurun dengan cepat.

3. Setelah epoch ke-20, garis biru (train) terus menurun lebih tajam dibandingkan garis oranye (test). Ini menunjukkan bahwa model menjadi lebih baik pada data pelatihan.

##### Menguji Model dan Memberikan Rekomendasi
Setelah model dilatih, kode berikut digunakan untuk memberikan rekomendasi kepada user berdasarkan preferensi film mereka

Mengambil user dan film yang belum ditonton
```
python
movie_not_watched = movie_df[~movie_df['id'].isin(movie_watched_by_user.movieId.values)]
```
Model memberikan rekomendasi berdasarkan prediksi rating tertinggi untuk film yang belum ditonton oleh user.

### CONTENT-BASE FILTERING
#### PRECISION
Untk matriks yang digunakan di CBF adalah precisiion
![dos_819311f78d87da1e0fd8660171fa58e620211012160253](https://github.com/user-attachments/assets/cec9e128-655e-4724-a57e-b93308ff579d)

#### Penjelasan Rumus:
* of our recommendations that are relevant: Jumlah rekomendasi yang relevan atau sesuai dengan preferensi pengguna.
* of items we recommended: Total jumlah item yang direkomendasikan oleh sistem.

dimana dari rumus diatas dapat diperoleh
Precision  = 9/10
            = 90%

### HASIL EVALUASI
Berdasarkan Hasil Evaluasi dari kejua jenis model diperoleh
1. Untuk hasil RMSE dari Collaborative Filtering didapat 0,22
2. untuk hasil Precision dari Content-Based Filtering didapat 90%

## Kesimpulan
Content-Based Filtering memberikan rekomendasi berdasarkan genre film, menggunakan TF-IDF dan cosine similarity untuk menemukan film serupa.
Collaborative Filtering memberikan rekomendasi berdasarkan interaksi pengguna, menggunakan embedding user dan film untuk memprediksi preferensi rating.
Kedua pendekatan ini sangat berguna untuk memberikan rekomendasi film yang relevan kepada pengguna, berdasarkan genre (Content-Based) dan preferensi pengguna (Collaborative Filtering).
