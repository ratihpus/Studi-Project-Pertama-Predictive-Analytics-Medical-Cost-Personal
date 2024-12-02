# Laporan Proyek Machine Learning - RATIH PUSPITASARI
## Domain Proyek

Domain proyek yang dipilih dalam proyek _machine learning_ ini adalah mengenai prediksi biaya asuransi berdasarkan sejumlah fitur lainnya dengan judul proyek "Predictive Analytics  Medical Cost Personal".

### Latar Belakang
  
Industri asuransi kesehatan memiliki peran penting dalam menyediakan perlindungan finansial terhadap biaya perawatan medis yang tidak terduga. Salah satu tantangan utama yang dihadapi perusahaan asuransi adalah menentukan premi asuransi yang adil dan kompetitif untuk setiap individu berdasarkan profil risiko mereka. Dalam proses ini, analisis data menjadi alat yang sangat penting untuk mengevaluasi berbagai faktor yang memengaruhi biaya asuransi kesehatan. Biaya asuransi (charges) yang dibayarkan oleh pelanggan sering kali dipengaruhi oleh berbagai faktor, seperti usia, jenis kelamin, gaya hidup (misalnya, status perokok), dan kondisi fisik seperti Indeks Massa Tubuh (BMI). Selain itu, faktor lingkungan seperti wilayah tempat tinggal juga dapat memainkan peran signifikan dalam menentukan biaya asuransi. Memahami hubungan antara faktor-faktor ini dan biaya asuransi tidak hanya membantu perusahaan dalam menetapkan premi yang wajar, tetapi juga dapat memberikan wawasan tentang pola risiko yang berbeda di antara kelompok pelanggan.
Dalam konteks ini, pengolahan dataset asuransi kesehatan menjadi sangat relevan. Dengan menggunakan metode analisis data dan model machine learning, seperti regresi linier atau algoritma lainnya, penulis dapat mengidentifikasi pola, memahami faktor dominan yang memengaruhi biaya asuransi, dan bahkan memprediksi biaya tersebut untuk pelanggan baru. Hal ini memungkinkan perusahaan asuransi untuk: 
-   Mengoptimalkan Penetapan Premi: Menentukan premi yang sesuai dengan risiko setiap individu, sehingga menciptakan keseimbangan antara profitabilitas perusahaan dan keterjangkauan bagi pelanggan.
-   Mengidentifikasi Kelompok Risiko Tinggi: Menganalisis kelompok pelanggan dengan risiko kesehatan tinggi, seperti perokok atau individu dengan BMI tinggi, untuk merancang strategi mitigasi risiko.
-   Meningkatkan Kepuasan Pelanggan: Dengan penetapan premi yang lebih transparan dan adil, perusahaan dapat meningkatkan kepercayaan dan kepuasan pelanggan.
-   Efisiensi Operasional: Mengurangi biaya manual untuk penilaian risiko melalui otomasi proses menggunakan prediksi berbasis machine learning.

Dataset asuransi kesehatan ini mencakup variabel seperti usia, jenis kelamin, BMI, jumlah anak dalam tanggungan, status perokok, wilayah tempat tinggal, dan biaya asuransi yang dibayarkan. Dengan mengolah data ini, penulis dapat mengeksplorasi berbagai wawasan, seperti sejauh mana usia atau status perokok memengaruhi biaya asuransi atau apakah ada perbedaan regional dalam penetapan biaya. Selain itu, visualisasi data dapat membantu memahami pola-pola yang muncul dari dataset tersebut, seperti korelasi antara BMI dan biaya asuransi atau distribusi biaya di berbagai kelompok usia. Melalui pendekatan ini, studi kasus ini berkontribusi pada pemahaman yang lebih baik tentang bagaimana berbagai faktor memengaruhi biaya asuransi kesehatan, serta memberikan dasar bagi penerapan machine learning untuk mendukung pengambilan keputusan di sektor asuransi.

## Business Understanding

### Problem Statements
Berdasarkan pada latar belakang di atas, permasalahan yang dapat diselesaikan pada proyek ini adalah sebagai berikut:

-   Bagaimana cara melakukan pra-pemrosesan data biaya asuransi ksehatan sehingga dapat digunakan untuk membuat model yang baik?
-   Bagaimana cara membangun model _machine learning_ untuk memprediksi data biaya asuransi kesehatan di masa mendatang?

### Goals
Tujuan dibuatnya proyek ini adalah sebagai berikut:

-  Tujuan umum dari analisis dataset ini adalah untuk memahami fitur mana yang paling memengaruhi biaya asuransi. 
-  Membandingkan performa beberapa model regresi. 
-   Menentukan model terbaik untuk prediksi biaya asuransi (charges).

### Solution statements
Solusi yang dapat dilakukan untuk memenuhi tujuan dari proyek ini di antaranya:

- **Pra-pemrosesan Data**. Pada pra-pemrosesan data dapat dilakukan beberapa tahapan, antara lain:
  
    -   Konversi fitur dari kategorikal ke numerik.
    -   One-hot encoding variable kategorikal.
    -   Menangai outlier dengan metode IQR.
    -   Split data (pembagian data).
    -   Melakukan transformasi data.
    -   Standarisasi data.
  
- **Pembangunan Model**. Pada pembangunan model terdapat beberapa algoritma yang digunakan, antara lain:
    - **K-Nearest Neighbor** 
    - **Random Forest**
    - **Boosting Algorithm**
    
## Data Understanding
- **Informasi Dataset**
  <br> Dataset yang digunakan pada proyek ini yaitu dataset lengkap dengan riwayat harga **insurance Health**, informasi lebih lanjut mengenai dataset tersebut dapat lihat pada tabel berikut:

  | Jenis                   | Keterangan                                                                              |
  | ----------------------- | --------------------------------------------------------------------------------------- |
  | Sumber                  | Dataset: [Kaggle](https://www.kaggle.com/datasets/rahulvyasm/medical-insurance-cost-prediction) |
  | Dataset Owner           | rahulvyasm                                                                         |
  | Lisensi                 | MIT                                                                   |
  | Kategori                | Health, Finance                                                                     |
  | Usability               | 10.00                                                                                      |
  | Jenis dan Ukuran Berkas | CSV (115.14 kB)                                                                           |

  Setelah melakukan observasi pada dataset yang diunduh melalui _link_ Kaggle yaitu `insurance.csv`, didapatkan informasi sebagai berikut :
  
  - Terdapat  1338 baris (_records_ atau jumlah pengamatan) yang berisi data yang menggambarkan informasi demografis, gaya hidup dan biaya asuransi kesehatan individu.
  - Terdapat 7 kolom yaitu `age, sex, bmi, children, smoker, region, charges` yang merupakan variabel - variabel pada data.
  - Dari kolom-kolom tersebut terdapat 4 kolom numerik yaitu `age, bmi, children, dan charges` kolom ini berguna untuk analisis kuantitatif.
  - Terdapat 3 kolom dengan tipe kategorikal yaitu `sex, smooker, dan region` kolom ini berguna untuk analisis kualitatif atau sebagai fitur dalam algoritma machine learning.
  - Tidak terdapat _missing value_ pada dataset. 
  
  Untuk penjelasan mengenai variabel-variabel pada dataset dapat dilihat pada poin-poin berikut ini:

    *   age(int64) : Usia individu dalam tahun. Kolom ini bersifat numerik dan menunjukkan faktor usia sebagai salah satu determinan biaya asuransi
    *   sex(object) : Jenis kelamin individu, dengan nilai kategori seperti  `male` dan `female`.
    *   bmi (float64) : Indeks Massa Tubuh (Body Mass Index), yang digunakan untuk menilai status berat badan seseorang (ideal, kurang atau obesitas). Kolom ini bertipe numerik
    *   children(int64) : Jumlah anak tanggungan yang dimiliki individu. Kolom ini juga bersifat numerik.
    *   smoker (object) : Status perokok individu, dengan nilai kategori seperti 'yes' (perokok) dan 'no' (bukan perokok). 
    *   region(object) : Wilayah tempat tinggal individu, kolom ini bersifat kategorikal
    *   charges(float64) : Biaya asuransi kesehatan yang dibayarkan oleh individu. 

- **Sebaran atau Distribusi Data pada Setiap Fitur**
  <br> sebelum masuk ke tahap distribusi data, yang harus dipersiapkan yaitu mengecek missing value (nilai hilang)
  <br> Berikut merupakan visualisasi data yang menunjukkan sebaran/distribusi data pada beberapa fitur-fitur numerik (`bmi dan charges`) :
  
  - Mengidentifikasi Missing Value dan Outlier
    <br>
    ![Image](https://drive.google.com/uc?id=1kFsPHiSo6Bo6KEa69tekE1praZMnrCeS)
    <br> Terlihat jika di atas banyak terdapat outlier pada setiap variabel, lalu untuk mengatasinya nantinya penulis akan menerapkan batas bawah dan batas atas menggunakan metode IQR
    
  - Univariate Analysis
    <br>
    ![Image](https://drive.google.com/uc?id=1_fgN1VzW973cFvJJrHPlYT0TIslWx_ls)
    <br> Terlihat bahwa pada grafik tersebut menunjukkan empat fitur pada dataset yaitu `age, bmi, children, charges`. 
    -   Grafik `age` menjukkan distribusi usia bahwa sebagian besar individu berusia muda, dengan pengamatan pada rentang usia sekitar 20-30 tahun.
    -   Grafik `bmi` menjukkan distribusi Indeks Massa Tubuh (BMI). BMI pada dataset ini tersebar cukup lebar, dengan konsentrasi yang jelas disekitar 25-35.
    -   Grafik `children` menunjukkan distribusi jumlah anak yang dimiliki Terlihat bahwa mayoritas memiliki 0 hingga 2 anak, dengan jumlah anak terbanyak adalah 0. Distribusi ini sangat skewed dengan banyak individu yang tidak memiliki anak, dan jumlah individu dengan jumlah 3 atau lebih anak sangat sedikit.
    -   Grafik 'charges' menunjukkan distribusi biaya asuransi. Terlihat bahwa sebagian besar individu membayar biaya asuransi yang relatif rendah, dengan beberapa puncak pada kisaran biaya 5.000 hingga 15.000. Namun, distribusi ini cenderung skewed ke kanan, dengan beberapa individu yang memebayar biaya asuransi yang jauh lebih tinggi (outlier) 

  
  - Multivariate Analysis
    <br>
    ![Image](https://drive.google.com/uc?id=1nlTh-0BrJ3tn4ia6majKZq2yLPmYpirT)
    <br> Terlihat bahwa pada grafik tersebut  sebagian besar fitur yang memiliki hubungan langsung terhadap charges (seperti age dan bmi) memang menunjukkan tren positif yang signifikan, dimana peningkatan pada sumbu X diikuti dengan peningkatan pada sumbu Y. Namun untuk fitur lainnya seperti children, hubungan tersebut tidak selalu terlihat atau membentuk pola garis lurus yang jelas.
    <br>
    ![Image](https://drive.google.com/uc?id=1SmkGSAcijHI6QdpjUlb-1mevxnovANGV)
    <br> Grafik diatas menunjukkan korelasi antar fitur dalam dataset yang berkaitan dengan biaya asuransi kesehatan. Di dalam grafik tersebut, penulis dapat melihat hubungan antar berbagai variabel yang ada, dengan fokus pada fitur yang memengaruhi biaya asuransi (charges).
    Kolom dan Baris: Setiap kolom dan baris mewakili variabel dalam dataset. Variabel-variabel tersebut meliputi:

    - age: Usia
    - sex: Jenis kelamin (binary: male = 0, female = 1)
    - bmi: Indeks Massa Tubuh
    - children: Jumlah anak
    - smoker: Status merokok (binary: yes = 1, no = 0)
    - region: Wilayah tempat tinggal
    - charges: Biaya asuransi
   <br> Nilai Korelasi:

    Nilai korelasi antara dua variabel menunjukkan kekuatan dan arah hubungan antara keduanya. Nilai berkisar antara -1 (hubungan negatif sempurna) hingga 1 (hubungan positif sempurna), dengan 0 menunjukkan tidak ada hubungan.

    <br> charges (biaya asuransi):

    Korelasi kuat dengan smoker (0.79) menunjukkan bahwa status merokok memiliki pengaruh besar terhadap biaya asuransi. Orang yang merokok memiliki biaya asuransi yang lebih tinggi.
    Korelasi moderat dengan age (0.30) dan bmi (0.20) menunjukkan bahwa semakin tua usia dan semakin tinggi BMI, semakin tinggi kemungkinan biaya asuransi.

    <br>smoker:

    Korelasi sangat tinggi dengan charges (0.79), menunjukkan bahwa faktor apakah seseorang merokok atau tidak adalah indikator yang kuat untuk biaya asuransi.
    Korelasi kecil dengan bmi (0.16) dan age (0.04), menunjukkan bahwa walaupun ada hubungan kecil dengan BMI dan usia, pengaruhnya lebih kecil dibandingkan dengan pengaruhnya terhadap biaya asuransi.

    <br>age dan bmi:

    Korelasi rendah antara age dan bmi (0.11) menunjukkan bahwa usia tidak terlalu berkaitan langsung dengan BMI.
    Korelasi kecil antara age dan charges (0.30) menunjukkan bahwa usia juga mempengaruhi biaya asuransi, tetapi tidak sekuat pengaruh merokok.

    <br>Variabel lain:

    sex, children, dan region menunjukkan korelasi yang sangat kecil dengan biaya asuransi dan fitur lainnya.

    <br>Grafik tersebut memberikan wawasan bahwa merokok adalah salah satu faktor yang paling mempengaruhi biaya asuransi. Selain itu, usia dan BMI juga mempengaruhi biaya asuransi, meskipun pengaruhnya lebih kecil dibandingkan dengan merokok. Variabel lainnya, seperti jenis kelamin, jumlah anak, dan wilayah, memiliki korelasi yang sangat kecil dengan biaya asuransi.
  
## Data Preparation
Berikut ini merupakan tahapan-tahapan dalam melakukan pra-pemrosesan data:
- **Konversi fitur dari kategorikal ke numerik**
  <br>
  ```python
      categ_to_num = {'sex': {'male' : 0 , 'female' : 1},
              'smoker': {'no': 0 , 'yes' : 1},
              'region' : {'northwest':0,
      'northeast':1,'southeast':2,'southwest':3}
             }
      dataset_path.replace(categ_to_num, inplace = True)

<br> sebelum masuk ke tahapan selanjutnya yaitu ada tahapan konversi fitur kategorikal ke numerik. Kode diatas untuk menggantikan nilai kategorikal dikolo [sex, smoker, dan region] dengan nilai numerik, agar data siap digunakan oleh model. inplace=True memastikan bahwa perubahan dilakukan langsung pada dataset tanpa perlu membuat salinan baru. 
<br> Konversi ini merupakan bagian penting dari preprocessing data yang memungknkan model machine learning untuk memahami, memproses, dan menganalisis data secara efektif.

- **One-Hot Encoding untuk variabel kategorikal**
<br>
  ```python
  categorical_columns = ["sex", "smoker", "region"]
  dataset_encoded = pd.get_dummies(dataset_path, drop_first=True)
  print(dataset_encoded.head())
  
<br> Kode diatas berfungsi untuk melakukan One-Hot Encoding pada variabel kategorikal dalam dataset.

Kode diatas mengubah dataset sehingga semua variabel kategorikal dapat digunakan dalam algoritma machine learning yang membutuhkan data numerik. Dengan drop_first=True, dimensi data yang dihasilkan lebih kecil dan menghindari redundansi.

- **Menangani Outlier dengan metode IQR**
<br>
```python
    # Hanya pilih kolom numerik
    numeric_data = dataset_path.select_dtypes(include=['number'])
    # Menghitung Q1, Q3, dan IQR pada data numerik
    Q1 = numeric_data.quantile(0.25)
    Q3 = numeric_data.quantile(0.75)
    IQR = Q3 - Q1
    # Menyaring outlier berdasarkan IQR
    dataset_copy_clean = dataset_path[~((numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))).any(axis=1)]
    # Menampilkan dimensi dataset asli
    print("Dimensi asli dataset:", dataset_path.shape)
    # Menampilkan dimensi dataset setelah pembersihan outlier
    print("Dimensi dataset setelah menghapus outlier:", dataset_copy_clean.shape)

<br>kode diatas bertujuan untuk membersihkan outlier dari dataset. Fokus pada     fitur yang memiliki nilai angka, karena hanya fitur numerik yang relevan untuk deteksi outlier berbasis IQR. Tujuan dari kode tersebut menghapus baris yang mengandung nilai outlier pada kolom numerik, agar model prediksi tidak terpengaruh oleh nilai ekstrem.

  Dengan hasil bahwa dataset yang lebih bersih, dengan ukuran lebih kecil karena outliers telah dihapus.

- **Split Data**
**Melakukan pembagian dataset**
  <br> Untuk mengetahui kinerja model ketika dihadapkan pada data yang belum pernah dilihat sebelumnya, maka perlu dilakukan pembagian dataset. Pada proyek ini dataset dibagi menjadi data latih dan data uji dengan rasio 70% untuk data latih dan 30% untuk data uji. Data latih merupakan data yang akan penulis latih untuk membangun model _machine learning_, sedangkan data uji merupakan data yang belum pernah dilihat oleh model dan digunakan untuk melihat kinerja atau performa dari model yang dilatih.  Pembagian dataset dilakukan dengan modul [train_test_split](https://scikit-learn.org/0.24/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) dari scikit-learn. Setelah melakukan pembagian dataset, didapatkan jumlah sample pada data latih yaitu 1940 sampel dan jumlah sample pada data uji yaitu 832 sampel dari total jumlah sample pada dataset yaitu 2772 sampel.
    
- **Melakukan transformasi data**
  <br> Data transformasi adalah proses mengubah data mentah menjadi bentuk yang lebih sesuai untuk analisis atau pelatihan model. Tujuannya adalah untuk memastikan bahwa data dalam format yang optimal dan relevan untuk digunakan oleh algoritma machine learning.

- **Standardisasi data**
  <br> Standardisasi merupakan teknik transformasi yang paling umum digunakan dalam tahap data _preparation_. Standardisasi membantu untuk membuat semua fitur numerik berada dalam skala data yang sama dan membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma. Pada proyek ini, standardisasi data dilakukan dengan menerapkan teknik [StandarScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) dari library Scikitlearn. StandardScaler melakukan proses standardisasi fitur dengan mengurangkan mean (nilai rata-rata) kemudian membaginya dengan standard deviasi untuk menggeser distribusi.  StandardScaler menghasilkan distribusi dengan standard deviasi sama dengan 1 dan mean sama dengan 0. Sekitar 68% dari nilai akan berada di antara -1 dan 1.

    ```python X_train_transformed type: <class 'numpy.ndarray'>
      X_train_transformed type: <class 'numpy.ndarray'>
      X_train_transformed shape: (1940, 9)
      y_train type: <class 'numpy.ndarray'>
      y_train shape: (1940,)

  Kode tersebut bertujuan untuk memastikan bahwa X_train_transformed memiliki tipe data yang sesuai dengan kebutuhan proses berikutnya, seperti pelatihan model atau evaluasi.
  `X_train_transformed dan y_train` telah diproses dengan benar dan siap untuk digunakan dalam model machine learning.

  Dimensi data antara fitur `(X_train_transformed) dan target (y_train)` konsisten, yaitu 1940 sample. Ini penting agar model dapat belajar tanpa error terkait ketidaksesuaian dimensi.
  
## Modeling
Pada proyek ini, Proses modeling dalam proyek ini menggunakan 3 algoritma _machine learning_ yaitu `K-Nearest Neighbor`, `Random Forest` dan `Boosting Algorithm` kemudian membandingkan performanya.

- **K-Nearest Neighbor**
  <br> KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat (dengan k adalah sebuah angka positif). Pada tahap ini pembuatan model dilakukan dengan menggunakan modul [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) dari library Scikitlearn dengan nilai k = 10 dan metric Euclidean yang artinya, pada model ini akan membandingkan jarak satu sampel data ke 10 sampel data tetangganya yang terdekat, agar hasil persamaan regresi yang dihasilkannya nantinya akan lebih halus, tahapan itu akan dilakukan berulang-ulang hingga mendapatkan hasil persamaan regresi dengan nilai yang maksimal. Kemudian proses selanjutnya melakukan prediksi menggunakan data uji dan melakukan pengujian.
  -   Kelebihan:
      -   Algoritma KNN merupakan algoritma yang sederhana dan mudah untuk diimplementasikan.
      -   Dapat di implementasikan pada beberapa kasus seperti klasifikasi, regresi dan pencarian.
  -   Kekurangan:
      -   Algoritma KNN menjadi lebih lambat secara signifikan seiring meningkatnya jumlah sampel dan/atau variabel independen.
      -   Rentan terhadap outlier.
      -   Performa menurun pada dataset besar karena kompleksitas komputasi (menghitung jarak untuk semua tetangga).

  <br>Parameter : n_neighbors=10: Jumlah tetangga terdekat yang digunakan untuk membuat prediksi. Dalam hal ini, model akan menggunakan rata-rata nilai target dari 10 data tetangga terdekat untuk memprediksi.

  <br>Tujuan: KNN adalah algoritma non-parametrik yang sederhana namun efektif untuk regresi. Model ini tidak membuat asumsi tentang hubungan antara fitur dan target.

- **Random Forest**
  <br> Algoritma ini disusun dari banyak algoritma pohon (decision tree) yang pembagian data dan fiturnya dipilih secara acak. Pembuatan model dilakukan dengan menggunakan modul [RandomForestClassifier](https://scikitlearn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) dari library Scikitlearn. terdapat parameter yang harus digunakan agar hasil dari pembuatan model dapat maksimal. Parameter pertama ialah parameter `n_estimator` yang merupakan jumlah _trees_ (pohon) di _forest_. Di proyek ini penulis melakukan _set_ nilai `n_estimator` sebesar 100 _trees_. Selanjutnya ialah parameter `max_depth` yang merupakan kedalaman atau panjang pohon. Itu merupakan ukuran seberapa banyak pohon dapat membelah (_splitting_) untuk membagi setiap _node_ ke dalam jumlah pengamatan yang di inginkan, di proyek ini penulis melakukan set nilai `max_depth` sebesar 10 _split_. Parameter `random_state` digunakan untuk mengontrol _random number generator_ yang digunakan. Di proyek ini penulis melakukan _set_ nilai pada parameter `random_state` sebesar 123. Kemudian proses selanjutnya melakukan prediksi menggunakan data uji dan melakukan pengujian.
  -   Kelebihan :
      -   Algoritma Random Forest merupakan algoritma dengan pembelajaran paling akurat yang tersedia. Untuk banyak kumpulan data, algoritma ini menghasilkan pengklasifikasi yang sangat akurat.
      -   Berjalan secara efisien pada data besar.
      -   Dapat menangani ribuan variabel input tanpa penghapusan variabel.
      -   Memberikan perkiraan variabel apa yang penting dalam klasifikasi.
      -   Memiliki metode yang efektif untuk memperkirakan data yang hilang dan menjaga akurasi ketika sebagian besar data hilang.
  -   Kekurangan :
      -   Algoritma Random Forest overfiting untuk beberapa kumpulan data dengan tugas klasifikasi/regresi yang _bising/noise_.
      -   Untuk data yang menyertakan variabel kategorik dengan jumlah level yang berbeda, Random Forest menjadi bias dalam mendukung atribut dengan level yang lebih banyak. Oleh karena itu, skor kepentingan variabel dari Random Forest tidak dapat diandalkan untuk jenis data ini.
  <br> Parameter : 
  random_state=123: Menetapkan nilai acak untuk memastikan hasil yang dapat diulang. Ini membantu menjaga konsistensi dalam eksperimen.
  n_estimators=100: Jumlah pohon keputusan (trees) yang akan digunakan dalam hutan acak. Semakin banyak pohon, biasanya semakin baik model, tetapi juga membutuhkan lebih banyak waktu komputasi.
  max_depth=10: Batasan kedalaman pohon keputusan. Menentukan kedalaman maksimum setiap pohon untuk menghindari overfitting.
  <br> Random Forest adalah model ensemble yang terdiri dari banyak pohon keputusan (decision trees) yang dilatih pada subset data dan fitur yang berbeda. Setiap pohon menghasilkan prediksi, dan hasil akhir adalah rata-rata prediksi dari semua pohon.
  Model ini lebih tahan terhadap overfitting daripada pohon keputusan tunggal.

- **Boosting Algorithm**
  <br> Algoritma ini bekerja dengan membangun model dari data latih. Kemudian ia membuat model kedua yang bertugas memperbaiki kesalahan dari model pertama. Model ditambahkan sampai data latih terprediksi dengan baik atau telah mencapai jumlah maksimum model untuk ditambahkan. Pada tahap ini pembuatan model dilakukan dengan menggunakan modul [Boosting Alghoritm](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) dari library Scikitlearn. Penulis meenggunakan metode adaptive boosting yaitu AdaBoost. Parameter yang penulis gunakan terdapat 2 parameter, yaitu parameter `learning_rate` yang merupakan bobot yang diterapkan pada setiap regressor di masing-masing proses iterasi boosting, dan parameter `random_state` yang digunakan untuk mengontrol random number generator yang digunakan. Kemudian proses selanjutnya melakukan prediksi menggunakan data uji dan melakukan pengujian.
  -   Kelebihan :
      -   Algoritma Boosting dapat mengurangi bias pada data.
      -   Prosedur Boosting cukup sederhana.
      -   Algoritma ini sangat powerful dalam meningkatkan akurasi prediksi.
      -   Algoritma boosting sering mengungguli model yang lebih sederhana seperti logistic regression dan random forest.
  -   Kekurangan :
      -   AdaBoost sangat dipengaruhi oleh outlier.

  Dapat disimpulkan model terbaik yang digunakan untuk dataset ini ialah model Random Forest di mana Random Forest memiliki nilai error terkecil dan nilai akurasi yang tinggi ketimbang kedua model lainnya(cek pada bagian Evaluasi)
 <br> Parameter :
 `n_estimators=100`: Menentukan jumlah estimator atau model dasar (weak learners) yang akan digunakan dalam proses boosting. Setiap model dasar pada dasarnya adalah pohon keputusan kecil yang akan dipelajari secara bertahap untuk meningkatkan kinerja model.
  `random_state=123`: Menetapkan nilai acak untuk memastikan bahwa hasil eksperimen dapat direproduksi (hasil yang konsisten setiap kali dijalankan).
  <br> AdaBoost (Adaptive Boosting) adalah algoritma ensemble yang meningkatkan kinerja model dengan fokus pada kesalahan yang dibuat oleh model sebelumnya. Model yang lebih lemah (misalnya pohon keputusan kecil) digabungkan untuk membentuk model yang lebih kuat, dengan setiap model baru mencoba untuk memperbaiki kesalahan yang terjadi pada model sebelumnya.

## Evaluation
Pada proyek ini, metrik evaluasi yang digunakan untuk mengukur kinerja model yaitu menggunakan metrik **akurasi** dan **MSE**. Akurasi di sini merupakan tingkat keakuratan data prediksi yang didasarkan dari data latih pada model, tingkat akurasi tertinggi ialah pada model Random Forest sebesar 94% dan ini menunjukkan bahwasannya Random Forest merupakan model terbaik dari kedua model lainnya dalam memprediksi biaya **Asuransi Kesehatan** di masa mendatang. MSE sendiri merupakan _Mean Squared Error_ yang menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi. MSE didefinisikan dalam persamaan berikut: <br>
<image src='https://www.pythonpool.com/wp-content/uploads/2021/08/20210812_200937_0000-1024x270.png' width= 500/>
<br> Keterangan:
- N = jumlah dataset
- yi = nilai sebenarnya
- yi^ = nilai prediksi

Sebelum menggunakan metrik MSE, harus dilakukan scaling fitur numerik terlebih dahulu pada data uji untuk menghindari kebocoran data. Setelah melakukan evaluasi berdasarkan metrik MSE dan tingkat akurasi prediksi pada model, penulis mencoba memprediksi harga untuk 30 hari ke depan dengan model KNN dan hasilnya cukup memuaskan karena nilainya tidak jauh berbeda dengan data sebelumnya.

Berikut ini perbandingan grafik metrik MSE pada ketiga model:
<br>
![Image](https://drive.google.com/uc?export=view&id=1BMckM-bP5zNgtD-m7D6--WI2tdUTBlHQ)

<br> Selain akurasi untuk menentukan model terbaik dapat dilihat juga berdasarkan tingkat eror pada grafik di atas, semakin kecil tingkat eror maka semakin baik model tersebut memprediksi data. jika dilihat dari gambar di atas Random Forest lah model yang memiliki tingkat eror terendah dibandingkan dengan model lainnya.

- **Dampak dari Model yang Dievaluasi terhadap Business Understanding**

<br> Hasil evaluasi yang menunjukkan akurasi tinggi, khususnya dari Random Forest yang mencapai 94%, memiliki dampak signifikan pada pemahaman bisnis dalam konteks prediksi biaya asuransi kesehatan.

<br> Berikut dampak utama yang di dapatkan :
1. Penetapan Premi Asuransi yang Lebih Akurat:

Dengan menggunakan model yang akurat, perusahaan asuransi dapat menetapkan premi yang lebih tepat untuk individu berdasarkan prediksi biaya yang lebih mendekati biaya asuransi yang sebenarnya. Ini memungkinkan perusahaan untuk menawarkan harga yang lebih adil dan lebih kompetitif.

2. Segmentasi Pasien:

Memahami faktor-faktor yang memengaruhi biaya asuransi, seperti usia, BMI, status perokok, dan jumlah anak, memungkinkan perusahaan asuransi untuk mengelompokkan pelanggan mereka ke dalam segmen yang lebih sesuai dengan profil risiko mereka. Hal ini dapat membantu dalam perencanaan dan alokasi sumber daya yang lebih efisien.

3. Peningkatan Profitabilitas:

Dengan prediksi yang lebih baik tentang biaya, perusahaan dapat menghindari kerugian finansial dari ketidakakuratan dalam perhitungan premi. Model yang tepat memungkinkan prediksi yang lebih realistis, meningkatkan profitabilitas jangka panjang perusahaan.

4. Pengelolaan Risiko:

Penggunaan model seperti Random Forest yang dapat menangani data yang lebih kompleks dan melakukan feature importance memungkinkan perusahaan asuransi untuk lebih baik dalam mengidentifikasi risiko tinggi (seperti individu dengan status perokok atau BMI tinggi) dan menyesuaikan harga premi untuk mencerminkan tingkat risiko yang lebih tinggi.

<br> **Berdasarkan analisis dan evaluasi model yang dilakukan** 

1. Memahami Faktor yang Mempengaruhi Biaya Asuransi: 
Tujuan untuk memahami faktor-faktor yang mempengaruhi biaya asuransi telah tercapai. Fitur seperti age, bmi, smoker, dan children memberikan kontribusi signifikan terhadap prediksi biaya asuransi.
2. Membandingkan Performa Beberapa Model Regresi: 
Dengan membandingkan beberapa model regresi, penulis dapat menyimpulkan bahwa Random Forest memberikan performa terbaik di antara KNN dan AdaBoost, dengan akurasi mencapai 94%.
3. Menentukan Model Terbaik untuk Prediksi Biaya Asuransi (Charges): 
Random Forest adalah model terbaik untuk memprediksi biaya asuransi kesehatan berdasarkan dataset ini, karena akurasi yang sangat tinggi dan kinerja stabil pada data uji.

<br> **Kesimpulan**
Dengan membandingkan performa model dan memilih Random Forest sebagai model terbaik, penulis berhasil mencapai tujuan untuk memprediksi biaya asuransi kesehatan dengan akurasi tinggi. Melalui tahap pra-pemrosesan, feature engineering, dan pemilihan model yang tepat, penulis dapat membangun model prediktif yang kuat dan dapat diandalkan untuk peramalan biaya asuransi di masa depan.
