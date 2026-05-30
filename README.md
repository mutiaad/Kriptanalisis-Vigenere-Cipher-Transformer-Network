# Kriptanalisis Vigenere Cipher Menggunakan Model Transformer

Projek ini merupakan penelitian skripsi yang bertujuan untuk melakukan dekripsi otomatis (kriptanalisis) pada pesan yang dienkripsi menggunakan algoritma klasik **Vigenere Cipher** tanpa mengetahui kunci enkripsinya (_Ciphertext-only attack_), melainkan dengan memanfaatkan pemodelan Deep Learning menggunakan arsitektur **Transformer**.

## Hasil
Setelah melalui beberapa skenario pengujian dan pelatihan arsitektur yang bervariasi (penambahan atau pengurangan jumlah _attention heads_ dan layer), model Transformer terbaik berhasil mencapai tingkat akurasi sebesar **75% - 80%** dalam menebak plainteks dengan tepat berdasarkan cipherteks yang diberikan.

## Teknologi & Library yang Digunakan
* **Bahasa Pemrograman:** Python
* **Framework Deep Learning:** TensorFlow
* **Environment:** Jupyter Notebook / Google Colab

## Referensi & Atribusi
Arsitektur dan sintaks dasar model pada projek ini diadaptasi dan dimodifikasi dari repositori open-source **Pylesson**. Modifikasi dan penyesuaian dilakukan pada bagian manajemen dataset, prapemrosesan data teks kustom, serta tuning struktur layer untuk kebutuhan kriptanalisis ini. Selain itu, proses pembangkitan _plaintexts_, _ciphertexts_, dan kunci juga diadaptasi dan dimodifikasi dari repositori open-source **github.com/ashwiniyer176/Neural-Cryptanalysis**. Modifikasi dilakukan pada bagian pembangkitan kunci dan karakter vigenere cipher yang diubah menjadi huruf besar (A-Z), huruf kecil (a-z), dan angka (0-9).
