# IMRON-Chatbot Backend (Beta V0.1)

"Mas Imron" adalah asisten virtual cerdas berbasis AI pada platform web IMRON. Sistem ini didesain khusus untuk menerjemahkan pertanyaan berbahasa alami (natural language) dari pengguna menjadi query SQL (MySQL/MariaDB) secara dinamis, mengeksekusinya ke database IoT IMRON, dan mengembalikan jawaban dalam bahasa Indonesia/Sunda secara ringkas dan informatif.

## Arsitektur & Alur Kerja Sistem

Proses penerjemahan pertanyaan hingga pengembalian jawaban melewati beberapa tahapan filter, routing, optimasi caching, dan validasi keamanan.

### Interaksi Chatbot 
Berikut adalah alur interaksi pengguna dengan chatbot IMRON Text-to-SQL dalam query data E-Nose dan juga deteksi objek

![Product Catalog App Screenshot](https://i.postimg.cc/gjGm61xW/Screenshot-2026-07-01-120238.png)

### Diagram Sekuensial Proses Text-to-SQL
Berikut adalah alur perjalanan request saat pengguna mengirimkan pertanyaan ke endpoint `/api/query`:

![Product Catalog App Screenshot](https://i.postimg.cc/JzsxLnP2/Untitled-2026-06-15-1338.png)

## Tech Stack & Dependensi Utama

Detail library dan platform yang digunakan di sisi backend:

*   **Core Language:** Python 3.10+
*   **Web Framework:** Flask 3.1.2 & Flask-CORS 6.0.2
*   **AI/LLM Orchestration:** LlamaIndex 0.14.14
*   **Large Language Model (LLM):** Ollama client dengan base model `masimronlite-bagas:latest` (alternatif: `qwen2.5:3b`)
*   **Embedding Model:** `BAAI/bge-m3` via SentenceTransformers (digunakan oleh LlamaIndex untuk indexing schema)
*   **Database:** MySQL / MariaDB
*   **ORM / Drivers:** SQLAlchemy 2.0.46, PyMySQL 1.1.2, & mysql-connector-python 9.6.0
*   **Logging:** RotatingFileHandler (maksimal 3 file log @ 10MB per file)


## Skema Database & Pemetaan Sensor

Sistem backend ini terhubung ke database MySQL dengan nama default `endpoint`. Di bawah ini adalah struktur tabel yang didekati oleh engine Text-to-SQL:

### 1. Tabel Konfigurasi & Pengguna
*   **`device`**: Menyimpan data alat IoT.
    *   Kolom: `device_id` (PK), `device_name`, `ip_address`, `mac_address`, `type`, `description`.
*   **`user`**: Data kredensial pengguna aplikasi.
    *   Kolom: `user_name` (PK), `password`, `user_group`, `status`.
*   **`user_detail`**: Profil lengkap pengguna.
    *   Kolom: `user_name` (PK/FK), `first_name`, `middle_name`, `last_name`, `country`, `province`, `city`, `address`, `email`, `phone`, `company`, `description`.
*   **`device_user_mapping`**: Relasi kepemilikan alat oleh user.
    *   Kolom: `user_name` (FK), `device_id` (FK), `description`.

### 2. Tabel Transaksi Sensor & Deteksi (IoT Core Data)
*   **`transaction_object_detection`**: Log deteksi objek kamera berbasis Edge.
    *   Kolom: `detection_time` (DATETIME), `type` (VARCHAR - e.g., 'car', 'mahasiswa', 'dosen-staff'), `value` (INT - jumlah objek terdeteksi), `device_id` (VARCHAR).
*   **`transaction_enose`**: Log hasil sampling alat sensor gas elektronik (E-Nose).
    *   Kolom: `device_id` (VARCHAR), `date_time` (DATETIME), `type` (VARCHAR - e.g., 'greentea', 'blacktea'), `data_send` (JSON), `value` (JSON).

#### Penting: Pemetaan Struktur JSON di `transaction_enose`
Model LLM sering kali kesulitan melakukan query langsung pada kolom bertipe data JSON. Oleh karena itu, backend mengimplementasikan parser regex khusus (`post_process_enose_sql`) untuk memperbaiki hasil query LLM:
1.  **Kolom `data_send`**: Menyimpan array object sensor di index yang **tetap**:
    *   `$[0]` = MQ9 | `$[1]` = MQ4 | `$[2]` = TGS2602 | `$[3]` = MQ6 | `$[4]` = MQ5 | `$[5]` = TGS2620 | `$[6]` = MQ138 | `$[7]` = MQ3 | `$[8]` = TGS822
    *   *Koreksi Otomatis:* Query sensor `MQ3` akan diterjemahkan menjadi `CAST(JSON_UNQUOTE(JSON_EXTRACT(data_send, '$[7].MQ3')) AS DECIMAL(10,4))`.
2.  **Kolom `value`**: Menyimpan hasil olah klasifikasi model AI E-Nose.
    *   `Score` = Diambil dari key `Score[0]` (tingkat akurasi).
    *   `Class` = Klasifikasi utama (e.g. 'Baik', 'Cacat Mutu').
    *   `Multiclass` = Klasifikasi kelas spesifik di index `Multiclass[0]` (e.g., 'A', 'B', 'C').
    *   *Koreksi Otomatis:* Backend menyatukan format array (`$[0].Class`) dan format key object (`$.0.Class`) menggunakan `COALESCE` agar data selalu terbaca meskipun struktur JSON di DB bervariasi.

---

## Konfigurasi Environment Variables

Aplikasi membaca konfigurasi dari OS Environment Variables. Jika tidak tersedia, sistem akan menggunakan nilai default berikut:

| Variabel | Deskripsi | Default Value |
| :--- | :--- | :--- |
| `DB_USER` | Username database MySQL | `root` |
| `DB_PASS` | Password database MySQL | `""` (kosong) |
| `DB_HOST` | Host database MySQL | `localhost` |
| `DB_PORT` | Port database MySQL | `3306` |
| `DB_NAME` | Nama database utama | `endpoint` |
| `OLLAMA_BASE_URL` | URL endpoint server Ollama | `http://localhost:11434` |
| `OLLAMA_MODEL` | Model LLM yang digunakan | `masimronlite-bagas:latest` |
| `OLLAMA_TIMEOUT` | Timeout request ke Ollama (detik)| `620` |
| `CACHE_TTL_SECONDS`| Masa aktif cache query SQL (detik) | `600` (10 Menit) |
| `DEFAULT_TIMEZONE` | Timezone server untuk salam Sunda | `Asia/Jakarta` |

## Panduan Instalasi & Menjalankan Aplikasi

### Prerequisites
*   Python 3.10.x terinstall di sistem.
*   Ollama terinstall dan model `masimronlite-bagas:latest` sudah di-pull atau di-create.
*   Database MySQL/MariaDB aktif dengan skema tabel IMRON terisi.

### Langkah-langkah
1.  **Clone / Masuk ke direktori proyek:**
    ```bash
    cd "e:/New Volume D/Project Magang Bagas/Deployment Operation/BE_OLD"
    ```
2.  **Aktifkan Virtual Environment (Venv):**
    *   Di **PowerShell / CMD Windows** (bisa merujuk pada [activate_venv.txt](file:///e:/New Volume D/Project Magang Bagas/Deployment Operation/BE_OLD/activate_venv.txt)):
        ```powershell
        .\venv\Scripts\Activate.ps1
        ```
3.  **Install dependensi proyek:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Jalankan API Server:**
    ```bash
    python api_server.py
    ```
    *Server akan berjalan di port `5006` secara lokal (`http://localhost:5006`).*
5.  **Warm-Up Otomatis:**
    Saat startup, server akan mengirimkan dummy request ke Ollama. Proses ini memuat model ke RAM/VRAM sehingga request pertama dari user tidak mengalami delay cold-start (~30-60 detik).



##  API Endpoints Documentation

### 1. POST `/api/query`
Endpoint utama untuk interaksi chatbot berbasis Text-to-SQL.

*   **URL:** `/api/query`
*   **Method:** `POST`
*   **Headers:** `Content-Type: application/json`
*   **Payload Request:**
    ```json
    {
      "question": "Berapa rata-rata score untuk transaksi enose jenis greentea?"
    }
    ```

*   **Response Sukses (Text-to-SQL):**
    ```json
    {
      "answer": "Rata-rata score untuk transaksi e-nose jenis greentea adalah 82.45.",
      "columns": [
        "avg_score"
      ],
      "query_time_ms": 1420.52,
      "row_count": 1,
      "rows": [
        {
          "avg_score": 82.45
        }
      ],
      "source": "text_to_sql",
      "sql_query": "SELECT AVG(CAST(JSON_UNQUOTE(COALESCE(JSON_EXTRACT(value,'$[0].Score[0]'), JSON_EXTRACT(value,'$.0.Score[0]'))) AS DECIMAL(10,2))) AS avg_score FROM transaction_enose WHERE type = 'greentea'",
      "status": "success",
      "table_used": "transaction_enose"
    }
    ```

*   **Response Sukses (Greeting):**
    ```json
    {
      "answer": "Wilujeng enjing! Abdi Imron. Aya nu tiasa abdi ngabantosan?",
      "query_time_ms": 320.15,
      "source": "greeting",
      "status": "success"
    }
    ```

*   **Response Gagal / Fallback:**
    ```json
    {
      "answer": "Mohon Maaf, saya hanya dapat membantu pertanyaan terkait sistem IMRON, sensor enose, atau data object detection.",
      "source": "fallback",
      "status": "success"
    }
    ```

*   **Response Error (Rate Limit / Security / Parameter Kosong):**
    ```json
    {
      "error": "Rate limit exceeded. Mohon tunggu sebentar.",
      "status": "error"
    }
    ```

### 2. GET `/health`
Mengecek status kesehatan server dan konfigurasi fitur yang aktif.

*   **URL:** `/health`
*   **Method:** `GET`
*   **Response:**
    ```json
    {
      "cache_entries": 3,
      "cache_ttl_seconds": 600,
      "features": ["Text-to-SQL", "SQL-Cache", "Greeting", "Rate-Limiter"],
      "model": "masimronlite-bagas:latest",
      "service": "IMRON Chatbot API",
      "status": "healthy",
      "timestamp": "2026-07-01 12:35:00",
      "version": "3.1"
    }
    ```

### 3. GET `/api/cache-status`
Melihat status penggunaan memori cache SQL yang tersimpan beserta usianya.

*   **URL:** `/api/cache-status`
*   **Method:** `GET`
*   **Response:**
    ```json
    {
      "cache_ttl_seconds": 600,
      "entries": [
        {
          "age_seconds": 45.2,
          "sql_preview": "SELECT COUNT(*) FROM user...",
          "table_used": "general"
        }
      ],
      "status": "ok",
      "timestamp": "2026-07-01 12:35:10",
      "total_entries": 1
    }
    ```

### 4. POST `/api/cache-clear`
Menghapus seluruh cache SQL secara manual.

*   **URL:** `/api/cache-clear`
*   **Method:** `POST`
*   **Response:**
    ```json
    {
      "message": "1 cache entries berhasil dihapus",
      "status": "ok",
      "timestamp": "2026-07-01 12:35:15"
    }
    ```

---

## Fitur Keamanan & Guardrails

Backend ini memiliki perlindungan ganda untuk menghindari injeksi query berbahaya ataupun kebocoran database:
1.  **Sanitisasi Input Pertanyaan (`is_security_sensitive_question`)**:
    Memblokir pertanyaan jika mengandung kata kunci sensitif terkait skema sistem atau manipulasi SQL seperti `union`, `information_schema`, `drop table`, `update`, `delete`, dll.
2.  **Validasi Struktur SQL (`validate_sql_safety`)**:
    Mengecek kode SQL yang dihasilkan oleh LLM sebelum dijalankan ke database.
    *   Hanya memperbolehkan statement `SELECT`.
    *   Memastikan query hanya mengakses tabel yang diizinkan sesuai peruntukan domain (e.g. jika diarahkan ke engine E-Nose, query tidak boleh menyentuh tabel konfigurasi user).
    *   Mencegah eksekusi multi-statement (pemisah titik koma `;` untuk query berantai).
3.  **Rate Limiter**:
    Membatasi request IP klien maksimal 20 request per menit guna mencegah serangan DoS/DDoS pada API dan model lokal.
