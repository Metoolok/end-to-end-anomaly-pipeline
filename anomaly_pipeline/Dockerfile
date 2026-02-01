# 1. Hafif bir Python imajı seçiyoruz (Ubuntu tabanlı)
FROM python:3.12-slim

# 2. Sistem bağımlılıklarını kuruyoruz (Matplotlib ve istatistik kütüphaneleri için)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3. Çalışma dizini oluşturuyoruz
WORKDIR /app

# 4. Bağımlılıkları kopyalayıp kuruyoruz (Cache avantajı için önce bunu yaparız)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Tüm proje dosyalarını kopyalıyoruz
COPY . .

# 6. Grafiklerin ekrana basılması Docker içinde zordur (Display yok),
# Bu yüzden sonuçları kaydetmesi için bir ortam değişkeni atıyoruz.
ENV MPLBACKEND=Agg

# 7. Konteyner çalıştığında başlayacak komut
CMD ["python", "main.py"]