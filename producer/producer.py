import time
import json
import random
from datetime import datetime
from kafka import KafkaProducer
import pandas as pd

print("Sistem başlatılıyor, Lütfen bekleyin...")
time.sleep(20)

producer = KafkaProducer(
    bootstrap_servers='kafka:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

csv_path = '/data/dataset.csv'

try:
    print(f"Veriler okunuyor: {csv_path}")
    df = pd.read_csv(csv_path).fillna(0)
except FileNotFoundError:
    print("Data klasöründe dataset.csv bulunamadı! Lütfen dosyayı ekleyin.")

print("Kafka'ya veri aktarımı başlıyor...")

msg_count = 0

for index, row in df.iterrows():
    message = {
        "timestamp": str(datetime.now()),
        "user_id": random.randint(1000, 9999), 
        "event_type": random.choice(["click", "view", "purchase"]),
        "related_id": random.randint(100, 999),
        "data": row.to_dict()
    }
    
    producer.send('project-topic', message)
    msg_count += 1
    
    if msg_count % 50 == 0:
        print(f"{msg_count} adet mesaj Kafka'ya başarıyla iletildi.")
    
    sleep_duration = random.uniform(0.3, 0.7)
    time.sleep(sleep_duration)

print(f"İşlem tamamlandı. Toplam {msg_count} mesaj Kafka'ya gönderildi.")