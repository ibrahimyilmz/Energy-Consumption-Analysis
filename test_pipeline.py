from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.clustering_engine import reduce_and_cluster
from src.data_loader import load_consumption_data
from src.features import build_behavioral_features


DATA_PATH = Path("data/export.csv")
OUTPUT_PATH = Path("data/processed/labeled_customers.csv")


def main() -> None:
    try:
        print("--- 1. Adim: Veri Yukleniyor ve Donusturuluyor ---")
        if not DATA_PATH.exists():
            raise FileNotFoundError(
                f"Veri dosyasi bulunamadi: {DATA_PATH}. Dosyayi proje kokune kopyalayin."
            )

        # Veri setindeki guc kolonu "valeur" olarak geliyor.
        df = load_consumption_data(DATA_PATH, power_col="valeur")
        print(f"Basarili! Toplam satir sayisi: {len(df)}")
        print(df[["timestamp", "power_kw", "energy_kwh"]].head())

        print("\n--- 2. Adim: Davranissal Ozellikler Cikariliyor ---")
        features = build_behavioral_features(df)
        print(f"Basarili! Musteri sayisi: {len(features)}")
        print(features[["customer_id", "occupancy_rate", "fft_daily_amp"]].head())

        print("\n--- 3 & 4. Adim: PCA ve Kumeleme (RS/RP Etiketleme) ---")
        final_df = reduce_and_cluster(features, n_components=2)
        print("Basarili! Etiketleme tamamlandi.")

        print("\n--- Sonuc Ozeti ---")
        print(final_df["label"].value_counts())
        print("\nKume Bazli Ortalama Doluluk Orani:")
        print(final_df.groupby("label")["occupancy_rate"].mean())

        print("\n--- Basari Kriteri Kontrolleri ---")
        ratio = (df["energy_kwh"] / (df["power_kw"] + 1e-12)).head(5)
        print("energy_kwh / power_kw (ilk 5 satir, 0.5 olmali):")
        print(ratio)
        print("PCA kolonlari mevcut mu?", {"pca_1", "pca_2"}.issubset(final_df.columns))

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(OUTPUT_PATH, index=False)
        print(f"\nIslem tamamlandi: '{OUTPUT_PATH}' olusturuldu.")

    except Exception as exc:
        print(f"\nHATA: {exc}")


if __name__ == "__main__":
    main()
