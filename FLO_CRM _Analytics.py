###################################
   ##     FLO RFM ANALİZİ     ##
###################################


import pandas as pd
import datetime as dt

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.2f" % x)
pd.set_option("display.max_rows", 10000)
pd.set_option("display.width", 500)

df_ = pd.read_csv("flo_data_20k.csv") # Veri setini df_ olarak okutuyoruz.
df = df_.copy() # Orjinal veri setinin bozmamak için veri setinin kopyasını alıyoruz.
df.head()

# Veri Setini tanımak için bir fonksiyon oluşturuyoruz.
def check_dataframe(dataframe, head=5):
    print("---------- SHAPE ----------")
    print(dataframe.shape)
    print("---------- TYPES ----------")
    print(dataframe.dtypes)
    print("---------- HEAD ----------")
    print(dataframe.head(head))
    print("---------- TAIL ----------")
    print(dataframe.tail(head))
    print("---------- QUANTILES ----------")
    print(dataframe.quantile([0, 0.05, 0.5, 0.95, 0.99, 1]).T) #

check_dataframe(df)

# Müşterilerin toplam alışveriş sayısı ve toplam harcamalarını içeren 2 yeni değişken oluşturuyoruz.
df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
df.head()

# Tarih belirten değişkenlerin cinsini date olarak değiştirdik.
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)
df.info()

# Alışveriş kanallarına göre müşterilerin sayısını, toplam siperişleri ve toplam harcamaları gösteriyoruz.
df.groupby("order_channel").agg({"master_id": "count",
                                 "order_num_total": "sum",
                                 "customer_value_total": "sum"})


df.sort_values("customer_value_total", ascending=False)[:10] # En fazla kazancı getiren 10 müşteri
df.sort_values("order_num_total", ascending=False)[:10] # En fazla harcamayı yapan 10 müşteri

# Veri setini analize hazır hale getirmek için yaptığımız işlemleri fonksiyonlaştırdık.
def data_prep(dataframe):
    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)
    return dataframe


df["last_order_date"].max() # Son sipariş tarihini öğrendik. 2021-05-30
analysis_date = dt.datetime(2021,6,1) # Son sipariş tarihinden 2 gün sonrasını analiz yaptığımız tarih olarak belirledik.

# RFM analizi yapabilmemiz için gerekli değişkenleri hesaplayıp rfm adında bir dataframe'e atadık.
rfm = pd.DataFrame()
rfm["customer_id"] = df["master_id"] # Müşteri ID'lerini çektik
rfm["recency"] = (analysis_date - df["last_order_date"]).astype("timedelta64[D]") # Recency değerlerinin oluşturduk.
rfm["frequency"] = df["order_num_total"] # Müşterilerin frequency değerlerini oluşturduk.
rfm["monetary"] = df["customer_value_total"] # Müşterilerin monetary değerlerini oluştueduk.
rfm.head()

#RFM skorlarının hesaplanması
rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5,4,3,2,1]) # Recency değerlerinden yola çıkarak receny_score adında yeni bir değişken oluşturuldu.
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]) # frequency_score oluşturuldu
rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5]) # monetary_score oluşturuldu
rfm.head()

# Ayrı ayrı elde ettiğimiz R,F,M skorlarını RF ve RFM skoru olarak yeni değişkenlere atadık.
rfm["RF_SCORE"] = (rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str))
rfm["RFM_SCORE"] = (rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str) + rfm["monetary_score"].astype(str))
rfm.head()

# RFM skorlarına göre müşteri segmentasyonu yapacağımız segmentleri belirledik.
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True) # seg_map'te yer alan segmentlere müşterileri atadık.
rfm.head(10)

# Segmentlerin incelenmesi.
rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])