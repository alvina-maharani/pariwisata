import streamlit as st
import pandas as pd
import nltk
from nltk import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
import joblib


Home, Learn, Proses, Model, Implementasi = st.tabs(['Home', 'Learn Data', 'Preprocessing', 'Model', 'Implementasi'])

with Home :
   
    st.title="Klasifikasi Pariwisata"
    st.write("""
        Nama   : Alvina Maharani\n
        NIM    : 200411100029 \n
        Kelas  : Informatika Pariwisaata B
        """)
    st.header('KLASIFIKASI PARIWISATA KOTA TUBAN BERDASARKAN KATEGORINYA')

with Learn :

    st.write("Dataset yang digunakan adalah data keterangan tempat wisata dari website pariwisata Kota Tuban ")
    st.write("Total datanya adalah 44 dengan 3 kategori")
    df = pd.read_excel("https://github.com/alvina-maharani/pariwisata/blob/main/data_pariwisata.xlsx")
    st.dataframe(df)

with Proses :

    st.write("Menghapus tanda baca")
    def labels(Kategori):
        if Kategori == "Alam":
            return "1"
        elif Kategori == "Kuliner":
            return "2"
        else:
            return "3"

    df["LABEL"] = df["Kategori"].apply(labels)
    df
    import re
    def cleaning(Penjelasan):
        Penjelasan = re.sub(r'@[A-Za-a0-9]+',' ',Penjelasan)
        Penjelasan = re.sub(r'#[A-Za-z0-9]+',' ',Penjelasan)
        Penjelasan = re.sub(r"http\S+",' ',Penjelasan)
        Penjelasan = re.sub(r'[0-9]+',' ',Penjelasan)
        Penjelasan = re.sub(r"[-()\"#/@;:<>{}'+=~|.!?,_]", " ", Penjelasan)
        Penjelasan = Penjelasan.strip(' ')
        Penjelasan = Penjelasan.strip("\n")
        return Penjelasan

    df["Cleaning"]= df["Deskripsi"].apply(cleaning)
    st.dataframe(df)

    st.write("Case Folding")
    df['casefolding'] = [entry.lower() for entry in df['Cleaning']]

    st.dataframe(df)

    st.write("Tokenisasi")
    def word_tokenize_wrapper(text):
        return word_tokenize(text)

    df['Tokenizing'] = df['casefolding'].str.lower().apply(word_tokenize_wrapper)
    st.dataframe(df)

    st.write("Stopword Removal")
    daftar_stopword = stopwords.words('indonesian')
    # Masukan Kata dalam Stopwors Secara Manula
    # Tambahakan Data Stopwords Manual
    daftar_stopword.extend(["nya", "tuh", "kamu","banget"])
    daftar_stopword = set(daftar_stopword)

    def stopwordText(words):
        return [word for word in words if word not in daftar_stopword]
    # def datakosong(words):
    #   return [word for word in words if word ]

    df['Stopword Removal'] = df['Tokenizing'].apply(stopwordText)
    st.dataframe(df)

    st.write("Normalisasi")
    def convertToSlangword(penjelasan):
        kamusSlang = eval(open("D:\semester 6\Informatika pariwisata\pariwisata\slangwords.txt").read())
        pattern = re.compile(r'\b( ' + '|'.join (kamusSlang.keys())+r')\b')
        content = []
        for kata in penjelasan:
            filterSlang = pattern.sub(lambda x: kamusSlang[x.group()],kata)
            content.append(filterSlang.lower())
        penjelasan = content
        return penjelasan

    df['Normalisasi'] = df['Stopword Removal'].apply(convertToSlangword)
    st.dataframe(df)

    st.write("Stemming")
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    def stemmed_wrapper(term):
        return stemmer.stem(term)

    term_dict = {}

    for document in df['Stopword Removal']:
        for term in document:
            if term not in term_dict:
                term_dict[term] = ' '

    for term in term_dict:
        term_dict[term] = stemmed_wrapper(term)
        print(term,":" ,term_dict[term])
        
    def stemmingText(document):
        return [term_dict[term] for term in document]

    df['Stemming'] = df['Stopword Removal'].swifter.apply(stemmingText)
    st.dataframe(df)

    st.write("clean")
    def remove_punct(text):
        text = " ".join([char for char in text if char not in string.punctuation])
        return text
    df["Clean"] = df["Stemming"].apply(lambda x: remove_punct(x))
    st.dataframe(df)

    st.write("TF-IDF")

    vectorizer = TfidfVectorizer()
    # bobot = vectorizer.fit_transform(df["Normalisasi"])
    # st.text(bobot)
    # df_bobot = pd.DataFrame(bobot.todense().T,
    #                     index =vectorizer.get_feature_names_out(),
    #                     columns=[f'D{i+1}' for i in range(len(df["Stemming"]))])
    # st.dataframe (df_bobot)

with Model :
    X = df['Clean']
    Y = df['LABEL']

    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.4)

    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)

    cv = CountVectorizer()
    cv_matrix = cv.fit_transform(X)

    SVM = svm.SVC(kernel='linear') 
    SVM.fit(x_train,y_train)

    y_pred = SVM.predict(x_test)

    st.text(confusion_matrix(y_test, y_pred))
    st.text(classification_report(y_test, y_pred))

    #menyimpan hasil

    joblib.dump(SVM, "https://github.com/alvina-maharani/pariwisata/blob/main/klasifikasiuhuy")
    joblib_models = joblib.load("https://github.com/alvina-maharani/pariwisata/blob/main/klasifikasiuhuy")

with Implementasi :
    
    def prediksi(text):
        tfidf_vektor = vectorizer.transform([text])
        pred = joblib_models.predict(tfidf_vektor)
        if pred == 1:
            hate = "Alam"
        elif pred == 2:
            hate = "Kuliner"
        elif pred == 3:
            hate = "Religi"
        else:
            hate = "Error"
        return hate
    

    masukkan_kalimat = st.text_input("Masukan Deskripsi :")
    masukkan_kalimat = cleaning(masukkan_kalimat)
    masukkan_kalimat = masukkan_kalimat.lower()
    masukkan_kalimat = word_tokenize_wrapper(masukkan_kalimat)
    masukkan_kalimat = convertToSlangword(masukkan_kalimat)
    masukkan_kalimat = stopwordText(masukkan_kalimat)
    masukkan_kalimat = " ".join(masukkan_kalimat)
    # masukkan_kalimat = stemmer.stem(masukkan_kalimat)

    cek = st.button("cek kategori")
    if cek:
        st.text(masukkan_kalimat)
        st.text(f"Hasil Klasifikasi :{prediksi(masukkan_kalimat)}")

