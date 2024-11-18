# Zadanie 1: Wczytanie danych i wstępna analiza
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Wczytanie danych
mnist = pd.read_csv('MNIST.csv')

# Dodanie nazw kolumn
columns = ['class'] + [f'pixel{i}' for i in range(1, 785)]
mnist.columns = columns

# Podział danych na zbiór treningowy i testowy
train_data, test_data = train_test_split(mnist, test_size=0.3, random_state=42)

# Wyznaczenie liczby rekordów i cech
print(f'Liczba rekordów w zbiorze treningowym: {train_data.shape[0]}')
print(f'Liczba cech w zbiorze treningowym: {train_data.shape[1] - 1}')  # bez kolumny 'class'
print(f'Liczba rekordów w zbiorze testowym: {test_data.shape[0]}')

# Wyznaczenie rozkładu kategorii
train_distribution = train_data['class'].value_counts(normalize=True) * 100
print("Rozkład kategorii w zbiorze treningowym:")
print(train_distribution)

# Wykres słupkowy rozkładu kategorii
plt.figure(figsize=(10, 5))
train_distribution.plot(kind='bar')
plt.title('Rozkład kategorii w zbiorze treningowym')
plt.xlabel('Klasa')
plt.ylabel('Procent')
plt.show()

# Zadanie 2: Analiza PCA
from sklearn.decomposition import PCA

# Zadanie 2.1: Ustawienie PCA dla wyjaśnionej wariancji 0.90
pca = PCA(n_components=0.90)
pca.fit(train_data.drop(columns=['class']))

# Liczba wymiarów, aby osiągnąć 0.90 wyjaśnionej wariancji
print(f'Liczba wymiarów dla 90% wyjaśnionej wariancji: {pca.n_components_}')

# Zadanie 2.2: Wyjaśniona wariancja dla wymiarów [10, 150]
explained_variances = []
for i in range(10, 151):
    pca_temp = PCA(n_components=i)
    pca_temp.fit(train_data.drop(columns=['class']))
    explained_variances.append(np.sum(pca_temp.explained_variance_ratio_))

plt.figure(figsize=(12, 6))
plt.plot(range(10, 151), explained_variances)
plt.title('Wyjaśniona wariancja w zależności od liczby wymiarów')
plt.xlabel('Liczba wymiarów')
plt.ylabel('Procent wyjaśnionej wariancji')
plt.grid(True)
plt.show()

# Zadanie 3: Redukcja danych do przestrzeni PCA
pca_final = PCA(n_components=pca.n_components_)
train_data_pca = pca_final.fit_transform(train_data.drop(columns=['class']))
test_data_pca = pca_final.transform(test_data.drop(columns=['class']))

# Zadanie 4: Wizualizacja danych
import seaborn as sns

plt.figure(figsize=(8, 8))
sns.scatterplot(x=train_data_pca[:, 0], y=train_data_pca[:, 1], hue=train_data['class'], palette='tab10')
plt.title('Wizualizacja danych skompresowanych PCA')
plt.show()

# Zadanie 5: Trening Decision Tree na danych oryginalnych
from sklearn.tree import DecisionTreeClassifier
import time

clf_dt = DecisionTreeClassifier()
start_time = time.time()
clf_dt.fit(train_data.drop(columns=['class']), train_data['class'])
dt_time_original = time.time() - start_time
print(f'Czas treningu Decision Tree na danych oryginalnych: {dt_time_original:.2f} s')

# Zadanie 6: Trening Decision Tree na danych skompresowanych
start_time = time.time()
clf_dt.fit(train_data_pca, train_data['class'])
dt_time_pca = time.time() - start_time
print(f'Czas treningu Decision Tree na danych skompresowanych: {dt_time_pca:.2f} s')

# Zadanie 7: Trening Logistic Regression na danych oryginalnych
from sklearn.linear_model import LogisticRegression

clf_lr = LogisticRegression(max_iter=1000)
start_time = time.time()
clf_lr.fit(train_data.drop(columns=['class']), train_data['class'])
lr_time_original = time.time() - start_time
print(f'Czas treningu Logistic Regression na danych oryginalnych: {lr_time_original:.2f} s')

# Skuteczność modelu Logistic Regression
score_lr_original = clf_lr.score(test_data.drop(columns=['class']), test_data['class'])
print(f'Skuteczność Logistic Regression na danych oryginalnych: {score_lr_original:.4f}')

# Zadanie 8: Trening Logistic Regression na danych skompresowanych
start_time = time.time()
clf_lr.fit(train_data_pca, train_data['class'])
lr_time_pca = time.time() - start_time
score_lr_pca = clf_lr.score(test_data_pca, test_data['class'])
print(f'Czas treningu Logistic Regression na danych skompresowanych: {lr_time_pca:.2f} s')
print(f'Skuteczność Logistic Regression na danych skompresowanych: {score_lr_pca:.4f}')

# Zadanie 9: Trening kNN na danych oryginalnych i zredukowanych
from sklearn.neighbors import KNeighborsClassifier

clf_knn = KNeighborsClassifier()

# Na danych oryginalnych
start_time = time.time()
clf_knn.fit(train_data.drop(columns=['class']), train_data['class'])
knn_time_original = time.time() - start_time
score_knn_original = clf_knn.score(test_data.drop(columns=['class']), test_data['class'])
print(f'Czas treningu kNN na danych oryginalnych: {knn_time_original:.2f} s')
print(f'Skuteczność kNN na danych oryginalnych: {score_knn_original:.4f}')

# Na danych zredukowanych
start_time = time.time()
clf_knn.fit(train_data_pca, train_data['class'])
knn_time_pca = time.time() - start_time
score_knn_pca = clf_knn.score(test_data_pca, test_data['class'])
print(f'Czas treningu kNN na danych skompresowanych: {knn_time_pca:.2f} s')
print(f'Skuteczność kNN na danych skompresowanych: {score_knn_pca:.4f}')

# Zadanie 10: Podsumowanie
print("\nPodsumowanie:")
print(f"Najlepsza jakość klasyfikatora na danych skompresowanych przy 90% wyjaśnionej wariancji: {score_lr_pca:.4f}")
print(f"Najlepszy klasyfikator: {'Logistic Regression' if score_lr_original >= max(score_knn_original, score_dt_original) else 'kNN' if score_knn_original > score_lr_original else 'Decision Tree'}")
print(f"Algorytm zyskujący najwięcej czasu na danych zredukowanych: {'Decision Tree' if dt_time_original - dt_time_pca >= max(lr_time_original - lr_time_pca, knn_time_original - knn_time_pca) else 'Logistic Regression' if lr_time_original - lr_time_pca > dt_time_original - dt_time_pca else 'kNN'}")