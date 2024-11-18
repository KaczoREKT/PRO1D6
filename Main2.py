import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import time

# Zadanie 1: Ładowanie zbioru danych i wstępna analiza
# Wczytaj dane MNIST.csv
data = pd.read_csv('MNIST.csv')

# Dodanie nazw kolumn
column_names = ['class'] + [f'pixel{i}' for i in range(1, 785)]
data.columns = column_names

# Wyświetlenie liczby rekordów i liczby cech
print("Liczba rekordów w zbiorze danych:", data.shape[0])
print("Liczba cech:", data.shape[1] - 1)

# Wyznaczenie rozkładu kategorii w procentach
class_distribution = data['class'].value_counts(normalize=True) * 100
print("Rozkład kategorii (w %):\n", class_distribution)

# Wykres słupkowy rozkładu kategorii
plt.figure(figsize=(10, 6))
sns.barplot(x=class_distribution.index, y=class_distribution.values)
plt.title('Rozkład kategorii w zbiorze danych')
plt.xlabel('Klasa')
plt.ylabel('Procent')
plt.show()

# Podział na zbiór treningowy i testowy
X = data.drop('class', axis=1)
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Zadanie 2: Analiza głównych składowych PCA
from sklearn.decomposition import PCA

# Ustawienie PCA tak, aby procent wyjaśnionej wariancji wynosił 0.90
pca = PCA(0.90)
pca.fit(X_train)
n_components = pca.n_components_
print("Liczba wymiarów, aby wyjaśnić 90% wariancji:", n_components)

# Zadanie 2.2: Stopień wyjaśnionej wariancji dla różnych wymiarów
explained_variance_ratios = []
for n in range(10, 151):
    pca_temp = PCA(n_components=n)
    pca_temp.fit(X_train)
    explained_variance_ratios.append(sum(pca_temp.explained_variance_ratio_))

plt.figure(figsize=(12, 6))
plt.plot(range(10, 151), explained_variance_ratios, marker='o')
plt.title('Stopień wyjaśnionej wariancji w zależności od liczby wymiarów')
plt.xlabel('Liczba wymiarów')
plt.ylabel('Wyjaśniona wariancja')
plt.grid()
plt.show()

# Zadanie 3: Redukcja wymiarów za pomocą PCA
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# Zadanie 4: Wizualizacja danych (pierwsze dwie składowe PCA)
plt.figure(figsize=(10, 6))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis', alpha=0.5)
plt.colorbar()
plt.title('Wizualizacja danych skompresowanych (PCA)')
plt.xlabel('Pierwsza składowa PCA')
plt.ylabel('Druga składowa PCA')
plt.show()

# Zadanie 5: Trenowanie klasyfikatora Decision Tree na danych oryginalnych
start_time = time.time()
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)
dt_train_time = time.time() - start_time
print("Czas trenowania Decision Tree na danych oryginalnych:", dt_train_time, "s")

y_pred_dt = dt_clf.predict(X_test)
print("Skuteczność Decision Tree:", accuracy_score(y_test, y_pred_dt))

# Zadanie 6: Trenowanie klasyfikatora Decision Tree na danych skompresowanych
start_time = time.time()
dt_clf_pca = DecisionTreeClassifier()
dt_clf_pca.fit(X_train_pca, y_train)
dt_train_time_pca = time.time() - start_time
print("Czas trenowania Decision Tree na danych skompresowanych:", dt_train_time_pca, "s")

y_pred_dt_pca = dt_clf_pca.predict(X_test_pca)
print("Skuteczność Decision Tree na danych skompresowanych:", accuracy_score(y_test, y_pred_dt_pca))

# Zadanie 7: Trenowanie klasyfikatora Logistic Regression na danych oryginalnych
start_time = time.time()
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
log_reg_train_time = time.time() - start_time
print("Czas trenowania Logistic Regression na danych oryginalnych:", log_reg_train_time, "s")

y_pred_log_reg = log_reg.predict(X_test)
print("Skuteczność Logistic Regression:", accuracy_score(y_test, y_pred_log_reg))

# Zadanie 8: Trenowanie Logistic Regression na danych skompresowanych
start_time = time.time()
log_reg_pca = LogisticRegression(max_iter=1000)
log_reg_pca.fit(X_train_pca, y_train)
log_reg_train_time_pca = time.time() - start_time
print("Czas trenowania Logistic Regression na danych skompresowanych:", log_reg_train_time_pca, "s")

y_pred_log_reg_pca = log_reg_pca.predict(X_test_pca)
print("Skuteczność Logistic Regression na danych skompresowanych:", accuracy_score(y_test, y_pred_log_reg_pca))

# Zadanie 9: Trenowanie kNN
knn = KNeighborsClassifier()

# Na danych oryginalnych
start_time = time.time()
knn.fit(X_train, y_train)
knn_train_time = time.time() - start_time
print("Czas trenowania kNN na danych oryginalnych:", knn_train_time, "s")

y_pred_knn = knn.predict(X_test)
print("Skuteczność kNN:", accuracy_score(y_test, y_pred_knn))

# Na danych skompresowanych
start_time = time.time()
knn.fit(X_train_pca, y_train)
knn_train_time_pca = time.time() - start_time
print("Czas trenowania kNN na danych skompresowanych:", knn_train_time_pca, "s")

y_pred_knn_pca = knn.predict(X_test_pca)
print("Skuteczność kNN na danych skompresowanych:", accuracy_score(y_test, y_pred_knn_pca))

# Zadanie 10: Podsumowanie
print("\nPodsumowanie:")
print("Najlepsza skuteczność na danych oryginalnych:")
print("Decision Tree:", accuracy_score(y_test, y_pred_dt))
print("Logistic Regression:", accuracy_score(y_test, y_pred_log_reg))
print("kNN:", accuracy_score(y_test, y_pred_knn))

print("\nNajlepsza skuteczność na danych skompresowanych:")
print("Decision Tree:", accuracy_score(y_test, y_pred_dt_pca))
print("Logistic Regression:", accuracy_score(y_test, y_pred_log_reg_pca))
print("kNN:", accuracy_score(y_test, y_pred_knn_pca))

# Analiza czasu trenowania
print("\nCzas trenowania na danych oryginalnych:")
print("Decision Tree:", dt_train_time)
print("Logistic Regression:", log_reg_train_time)
print("kNN:", knn_train_time)

print("\nCzas trenowania na danych skompresowanych:")
print("Decision Tree:", dt_train_time_pca)
print("Logistic Regression:", log_reg_train_time_pca)
print("kNN:", knn_train_time_pca)
