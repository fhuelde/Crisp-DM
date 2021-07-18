# Crisp-DM

# Importieren der Pakete pandas und numpy und Zuweisung zu den Abkürzungen
# pandas: zuständig für Datenanalyse und -bearbeitung
# numpy: zuständig für effizientes Rechnen mit großen Matrizen und Arrays
import pandas as pd
import numpy as np

# Importieren der Funktionen LabelBinarizer und OneHotEncoder aus dem sklearn.preprocessing Paket
# LabelBinarizer: zuständig für das Binarisieren von Labeln
# OneHotEncoder: zuständig für Codieren von kategoralen Merkmalen
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder

# Importieren der Pakete matplotlib.pyplo und seaborn und Zuweisung zu den Abkürzungen
# matplotlib.pyplot: zuständig für Erstellung und Änderung Plots; Funktionen basieren auf Matlab
# seaborn: zuständig für statistische Datenvisulalisierung
# seaborn "Default-Werte" setzen
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Importieren der Funktion KMeans aus dem Paket sklearn.cluster
# KMeans: zuständig für KMeans Clustering
# Importieren der Funktion MinMaxScaler aus dem Paket sklearn.prepocessing
# MinMaxScaler: zuständig für Skalierung von Objekten auf bestimmten Bereich
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Importieren des Pakets statsmodels.formula.api und Zuweisung zur Abkürzung
# statsmodels.formula.api: zuständig für das Erstellen von Modellen aus Formeln und DataFrames
import statsmodels.formula.api as smf

# Importieren der Funktion train_test_split aus dem sklearn.model_selection Paket
# train_test_split: zuständig für die Aufteilung eines Datensatzes in Test- und Trainingsdaten
from sklearn.model_selection import train_test_split

# Importieren des Pakets pickle und Zuweisung zur Abkürzung
# pickle: zuständig für die Serialisierung und Deserialisierung von Pythonobjekten in Byteströme
import pickle as pk

# Importieren der Funktion StandardScaler aus dem sklearn.preprocessing Paket
# StandardScaler: zuständig für Standardisierung von Merkmalen  
from sklearn.preprocessing import StandardScaler

# Importieren der Funktion VarianceThreshold aus dem sklearn.feature_selection Paket
# VarianceThreshold: zuständig für Entfernung von Merkmalen mit geringer Varianz
from sklearn.feature_selection import VarianceThreshold

# Importieren der Funktion metrics aus dem sklearn Paket
# metrics: zuständig für Berwertung von Vorhersagefehlern
from sklearn import metrics

# Importieren der Funktion RFE aus dem sklearn.feature_selection Paket
# RFE: zuständig für Auswahl von Merkmalen
# Importieren der Funktion LinearRegression aus dem sklearn.linear_model Paket
# LinearRegression: zuständig für Lineare Regression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

# Laden der CSV Datei in einen DataFrame namens house_prices
# Ausgabe der ersten 5 Zeilen
house_prices = pd.read_csv("C:/Users/User/Desktop/Felix/UNI/Masterstudium/Python II Advanced/Abschlussgespräch/Python Code Zwischenablage/house_prices_dataframe.csv")
house_prices.head()

# Ausgabe der Form des DataFrames (Anzahl Zeilen und Spalten)
house_prices.shape

# Ausgabe des Datentyps der jeweiligen Spalte
house_prices.dtypes

# Funktion zur Ausreißeridentifizierung mit Z-Score Verfahren
# Definieren der Funktion outliers_z_score mit der Werte identifiziert werden, 
# die größer sind als 3mal die Standardabweichung
def outliers_z_score(df):
    threshold = 3

    mean = np.mean(df)
    std = np.std(df)
    z_scores = [(y - mean) / std for y in df]
    return np.where(np.abs(z_scores) > threshold)
   
# Erstellen der Liste my_list mit ausschließlich numerischen Werten als Datentypen
# Erstellen der Liste num_columns der die Spalten aus dem DF mit den numerischen Datentypen aus my_list beihalten
# Erstellen des DF numerical_columns durch Filtern des DF house_prices mit den in num_columns definierten Spalten
# Ausgabe der ersten drei Zeilen des DF numerical_columns
my_list = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
num_columns = list(house_prices.select_dtypes(include=my_list).columns)
numerical_columns = house_prices[num_columns]
numerical_columns.head(3)

# Ausgabe der numerischen Spalten des DF numerical_columns 
# mit jeweiligen Ausreißern durch Anwenden der zuvor erstellten Funktion outliers_z_score (x)
outlier_list = numerical_columns.apply(lambda x: outliers_z_score(x), axis=0)
outlier_list

# Erstellen des DF df_of_outlier aus dem zuvor erstellen Array outlier_list
# Zuweisung eines Spaltennamens zum erstellten Dataframes
# Ausgabe des DF df_of_outlier
df_of_outlier = pd.DataFrame(outlier_list)
df_of_outlier.columns = ['Rows_to_exclude']
df_of_outlier

# Konvertieren der Spaltenwerte des DFs in einen numpy array
# Ausgabe des Arrays
outlier_list_final = df_of_outlier['Rows_to_exclude'].to_numpy()
outlier_list_final

# Verketten der Arrays zu einem
outlier_list_final = np.concatenate( outlier_list_final, axis=0 )

# Duplilizierte Werte werden entfernt
outlier_list_final_unique = set(outlier_list_final)
outlier_list_final_unique

# Boolesches Filtern der Werte aus outlier_list_final_unique mit denen 
# des anfänglichen DF house_prices in den Array filter_rows_to_exclude

# Erstellen eines DF df_without_outliers welcher die Werte von DF house_prices 
# ohne die Werte vom Array ilter_rows_to_exclude enthält
filter_rows_to_exclude = house_prices.index.isin(outlier_list_final_unique)
df_without_outliers = house_prices[~filter_rows_to_exclude]

# Ausgabe von der Länge der DFs house_prices und df_without_outliers, 
# Längendifferenz zwischen den beiden und der Anzahl der einzelnen Ausreißer DF outlier_list_final_unique 
# mit entsprechendem Text (String)
print('Length of original dataframe: ' + str(len(house_prices)))
print('Length of new dataframe without outliers: ' + str(len(df_without_outliers)))
print('----------------------------------------------------------------------------------------------------')
print('Difference between new and old dataframe: ' + str(len(house_prices) - len(df_without_outliers)))
print('----------------------------------------------------------------------------------------------------')
print('Length of unique outlier list: ' + str(len(outlier_list_final_unique)))

# Resetten bzw. Setzen eines neuen des Index des DFs df_without_outliers
df_without_outliers = df_without_outliers.reset_index()

# Umbennen der alten Indexspalte 
df_without_outliers = df_without_outliers.rename(columns={'index':'old_index'})

# Ausgabe der ersten 6 Zeilen des DFs df_without_outliers
df_without_outliers.head(6)

# Erstellen eines Boxplots mit den Daten aus der Spalte price vom DF house_prices 
sns.boxplot(x='price', data=house_prices)

# Erstellen eines Boxplots mit den Daten aus der Spalte price vom DF df_without_outliers --> Viel weniger Ausreißer
sns.boxplot(x='price', data=df_without_outliers)

# Definieren der Funktion missing_values_table
def missing_values_table(df):
        # Anzahl der fehldenden Werte summieren
        mis_val = df.isnull().sum()
        
        # Anteil der fehlenden Werte im Bezug auf alle berechnen
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Erstellen eines DF mit zwei Spalten mis_val und mis_val_percent
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Zuweisung von neuen Spaltennamen über ihre jeweiligen Indizies 
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sortieren der Prozentwerte aus Spalte 2 absteigend
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Ausgabe der Anzahl der Spalten des ausgangs DFs und Ausgabe wieviele Spalten fehlende Werte haben
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns

# Ausgabe der Funktion missing_values_table auf den DF df_without_outliers
missing_values_table(df_without_outliers)

# Entfernen der Spalte yr_renovated aus dem DF df_without_outliers
# Ausgabe der ersten 5 Zeilen des DFs
df_without_outliers = df_without_outliers.drop(['yr_renovated'], axis=1)
df_without_outliers.head()

# Füllen der leeren Werte in der Spalte grade mit dem Mittelwert der Spalte grade
df_without_outliers['grade'] = df_without_outliers['grade'].fillna(df_without_outliers['grade'].mean())

# Ausgabe der Funktion missing_values_table auf den DF df_without_outliers
missing_values_table(df_without_outliers)

# Zuweisung des DF df_without_outliers zu neuem Namen df_without_MV (MIssing Values) 
# --> Ohne Ausreißer und ohne fehlende Werte
df_without_MV = df_without_outliers

# Erstellen der Liste obj_col mit Werten des Datentyps object
# Erstellen der Liste object_columns der die Spalten aus dem DF df_without_MV 
# mit den Datentypen aus obj_col beihalten

# Erstellen des DF house_prices_categorical durch Filtern des DF df_without_MV 
# mit den in object_columns definierten Spalten
# Ausgabe der ersten drei Zeilen des DF numerical_columns
obj_col = ['object']
object_columns = list(df_without_MV.select_dtypes(include=obj_col).columns)
house_prices_categorical = df_without_MV[object_columns]

# Ausgabe der Anzahl der Spalten des DFs house_prices_categorical
print()
print('There are ' + str(house_prices_categorical.shape[1]) + ' categorical columns within dataframe:')

# Ausgabe der ersten 5 Zeilen des DFs
house_prices_categorical.head()

# Ausgabe wie oft jeder Wert in den drei Spalten waterfront, view und property_typ vorkommt
print('Values of the variable waterfront:')
print()
print(df_without_MV['waterfront'].value_counts())

print('--------------------------------------------')

print('Values of the variable view:')
print()
print(df_without_MV['view'].value_counts())

print('--------------------------------------------')

print('Values of the variable property_typ:')
print()
print(df_without_MV['property_typ'].value_counts())

# Umbenennen der Funktion LabelBinarizer zu encoder_waterfront
encoder_waterfront = LabelBinarizer()

# Anwenden der Funktion LabelBinarizer auf die Spalte waterfront des DFs --> No = 0, Yes = 1
waterfront_encoded = encoder_waterfront.fit_transform(df_without_MV.waterfront.values.reshape(-1,1))

# Einfügen der codierten Werte in das originale DF df_without_MV in eine neue Spalte waterfront_encoded
df_without_MV['waterfront_encoded'] = waterfront_encoded

# Entfernen der originalen Spalte waterfront aus dem DF
df_without_MV = df_without_MV.drop(['waterfront'], axis=1)

# Ausgabe der möglichen Werte der Spalte und der ersten 5 Zeilen des DFs df_without_MV
print(encoder_waterfront.classes_)
print('Codierung: no=0, yes=1')
print('-----------------------------')
print()
print('New Data Frame:')
df_without_MV.head()

# Erstellen eines Dictionaries view_dict, bei dem jedem möglichen Wert der Spalte view eine Zahl zugewiesen wird
view_dict = {'bad' : 0,
             'medium' : 1,
             'good' : 2,
             'very_good' : 3,
             'excellent' : 4}

# Zuordnung der Werte aus der Spalte view zu den Zahlen entsprechend des Dictionaries und 
# Einfügen in die neue Spalte view_encoded
df_without_MV['view_encoded'] = df_without_MV.view.map(view_dict)

# Entfernen der originalen Spalte view aus dem DF
df_without_MV = df_without_MV.drop(['view'], axis=1)

# Ausgabe der ersten 5 Spalten des DFs df_without_MV
print('New Data Frame:')
df_without_MV.head()

# Umbenennen der Funktion OneHotEncoder() zu encoder_property_typ
encoder_property_typ = OneHotEncoder()

# Anwenden der Funktion OneHotEncoder auf die Spalte property_typ des DFs --> Für jedes Attribut eine Spalte
OHE = encoder_property_typ.fit_transform(df_without_MV.property_typ.values.reshape(-1,1)).toarray()

# Umwandeln der Daten aus OHE in einen neuen DF df_OHE mithilfe einer for loop
df_OHE = pd.DataFrame(OHE, columns = ["property_typ_" + str(encoder_property_typ.categories_[0][i]) 
                                     for i in range(len(encoder_property_typ.categories_[0]))])


# Einfügen bzw. der neuen Spalten aus dem DF df_OHE in den DF df_without_MV
df_without_MV = pd.concat([df_without_MV, df_OHE], axis=1)


# Entfernen der originalen Spalte property_typ aus dem DF
df_without_MV = df_without_MV.drop(['property_typ'], axis=1)

# Ausgabe der ersten 5 Spalten des DFs df_without_MV
print('New Data Frame:')
df_without_MV.head()

# Zuweisung des DF df_without_MV zu neuem Namen final_df_house_prices 
final_df_house_prices = df_without_MV

# Umwnadeln und Abspeichern des DF final_df_house_prices als CSV
final_df_house_prices.to_csv("C:/Users/User/Desktop/Felix/UNI/Masterstudium/Python II Advanced/Abschlussgespräch/Python Code Zwischenablage/final_df_house_prices.csv", index=False)

#Ausgabe wie oft 0 und 1 in der Spalte waterfront_encoded vorkommt
print('Absolute distribution: ')
print()
print(final_df_house_prices['waterfront_encoded'].value_counts())

print('-------------------------------------------------------')

#Ausgabe wieviel Prozent aller Zeilen den Wert 0 oder 1 aufweisen
print('Percentage distribution: ')
print()
print(pd.DataFrame({'Percentage': final_df_house_prices.groupby(('waterfront_encoded')).size() / len(final_df_house_prices)}))

# Gruppieren des DFs nach waterfront_encoded Berechnung und Ausgabe der Mittelwerte der Werte der einzelnen Spalten
final_df_house_prices.groupby(('waterfront_encoded')).mean()

# Entfernen der Spalte old_inedx und Zuweisung zu neuem Namen house_prices_cluster
# Ausgabe des DFs house_prices_cluster

house_prices_cluster = final_df_house_prices.drop(['old_index'], axis=1)
house_prices_cluster

# Erstellung eines Historgramms aus der Spalte price des DF house_prices_cluster
plt.hist(house_prices_cluster['price'], bins='auto')
plt.title("Histogram for house prices")
plt.xlim(xmin=0, xmax = 1200000)
plt.show()

# Umbenennen der Funktion MinMaxScaler() zu mms
mms = MinMaxScaler()
# Anpassen der Werte des DFs house_prices_cluster für die Funktion 
mms.fit(house_prices_cluster)
# Transformation der Werte des DF house_prices_cluster zu einem Array und Zuweisung zu neuem Namen data_transformed
data_transformed = mms.transform(house_prices_cluster)

# Erstellen einer leeren Liste Sum_of_squared_distances
Sum_of_squared_distances = []
# Zuweisung eines Bereichs zwischen 1 und 15 zur Variable K
K = range(1,15)
# Durchführung KMeans Funktion auf den Array data_transformed und Einfügen der Werte in die leere Liste  
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(data_transformed)
    Sum_of_squared_distances.append(km.inertia_)
    
# Erstellen eines Liniengrafen mit den X-Werten von K (1-15) und den y-Werten von Sum_of_squared_distances
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

# Definition der KMeans Funktion mit 4 Clusterbildungen und random state=1 für wiederholbaren Start der Berechnungen
# Zuweisung der Funktion zum Namen km
km = KMeans(n_clusters=4, random_state=1)
# Anpassen der Werte für die Funktion bzw. den Algorithmus
km.fit(house_prices_cluster)

# Vorhersage des KMeans auf den DF house_prices_cluster als Liste unter dem Namen predict
predict=km.predict(house_prices_cluster)
# Erstellung einer neuen Spalte clusters im DF house_prices_cluster, in der die Vorhersagen eingetragen werden m it gleichem Index
house_prices_cluster['clusters'] = pd.Series(predict, index=house_prices_cluster.index)

# Werte der Spalten sqft_living und price des DF house_prices_cluster dem DF df_sub zuweisen
df_sub = house_prices_cluster[['sqft_living', 'price']].values

# Erstellung eines Scatter plots durch Vergleich zwischen der Liste predict und den Werten der beiden Spalten des DF df_sub
plt.scatter(df_sub[predict==0, 0], df_sub[predict==0, 1], s=100, c='red', label ='Cluster 1')
plt.scatter(df_sub[predict==1, 0], df_sub[predict==1, 1], s=100, c='blue', label ='Cluster 2')
plt.scatter(df_sub[predict==2, 0], df_sub[predict==2, 1], s=100, c='green', label ='Cluster 3')
plt.scatter(df_sub[predict==3, 0], df_sub[predict==3, 1], s=100, c='cyan', label ='Cluster 4')

# Zuweisung von Beschriftungen und Limits
plt.title('Cluster of Houses')
plt.xlim((0, 5000))
plt.ylim((0,2000000))
plt.xlabel('sqft_living \n\n Cluster1(Red), Cluster2 (Blue), Cluster3(Green), Cluster4(Cyan)')
plt.ylabel('Price')
plt.show()

#Zuweisung der beiden Spalten price und sqft_living aus dem DF final_df_house_prices zum DF HousePrices_SimplReg
HousePrices_SimplReg = final_df_house_prices[['price', 'sqft_living']]
# Ausgabe der ersten 5 Zeilen des neuen DFs
HousePrices_SimplReg.head()

# Zuweisung der Spalten des DFs zu x und y
x = HousePrices_SimplReg['sqft_living']
y = HousePrices_SimplReg['price']

#Erstellung eines Scatter Plots
plt.scatter(x, y)
plt.title('Scatter plot: sqft_living vs. price')
plt.xlabel('sqft_living')
plt.ylabel('price')
plt.show()

# Erstellen eines Modells aus der Funktion smf.ols aus der Formel und dem DF HousePrices_SimplReg und Zuweisung zu model1 ???
model1 = smf.ols(formula='price~sqft_living', data=HousePrices_SimplReg).fit()

# Ausgabe Infomationen über model1
model1.summary()

# Laden der CSV Datei in einen DataFrame namens house_prices
# Ausgabe der ersten 5 Zeilen
house_prices = pd.read_csv("C:/Users/User/Desktop/Felix/UNI/Masterstudium/Python II Advanced/Abschlussgespräch/Python Code Zwischenablage/house_prices_dataframe.csv")
house_prices.head()

#Aufteilen der Daten in Training und Test Datensatz, axis = 1 Werte entlang der Zeile
x = house_prices.drop(['price'], axis=1)
y = house_prices['price']

#Testgröße auf 20 % des Gesamtdatensatz festlegen
trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2)

#Reset des Index der 4 Variablen durch reset-Funktion
#Löschen des alten Index der 4 Variablen durch drop
trainX = trainX.reset_index().drop(['index'], axis=1)
testX = testX.reset_index().drop(['index'], axis=1)
trainY = trainY.reset_index().drop(['index'], axis=1)
testY = testY.reset_index().drop(['index'], axis=1)

#Funktion outliers_z_score definieren
#Threshold als Variable definieren mit Wert 3
def outliers_z_score(df):
    threshold = 3

    #Funktion numpy mean in Variable mean speichern
    #Funktion numpy Standardabweichung in Variable std speichern
    mean = np.mean(df)
    std = np.std(df)
    
    #Berechnung von z_scores
    #return einer Absoluten Zahl von Z_scores, die größer als Threshold (3) ist.
    z_scores = [(y - mean) / std for y in df]
    return np.where(np.abs(z_scores) > threshold)

#Erstellen einer Liste
#Erstellen einer Liste aus dem trainX-Datensatz, inklusive my_list
#Erstellen einer Liste aus trainX und num_columns
my_list = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
num_columns = list(trainX.select_dtypes(include=my_list).columns)
numerical_columns = trainX[num_columns]

#Anwenden der Funktion outliers_z_score auf numerical_columns
#Lambda-Funktion beschreibt anonyme Funktionen 
#Ausgabe outlier_list
outlier_list = numerical_columns.apply(lambda x: outliers_z_score(x))
outlier_list

#Speichern der Daten im DataFrame 
#Spalte umbenennen
#Ausgabe des DataFrames
df_of_outlier = pd.DataFrame(outlier_list)
df_of_outlier.columns = ['Rows_to_exclude']
df_of_outlier

#Umwandeln aller Werte in ein numpy Array
#Ausgabe der Array
outlier_list_final = df_of_outlier['Rows_to_exclude'].to_numpy()
outlier_list_final

#Verketten des Array in eine Spalte
outlier_list_final = np.concatenate( outlier_list_final, axis=0 )

#Entfernen von Duplikaten aus outlier_list_final
#Speichern in outliers_list_final_unique
outlier_list_final_unique = set(outlier_list_final)

#Speichern der Werte, dessen Index auch in der outlier_list_final_unique enthalten sind
#Erstellen eines DataFrame aus den Werten, die nicht in filter_rows_to_excluse
filter_rows_to_exclude = trainX.index.isin(outlier_list_final_unique)
trainX_wo_outlier = trainX[~filter_rows_to_exclude]

#Index reseten
#Löschen des Index der Zeilen
trainX_wo_outlier = trainX_wo_outlier.reset_index().drop(['index'], axis=1)

#Ausgabebefehle
#str = Ausgabe eines String, len = Länge des Wertes in der Klammer
print('Length of original dataframe: ' + str(len(trainX)))

print('Length of new dataframe without outliers: ' + str(len(trainX_wo_outlier)))
print('----------------------------------------------------------------------------------------------------')
print('Difference between new and old dataframe: ' + str(len(trainX) - len(trainX_wo_outlier)))
print('----------------------------------------------------------------------------------------------------')
print('Length of unique outlier list: ' + str(len(outlier_list_final_unique)))

#Speichern der Werte, dessen Index auch in der outlier_list_final_unique enthalten sind
#Erstellen eines DataFrame aus den Werten, die nicht in filter_rows_to_excluse
filter_rows_to_exclude = trainY.index.isin(outlier_list_final_unique)
trainY_wo_outlier = trainY[~filter_rows_to_exclude]

#Index reseten
#Löschen des Index der Zeilen
trainY_wo_outlier = trainY_wo_outlier.reset_index().drop(['index'], axis=1)

#Ausgabebefehle
print('Length of original dataframe: ' + str(len(trainY)))

print('Length of new dataframe without outliers: ' + str(len(trainY_wo_outlier)))
print('----------------------------------------------------------------------------------------------------')
print('Difference between new and old dataframe: ' + str(len(trainY) - len(trainY_wo_outlier)))
print('----------------------------------------------------------------------------------------------------')
print('Length of unique outlier list: ' + str(len(outlier_list_final_unique)))

#Erstellen einer FUnktion
def missing_values_table(df):
    
        #Variable erstellen der Summe fehlender Werte
        mis_val = df.isnull().sum()
        
        #Variable mit dem prozentualen Wert der Summe fehlender Werte
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        #Erstellen einer Tabelle mit den Ergebenissen der Variablen
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        #Umbenennen der Zeilen
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        #Tabelle nach absteigender Prozentzahl sortieren
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        #Ausgabe Anzahl aller Zeilen
        #Ausgabe Anzahl der Zeilen mit fehlenden Werten
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        return mis_val_table_ren_columns

#Ausgabe der erstellten Tabelle    
missing_values_table(trainX_wo_outlier)

#Löschen der Zeile yr_renovated, aufgrund zu viele fehlender Daten

trainX_wo_outlier = trainX_wo_outlier.drop(['yr_renovated'], axis=1)

#Einfügen von durchschnittswerten an fehlenden Stellen

trainX_wo_outlier['grade'] = trainX_wo_outlier['grade'].fillna(trainX_wo_outlier['grade'].mean())

#Überprüfen ob fehlende Werte ersetzt wurden

missing_values_table(trainX_wo_outlier)

#Ermitteln der Mittelwerte der 4 Spalten aus dem DataFrame trainX_wo_outlier
mean_grade = trainX_wo_outlier['grade'].mean()
mean_condition = trainX_wo_outlier['condition'].mean()
mean_sqft_living = trainX_wo_outlier['sqft_living'].mean()
mean_sqft_lot = trainX_wo_outlier['sqft_lot'].mean()

#Überschreiben der Werte anhand der ermittelten Mittelwerte (siehe oben).
pk.dump(mean_grade, open('C:/Users/User/Desktop/Felix/UNI/Masterstudium/Python II Advanced/Abschlussgespräch/Python Code Zwischenablage/Pickle/mean_grade.pkl', 'wb'))
pk.dump(mean_condition, open('C:/Users/User/Desktop/Felix/UNI/Masterstudium/Python II Advanced/Abschlussgespräch/Python Code Zwischenablage/Pickle/mean_condition.pkl', 'wb'))
pk.dump(mean_sqft_living, open('C:/Users/User/Desktop/Felix/UNI/Masterstudium/Python II Advanced/Abschlussgespräch/Python Code Zwischenablage/Pickle/mean_sqft_living.pkl', 'wb'))
pk.dump(mean_sqft_lot, open('C:/Users/User/Desktop/Felix/UNI/Masterstudium/Python II Advanced/Abschlussgespräch/Python Code Zwischenablage/Pickle/mean_sqft_lot.pkl', 'wb'))

#Ausgabe der Mittelwerte
print('Mean of column grade: ' + str(mean_grade))
print('Mean of column condition: ' + str(mean_condition))
print('Mean of column sqft_living: ' + str(mean_sqft_living))
print('Mean of column sqft_lot: ' + str(mean_sqft_lot))

#Datensatz umbenennen
trainX_wo_MV = trainX_wo_outlier

#Prüfen auf verschiedene Datentypen in Trainingsdatensatz
obj_col = ['object']
object_columns = list(trainX_wo_MV.select_dtypes(include=obj_col).columns)
trainX_categorical = trainX_wo_MV[object_columns]

#Ausgabe der Anzahl der Datentypen
print()
print('There are ' + str(trainX_categorical.shape[1]) + ' categorical columns within dataframe(trainX).')

# Umbenennen der Funktion LabelBinarizer zu encoder_waterfront
encoder_waterfront = LabelBinarizer()

#Fit_transform verbindet die beiden einzelnen Funktionen fit() und transform()
#reshape = Ändern der Form des Array ohne Daten zu verändern--> No = 0, Yes = 1
#Speichern in waterfron_encoded
waterfront_encoded = encoder_waterfront.fit_transform(trainX_wo_MV.waterfront.values.reshape(-1,1))

# Einfügen der codierten Werte in das originale DF df_without_MV in eine neue Spalte waterfront_encoded
trainX_wo_MV['waterfront_encoded'] = waterfront_encoded

# Entfernen der originalen Spalte waterfront aus dem DF
trainX_wo_MV = trainX_wo_MV.drop(['waterfront'], axis=1)

#Erstellen eines Dictionary mit vorgegebenem Schema
view_dict = {'bad' : 0,
             'medium' : 1,
             'good' : 2,
             'very_good' : 3,
             'excellent' : 4}
#Erstellen eines inveresen Dictionary
view_dict_inverse = {0 : 'bad',
                     1 : 'medium',
                     2 : 'good',
                     3 : 'very_good',
                     4 : 'excellent'}


# Zuordnung der Werte aus der Spalte view zu den Zahlen entsprechend des Dictionaries und 
# Einfügen in die neue Spalte view_encoded
trainX_wo_MV['view_encoded'] = trainX_wo_MV.view.map(view_dict)

# Entfernen der originalen Spalte view aus dem DF
trainX_wo_MV = trainX_wo_MV.drop(['view'], axis=1)

# Umbenennen der Funktion OneHotEncoder() zu encoder_property_typ
encoder_property_typ = OneHotEncoder()

#Fit_transform verbindet die beiden einzelnen Funktionen fit() und transform()
#reshape = Ändern der Form des Array ohne Daten zu verändern --> Für jedes Attribut eine Spalte
#Speichern in OHE
OHE = encoder_property_typ.fit_transform(trainX_wo_MV.property_typ.values.reshape(-1,1)).toarray()

#Umwandeln der neuen Daten in ein Dataframe durch for-Schleife
df_OHE = pd.DataFrame(OHE, columns = ["property_typ_" + str(encoder_property_typ.categories_[0][i]) 
                                     for i in range(len(encoder_property_typ.categories_[0]))])

#Einfügen der neuen Daten in den Originaldatensatz
trainX_wo_MV = pd.concat([trainX_wo_MV, df_OHE], axis=1)

# Entfernen der originalen Spalte property_typ aus dem DF
trainX_wo_MV = trainX_wo_MV.drop(['property_typ'], axis=1)

#Überschreiben der Dateien 
pk.dump(encoder_waterfront, open('C:/Users/User/Desktop/Felix/UNI/Masterstudium/Python II Advanced/Abschlussgespräch/Python Code Zwischenablage/Pickle/encoder_waterfront.pkl', 'wb'))
pk.dump(view_dict, open('C:/Users/User/Desktop/Felix/UNI/Masterstudium/Python II Advanced/Abschlussgespräch/Python Code Zwischenablage/Pickle/view_dict.pkl', 'wb'))
pk.dump(view_dict_inverse, open('C:/Users/User/Desktop/Felix/UNI/Masterstudium/Python II Advanced/Abschlussgespräch/Python Code Zwischenablage/Pickle/view_dict_inverse.pkl', 'wb'))
pk.dump(encoder_property_typ, open('C:/Users/User/Desktop/Felix/UNI/Masterstudium/Python II Advanced/Abschlussgespräch/Python Code Zwischenablage/Pickle/encoder_property_typ.pkl', 'wb'))

#Ausgabebefehle
print('Classes of encoder_waterfront :')
print(encoder_waterfront.classes_)
print('Codierung: no=0, yes=1')
print()
print('------------------------------------------------')
print()
print('Assignments by the dictionary for view :')
print(view_dict)
print()
print('------------------------------------------------')
print()
print('Assignments by the inverse dictionary for view :')
print(view_dict_inverse)
print()
print('------------------------------------------------')
print()
print('Categories of encoder_property_typ :')
print(encoder_property_typ.categories_)
print()

#Umbenennen des Datensatzes
trainX_encoded = trainX_wo_MV

#Speichern einer csv Datei
trainX_encoded.to_csv('C:/Users/User/Desktop/Felix/UNI/Masterstudium/Python II Advanced/Abschlussgespräch/Python Code Zwischenablage/Pickle/trainX_encoded.csv', index=False)

#Bennenen der Spaltenbezeichnung
col_names = trainX_encoded.columns
features = trainX_encoded[col_names]

#Skalieren der Daten mit Hilfe des Standardskalierers
#Transformieren der Daten
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)

#Erstellen eines DataFrame trainX_scaled
#Ausgabe der Kopfes des DataFrame
trainX_scaled = pd.DataFrame(features, columns = col_names)
trainX_scaled.head()

#Überschreiben der Datei mit scaler
pk.dump(scaler, open('C:/Users/User/Desktop/Felix/UNI/Masterstudium/Python II Advanced/Abschlussgespräch/Python Code Zwischenablage/Pickle/StandardScaler.pkl', 'wb'))

#Abbildungsgröße bestimmen
#Korrelation der Trainingsdaten erstellen
#Erstellen einer Heatmap, annot=true -> Werte in Heatmap, cmap =Farbpalette
#Abbildung zeigen
plt.figure(figsize=(8,6))
cor = trainX_scaled.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

#Erstellen einer correlationsmatrix mit den Werten aus dem skalierten Trainingsdatensatz
correlated_features = set()
correlation_matrix = trainX_scaled.corr()

#Variable definieren
threshold = 0.90

#Erstellen einer For-Schleife um zu überprüfen ob hochkorrelierte Variablen existieren
#Wenn ja (<0.9), Werte in colname speichern
for i in range(len(correlation_matrix .columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)
            
#Länge correlated_features bestimmen     
len(correlated_features)

#Anpassen des skalierten Trainingsdatensatz
constant_filter = VarianceThreshold(threshold=0)
constant_filter.fit(trainX_scaled)

#Ausschluss von konstanten Merkmalen
constant_columns = [column for column in trainX_scaled.columns
                    if column not in trainX_scaled.columns[constant_filter.get_support()]]



#Anpassen der skalierten Trainingsdatensatz mit einem thresholdwert von 0.01
qconstant_filter = VarianceThreshold(threshold=0.01)
qconstant_filter.fit(trainX_scaled)

#Ausschluss von konstanten Merkmalen
qconstant_columns = [column for column in trainX_scaled.columns
                    if column not in trainX_scaled.columns[qconstant_filter.get_support()]]


#Ausgabe der Anzahl konstanter features
print('Number of constant features: ' + str(len(constant_columns)))
print('Number of constant features: ' + str(len(qconstant_columns)))

#Achsen von trainX_scaled umkehren und trainX_scaled_T zuweisen
trainX_scaled_T = trainX_scaled.T

#Prüfen und Ausgabe Anzahl doppelter Merkmale in trainX_scaled_T
print('Number of duplicate features: ' + str(trainX_scaled_T.duplicated().sum()))

#Umbenennen des Datensatz
trainX_final = trainX_scaled

#Umbenennen des Datensatz
trainY_final = trainY_wo_outlier

#Erstellen CSV-Dateien
trainX_final.to_csv('C:/Users/User/Desktop/Felix/UNI/Masterstudium/Python II Advanced/Abschlussgespräch/Python Code Zwischenablage/Pickle/trainX_final.csv', index=False)
trainY_final.to_csv('C:/Users/User/Desktop/Felix/UNI/Masterstudium/Python II Advanced/Abschlussgespräch/Python Code Zwischenablage/Pickle/trainY_final.csv', index=False)

#Erstellen einer linearen Regression
#Anpassen der Daten durch die fit-Funktion
lm = LinearRegression()
lm.fit(trainX_final, trainY_final)

#Ausgabe des Wertes der linearen Regression
print('R² Score of fitted lm-model: ' + str(lm.score(trainX_final, trainY_final)))

#Überschreiben der Datei mit Hilfe der linearen regression
pk.dump(lm, open('C:/Users/User/Desktop/Felix/UNI/Masterstudium/Python II Advanced/Abschlussgespräch/Python Code Zwischenablage/Pickle/lm_model.pkl', 'wb'))

#Ausgabe testX
testX

#Löschen der Zeile yr_renovated
testX = testX.drop(['yr_renovated'], axis=1)

# Auffülen leerer Werte durch zuvor erstellte und gespeicherte Durchschnittswerte in Spalte grade
mean_grade_reload = pk.load(open("C:/Users/User/Desktop/Felix/UNI/Masterstudium/Python II Advanced/Abschlussgespräch/Python Code Zwischenablage/Pickle/mean_grade.pkl",'rb'))
testX['grade'] = testX['grade'].fillna(mean_grade_reload)

#Umbenennen des Datensatz
testX_wo_MV = testX

# Encoder und dict Pickle Files abrufen und in den Variablen abspeichern
# Verwenden der abgespeicherten Encoder und Dict, um die gleichen Modellparameter wie 
# beim Trainingsset beizubehalten (-> nur transform() kein erneutes fit())  
encoder_waterfront_reload = pk.load(open("C:/Users/User/Desktop/Felix/UNI/Masterstudium/Python II Advanced/Abschlussgespräch/Python Code Zwischenablage/Pickle/encoder_waterfront.pkl",'rb'))
view_dict_reload = pk.load(open("C:/Users/User/Desktop/Felix/UNI/Masterstudium/Python II Advanced/Abschlussgespräch/Python Code Zwischenablage/Pickle/view_dict.pkl",'rb'))
encoder_property_typ_reload = pk.load(open("C:/Users/User/Desktop/Felix/UNI/Masterstudium/Python II Advanced/Abschlussgespräch/Python Code Zwischenablage/Pickle/encoder_property_typ.pkl",'rb'))

# Anwenden der Funktion LabelBinarizer auf die Spalte waterfront des DFs --> No = 0, Yes = 1
waterfront_encoded = encoder_waterfront_reload.transform(testX_wo_MV.waterfront.values.reshape(-1,1))
# Einsetzen der neuen Spalte 'waterfront_encoded' in den DF testX_wo_MV mit den in waterfront_encoded codierten Daten
testX_wo_MV['waterfront_encoded'] = waterfront_encoded
# Entfernen der originalen Spalte waterfront aus dem DF
testX_wo_MV = testX_wo_MV.drop(['waterfront'], axis=1)

# Zuordnung der Werte aus der Spalte view zu den Zahlen entsprechend des Dictionaries view_dict_reload und 
# Einfügen in die neue Spalte view_encoded
testX_wo_MV['view_encoded'] = testX_wo_MV.view.map(view_dict_reload)
# Entfernen der originalen Spalte view aus dem DF
testX_wo_MV = testX_wo_MV.drop(['view'], axis=1)

# Anwenden der Funktion OneHotEncoder auf die Spalte property_typ_reload des DFs --> Für jedes Attribut eine Spalte
OHE = encoder_property_typ_reload.transform(testX_wo_MV.property_typ.values.reshape(-1,1)).toarray()
# Die neu generierten Daten aus OHE werden durch die for-Schleife in den neuen DataFrame df_OHE abgespeichert
# Namensgebung der Spalten durch "+ str()", sodass sich die Spalten nach Kategorie unterscheiden
df_OHE = pd.DataFrame(OHE, columns = ["property_typ_" + str(encoder_property_typ_reload.categories_[0][i]) 
                                      for i in range(len(encoder_property_typ_reload.categories_[0]))])
# Einfügen der codierten Werte aus df_OHE in den originalen DataFrame testX_wo_MV
testX_wo_MV = pd.concat([testX_wo_MV, df_OHE], axis=1)
# Löschen der Spalte 'property_typ' aus dem DataFrame um Duplikate zu vermeiden
testX_wo_MV = testX_wo_MV.drop(['property_typ'], axis=1)

testX_wo_MV.head()

# Umbenennung des DF's
testX_encoded = testX_wo_MV

# Ausgabe des DF's in eine CSV-Datei
testX_encoded.to_csv('C:/Users/User/Desktop/Felix/UNI/Masterstudium/Python II Advanced/Abschlussgespräch/Python Code Zwischenablage/Pickle/testX_encoded.csv', index=False)

# StandardScaler aus Pickle Files laden und in Variable scaler_reload abspeichern 
scaler_reload = pk.load(open("C:/Users/User/Desktop/Felix/UNI/Masterstudium/Python II Advanced/Abschlussgespräch/Python Code Zwischenablage/Pickle/StandardScaler.pkl",'rb'))

# speichern der Spaltenlabels des DF's in der Variable col_names
# abrufen der Spalten mittels der Labels und Speicherung in der Variable features
col_names = testX_encoded.columns
features = testX_encoded[col_names]

# Skalieren der Werte von features mithilfe des StandardScalers 
# Generiren eines DF's testX_scaled aus features mit Spaltennamen entsprechend der in col_names abgespeicherten Spaltennamen
features = scaler_reload.transform(features.values)
testX_scaled = pd.DataFrame(features, columns = col_names)

# Umbenennung der finalen testX und testY DF's
testX_final = testX_scaled
testY_final = testY

# Abspeichern in einer CSV-Datei
testX_final.to_csv('C:/Users/User/Desktop/Felix/UNI/Masterstudium/Python II Advanced/Abschlussgespräch/Python Code Zwischenablage/Pickle/testX_final.csv', index=False)
testY_final.to_csv('C:/Users/User/Desktop/Felix/UNI/Masterstudium/Python II Advanced/Abschlussgespräch/Python Code Zwischenablage/Pickle/testY_final.csv', index=False)

# Laden des LM-Modells aus den Pickle Files
lm_reload = pk.load(open("C:/Users/User/Desktop/Felix/UNI/Masterstudium/Python II Advanced/Abschlussgespräch/Python Code Zwischenablage/Pickle/lm_model.pkl",'rb'))

# Voraussagen der Daten des LM-Modells auf Basis der testX_final Daten
y_pred = lm_reload.predict(testX_final)

# Erstellen eines Arrays mit den vorhergesagten und den tatsächlichen Werten
# Umwandeln des Arrays in einen DataFrame mit Spaltennamen Actual und Predicted
actual_vs_predicted = np.concatenate((testY_final, y_pred), axis=1)
actual_vs_predicted = pd.DataFrame(actual_vs_predicted, columns = ["Actual", "Predicted"])

# Die ersten 30 Zeilen des DataFrames actual_vs_predicted werden in der Variable df1 abgespeichert
# Erstellung eines Balkendiagramms (Bar-Charts) mit Größe des Fensters von 10 * 6 inch
# Erstellung eines Hauptrasters mit durchgezogenen Linien, welche eine Dicke von 0,5 haben und die Farbe grün besitzen
# Erstellung eines Nebenrasters mit gepunkteten Linien, welche eine Dicke von 0,5 haben und die Farbe schwarz besitzen
# Ausgabe des Bar-Charts
df1 = actual_vs_predicted.head(30)
df1.plot(kind="bar", figsize=(10,6))
plt.grid(which="major", linestyle="-", linewidth="0.5", color="green")
plt.grid(which="minor", linestyle=":", linewidth="0.5", color="black")
plt.show()

# Ausgabe von Validierungsdaten, die mithilfe der metrics-Funktion berechnet wurden
# Die Berechnung erfolgt auf Basis der vorausgesagten und der tatsächlichen y-Werte
print('Mean Absolute Error:', metrics.mean_absolute_error(testY_final, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(testY_final, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(testY_final, y_pred)))

# Abspeichern der linearen Regressionsfunktion in lr
lr = LinearRegression()

# Anwenden der RFE-Funktion auf Basis der linearen Regression; 10 Merkmale sollen ausgewählt werden
# Fitten der RFE-Funtkion mithilfe der x und y Trainingsdaten
rfe = RFE(lr, n_features_to_select=10)
rfe.fit(trainX_final, trainY_final)

# speichern der Spaltenlabels aus trainX_final in Columns
# Berechnen von support und ranking, um Merkmale einzustufen und 5 der 16 auszuschließen
Columns = trainX_final.columns
RFE_support = rfe.support_
RFE_ranking = rfe.ranking_

# erstellen eines DF's dataset mit den Spaltennamen, dem RFE_support und dem RFE_ranking
# zum Ranking: Merkmale mit dem Wert 1 werden als "Beste" eingestuft
# zum Support: da nur 10 Merkmale ausgewählt werden können, werden die 6 auszuschließenden als False eingestuft
dataset = pd.DataFrame({'Columns': Columns, 'RFE_support': RFE_support, 'RFE_ranking': RFE_ranking}, 
                       columns=['Columns', 'RFE_support', 'RFE_ranking'])
dataset

# Behalten der Merkmale, deren support True ist und das Ranking 1 beträgt
df = dataset[(dataset["RFE_support"] == True) & (dataset["RFE_ranking"] == 1)]
# abspeichern der Spalte mit den übrig gebliebenen 10 Merkmalen und der Spaltennamen in filtered_features
filtered_features = df['Columns']
filtered_features

# speichern von filtered_features in einem Pickle File
pk.dump(filtered_features, open('C:/Users/User/Desktop/Felix/UNI/Masterstudium/Python II Advanced/Abschlussgespräch/Python Code Zwischenablage/Pickle/filtered_features_RFE.pkl', 'wb'))

# Einlesen der CSV-Datei trainX_encoded.csv
trainX_encoded_reload = pd.read_csv("C:/Users/User/Desktop/Felix/UNI/Masterstudium/Python II Advanced/Abschlussgespräch/Python Code Zwischenablage/Pickle/trainX_encoded.csv")
# Ausgabe der ersten 5 Zeilen des DataFrames
trainX_encoded_reload.head()

# Laden der Daten aus dem Pickle File in die Variable filtered_features_RFE_reload
filtered_features_RFE_reload = pk.load(open("C:/Users/User/Desktop/Felix/UNI/Masterstudium/Python II Advanced/Abschlussgespräch/Python Code Zwischenablage/Pickle/filtered_features_RFE.pkl",'rb'))

# Abrufen der filtered_features_RFE_reload aus trainX_encoded_reload und speichern in trainX_FS 
trainX_FS = trainX_encoded_reload[filtered_features_RFE_reload]

# speichern der Spaltenlabels des DF's in der Variable col_names
# abrufen der Spalten mittels der Labels und Speicherung in der Variable features
col_names = trainX_FS.columns
features = trainX_FS[col_names]

# Skalieren und Transformieren der Werte von features mithilfe des StandardScalers 
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)

# Generiren eines DF's testX_FS_scaled aus features mit Spaltennamen entsprechend der in col_names abgespeicherten Spaltennamen
trainX_FS_scaled = pd.DataFrame(features, columns = col_names)

# speichern von scaler in einem Pickle File
pk.dump(scaler, open('C:/Users/User/Desktop/Felix/UNI/Masterstudium/Python II Advanced/Abschlussgespräch/Python Code Zwischenablage/Pickle/StandardScaler2.pkl', 'wb'))

# Umbenennung der finalen trainX und trainY Daten
trainX_final2 = trainX_FS_scaled
trainY_final2 = trainY_final

# abspeichern trainX_final2 und trainY_final2 in einer CSV-Datei 
trainX_final2.to_csv('C:/Users/User/Desktop/Felix/UNI/Masterstudium/Python II Advanced/Abschlussgespräch/Python Code Zwischenablage/Pickle/trainX_final2.csv', index=False)
trainY_final2.to_csv('C:/Users/User/Desktop/Felix/UNI/Masterstudium/Python II Advanced/Abschlussgespräch/Python Code Zwischenablage/Pickle/trainY_final2.csv', index=False)

# Abspeichern der linearen Regressionsfunktion in lm2
# fitten der linearen Regressionsfunktion mit den Trainingsdaten
lm2 = LinearRegression()
lm2.fit(trainX_final2, trainY_final2)

# Ausgabe des R² Scores mithilfe der score-Funtkion auf Basis der Trainingsdaten
print('R² Score of fitted lm2-model: ' + str(lm2.score(trainX_final2, trainY_final2)))

# speichern von lm2 in einem Pickle File
pk.dump(lm2, open('C:/Users/User/Desktop/Felix/UNI/Masterstudium/Python II Advanced/Abschlussgespräch/Python Code Zwischenablage/Pickle/lm2_model.pkl', 'wb'))

# Einlesen der CSV-Datei testX_encoded.csv"
# Ausgabe der ersten 5 Zeilen des DF's
testX_encoded_reload = pd.read_csv("C:/Users/User/Desktop/Felix/UNI/Masterstudium/Python II Advanced/Abschlussgespräch/Python Code Zwischenablage/Pickle/testX_encoded.csv")
testX_encoded_reload.head()

# Laden der Daten aus dem Pickle File in die Variable filtered_features_RFE_reload
filtered_features_RFE_reload = pk.load(open("C:/Users/User/Desktop/Felix/UNI/Masterstudium/Python II Advanced/Abschlussgespräch/Python Code Zwischenablage/Pickle/filtered_features_RFE.pkl",'rb'))

# Abrufen der filtered_features_RFE_reload aus testX_encoded_reload und speichern in testX_FS 
testX_FS = testX_encoded_reload[filtered_features_RFE_reload]

# StandardScaler aus Pickle Files laden und in Variable scaler2_reload abspeichern 
scaler2_reload = pk.load(open("C:/Users/User/Desktop/Felix/UNI/Masterstudium/Python II Advanced/Abschlussgespräch/Python Code Zwischenablage/Pickle/StandardScaler2.pkl",'rb'))

# speichern der Spaltenlabels des DF's in der Variable col_names
# abrufen der Spalten mittels der Labels und Speicherung in der Variable features
col_names = testX_FS.columns
features = testX_FS[col_names]

# Skalieren der Werte von features mithilfe des Scalers 
features = scaler2_reload.transform(features.values)

# Generiren eines DF's testX_FS_scaled aus features mit Spaltennamen entsprechend der in col_names abgespeicherten Spaltennamen
testX_FS_scaled = pd.DataFrame(features, columns = col_names)

# Umbenennung der finalen testX 2 und testY 2 DF's
testX_final2 = testX_FS_scaled
testY_final2 = testY_final

# abspeichern von testX_final2 und testY_final2 in jeweils einer CSV-Datei
testX_final2.to_csv('C:/Users/User/Desktop/Felix/UNI/Masterstudium/Python II Advanced/Abschlussgespräch/Python Code Zwischenablage/Pickle/testX_final.csv', index=False)
testY_final2.to_csv('C:/Users/User/Desktop/Felix/UNI/Masterstudium/Python II Advanced/Abschlussgespräch/Python Code Zwischenablage/Pickle/testY_final.csv', index=False)

# Laden des LM-Modells aus den Pickle Files
lm2_reload = pk.load(open("C:/Users/User/Desktop/Felix/UNI/Masterstudium/Python II Advanced/Abschlussgespräch/Python Code Zwischenablage/Pickle/lm2_model.pkl",'rb'))

# Voraussagen der Daten des LM-Modells auf Basis der testX_final2 Daten
y_pred = lm2_reload.predict(testX_final2)

# Erstellen eines Arrays mit den vorhergesagten und den tatsächlichen Werten
# Umwandeln des Arrays in ein DataFrame mit Spaltennamen Actual und Predicted
actual_vs_predicted = np.concatenate((testY_final2, y_pred), axis=1)
actual_vs_predicted = pd.DataFrame(actual_vs_predicted, columns = ["Actual", "Predicted"])

# Die ersten 30 Zeilen des DataFrames actual_vs_predicted werden in der Variable df1 abgespeichert
# Erstellung eines Balkendiagramms (Bar-Charts) mit Größe des Fensters von 10 * 6 inch
# Erstellung eines Hauptrasters mit durchgezogenen Linien, welche eine Dicke von 0,5 haben und die Farbe grün besitzen
# Erstellung eines Nebenrasters mit gepunkteten Linien, welche eine Dicke von 0,5 haben und die Farbe schwarz besitzen
# Ausgabe des Bar-Charts
df1 = actual_vs_predicted.head(30)
df1.plot(kind="bar", figsize=(10,6))
plt.grid(which="major", linestyle="-", linewidth="0.5", color="green")
plt.grid(which="minor", linestyle=":", linewidth="0.5", color="black")
plt.show()

# Ausgabe von Validierungsdaten, die mithilfe der metrics-Funktion berechnet wurden
# Die Berechnung erfolgt auf Basis der vorausgesagten und der tatsächlichen y-Werte
print('Mean Absolute Error:', metrics.mean_absolute_error(testY_final2, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(testY_final2, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(testY_final2, y_pred)))

# Erstellen des DF's my_property aus einem Dictionary
my_property = pd.DataFrame({
                   'bedrooms': [3],
                   'bathrooms': [2],
                   'sqft_living': [180],
                   'sqft_lot': [1300],
                   'floors': [2],
                   'waterfront': ['yes'],
                   'view': ['good'],
                   'condition': [np.NaN],
                   'grade': [4],
                   'sqft_basement': [300],
                   'yr_built': [1950],
                   'yr_renovated': [2001],
                   'property_typ': ['apartment']})      
my_property

# laden des Pickle Files mean_condition.pkl und abspeichern in mean_condition_reload
mean_condition_reload = pk.load(open("C:/Users/User/Desktop/Felix/UNI/Masterstudium/Python II Advanced/Abschlussgespräch/Python Code Zwischenablage/Pickle/mean_condition.pkl",'rb'))
# Auffüllen des fehlenden Werts in der 'condition'-Spalte mit dem abgerufenen Wert aus mean_condition_reload
my_property['condition'] = my_property['condition'].fillna(mean_condition_reload)
# Umbenennen von my_property in my_property_wo_MV 
my_property_wo_MV = my_property

my_property_wo_MV

# Encoder und dict Pickle Files in Variablen abspeichern
encoder_waterfront_reload = pk.load(open("C:/Users/User/Desktop/Felix/UNI/Masterstudium/Python II Advanced/Abschlussgespräch/Python Code Zwischenablage/Pickle/encoder_waterfront.pkl",'rb'))
view_dict_reload = pk.load(open("C:/Users/User/Desktop/Felix/UNI/Masterstudium/Python II Advanced/Abschlussgespräch/Python Code Zwischenablage/Pickle/view_dict.pkl",'rb'))
encoder_property_typ_reload = pk.load(open("C:/Users/User/Desktop/Felix/UNI/Masterstudium/Python II Advanced/Abschlussgespräch/Python Code Zwischenablage/Pickle/encoder_property_typ.pkl",'rb'))


# Anwenden der Funktion LabelBinarizer auf die Spalte waterfront des DFs --> No = 0, Yes = 1
waterfront_encoded = encoder_waterfront_reload.transform(my_property_wo_MV.waterfront.values.reshape(-1,1))
# Einsetzen der neuen Spalte 'waterfront_encoded' in testX_wo_MV mit den in waterfront_encoded codierten Daten
my_property_wo_MV['waterfront_encoded'] = waterfront_encoded
# Entfernen der originalen Spalte waterfront aus dem DF
my_property_wo_MV = my_property_wo_MV.drop(['waterfront'], axis=1)


# Zuordnung der Werte aus der Spalte view zu den Zahlen entsprechend des Dictionaries view_dict_reload und 
# Einfügen in die neue Spalte view_encoded
my_property_wo_MV['view_encoded'] = my_property_wo_MV.view.map(view_dict_reload)
# Entfernen der originalen Spalte view aus dem DF
my_property_wo_MV = my_property_wo_MV.drop(['view'], axis=1)



# Anwenden der Funktion OneHotEncoder auf die Spalte property_typ des DFs --> Für jedes Attribut eine Spalte
OHE = encoder_property_typ_reload.transform(my_property_wo_MV.property_typ.values.reshape(-1,1)).toarray()
# Die neu generierten Daten aus OHE werden in den neuen DataFrame df_OHE abgespeichert
df_OHE = pd.DataFrame(OHE, columns = ["property_typ_" + str(encoder_property_typ_reload.categories_[0][i]) 
                                      for i in range(len(encoder_property_typ_reload.categories_[0]))])
# Einfügen der codierten Werte aus df_OHE in den originalen DataFrame my_property_wo_MV
my_property_wo_MV = pd.concat([my_property_wo_MV, df_OHE], axis=1)
# Löschen der Spalte 'property_typ' aus dem DataFrame um Duplikate zu vermeiden
my_property_wo_MV = my_property_wo_MV.drop(['property_typ'], axis=1)

# Umbenennung des DF's
my_property_encoded = my_property_wo_MV

# Laden der gefilterten Merkmale des RFE aus den Pickle Files
filtered_features_RFE_reload = pk.load(open("C:/Users/User/Desktop/Felix/UNI/Masterstudium/Python II Advanced/Abschlussgespräch/Python Code Zwischenablage/Pickle/filtered_features_RFE.pkl",'rb'))
# Abrufen der filtered_features_RFE_reload aus my_property_encoded und speichern in my_property_encoded 
my_property_encoded = my_property_encoded[filtered_features_RFE_reload]
# Umbenennung des DF's
my_property_FS = my_property_encoded

# StandardScaler aus Pickle Files laden und in Variable scaler2_reload abspeichern 
scaler2_reload = pk.load(open("C:/Users/User/Desktop/Felix/UNI/Masterstudium/Python II Advanced/Abschlussgespräch/Python Code Zwischenablage/Pickle/StandardScaler2.pkl",'rb'))

# speichern der Spaltenlabels des DF's in der Variable col_names
# abrufen der Spalten mittels der Labels und Speicherung in der Variable features
col_names = my_property_FS.columns
features = my_property_FS[col_names]

# Skalieren der Werte von features mithilfe des Scalers 
features = scaler2_reload.transform(features.values)

# Generieren eines DF's my_property_FS_scaled aus features mit 
# Spaltennamen entsprechend der in col_names abgespeicherten Spaltennamen
my_property_FS_scaled = pd.DataFrame(features, columns = col_names)

# Laden des LM-Modells aus den Pickle Files
lm2_reload = pk.load(open("C:/Users/User/Desktop/Felix/UNI/Masterstudium/Python II Advanced/Abschlussgespräch/Python Code Zwischenablage/Pickle/lm2_model.pkl",'rb'))

# Voraussagen der Daten des LM-Modells auf Basis der my_property_FS_scaled Daten
y_pred = lm2_reload.predict(my_property_FS_scaled)

# Ausgabe vom vorausgesagten Preis
print('The predicted price for my property is: ' + str(y_pred))

