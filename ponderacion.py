
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

archivo = r"d:\lmonten\Desktop\MET ANALISIS OP\tp final OLEGO\tracking_electoral_dataset.csv"
df = pd.read_csv(archivo, sep=",")


def normalizar_nombres_columnas(df):
    # Hacemos una copia del DataFrame
    
    # Normalizamos los nombres de las columnas
    df.columns = (
        df.columns
        .str.strip()                         # elimina espacios al inicio y final
        .str.lower()                         # pasa todo a minúsculas
        .str.replace(" ", "_")               # reemplaza espacios por guiones bajos
        .str.replace("á", "a")               # reemplaza acentos
        .str.replace("é", "e")
        .str.replace("í", "i")
        .str.replace("ó", "o")
        .str.replace("ú", "u")
        .str.replace("ñ", "n")               # reemplaza ñ por n
    )
    
    # Mostrar los nombres ya normalizados
    print("Nombres de columnas normalizados:\n", df.columns.tolist())
    
    # Devolver el DataFrame modificado
    return df
df = normalizar_nombres_columnas(df)

def recodificar_sexo(df):
    # hacemos una copia para no modificar el DataFrame original
    df = df.copy()
    
    # limpiamos, normalizamos y recodificamos la columna "sexo"
    df["sexo"] = (
        df["sexo"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"femenino": 1, "masculino": 2, "otro": 3})
        .fillna(4)  # valor 4 para casos no identificados
        .astype(int)
    )
    
    # mostramos para control visual
    print(df["sexo"])
    
    # devolvemos el dataframe modificado
    return df

df = recodificar_sexo(df) 

4#formula de pesos e implientacion 

df["sexo"].value_counts().sum()

cuentas = df["sexo"].value_counts(dropna=True)

# Total de respuestas válidas
total = cuentas.sum()


# Total de masculinos (codificados como 1)
total_masculino = cuentas.get(1, 0)

# Porcentaje de masculino sobre total
porcentaje_masculino = total_masculino / total *100

print("Total respuestas:", total)
print("Total masculino (1):", total_masculino)
print("Porcentaje masculino:", porcentaje_masculino)


porcentaje_poblacional = 48.34   ### hicimos a mano porcentaje Poblacional con CENSO2022
peso_masculino = porcentaje_masculino / porcentaje_poblacional 
print(peso_masculino)

total_femenino = cuentas.get(2,0)
porcentaje_poblacionalfemenino = 51.62   # hicimos a mano porcentaje Poblacional con CENSO2022
porcentaje_femenino = total_femenino / total *100
peso_femenino = porcentaje_femenino / porcentaje_poblacionalfemenino
print (peso_femenino)


###recodificacion estrato

df["estrato"].head(24)
cuentas_prov = df["estrato"].value_counts(dropna=True)


def recodificar_estrato(df):
    df = df.copy()

    # Diccionario con provincias normalizado
    cod_provincias = {
        "ciudad autonoma de buenos aires": 1,
        "buenos aires": 2,
        "catamarca": 3,
        "chaco": 4,
        "chubut": 5,
        "cordoba": 6,
        "corrientes": 7,
        "entre rios": 8,
        "formosa": 9,
        "jujuy": 10,
        "la pampa": 11,
        "la rioja": 12,
        "mendoza": 13,
        "misiones": 14,
        "neuquen": 15,
        "rio negro": 16,
        "salta": 17,
        "san juan": 18,
        "san luis": 19,
        "santa cruz": 20,
        "santa fe": 21,
        "santiago del estero": 22,
        "tierra del fuego, antartida e islas del atlantico sur": 23,
        "tucuman": 24
    }

    # normalizacion de seguridad 
    df["estrato_normalizado"] = (
        df["estrato"]
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace("á", "a")
        .str.replace("é", "e")
        .str.replace("í", "i")
        .str.replace("ó", "o")
        .str.replace("ú", "u")
        .str.replace("ü", "u")
        .str.replace("ñ", "n")
    )

    # Mapear provincias a códigos numéricos
    df["estrato_cod"] = df["estrato_normalizado"].map(cod_provincias).astype("Int64")

    return df
df=(recodificar_estrato(df))

print(df["estrato_cod"].head(50))
df["estrato_cod"].astype("Int64")


total_respuestas = cuentas_prov.sum()

print("Total respuestas válidas:", total_respuestas)
print(cuentas_prov)

################ calculo de pesos por provincia
porcentaje_poblacional_prov = ({
    "Ciudad Autónoma de Buenos Aires": 6.8,
    "Buenos Aires": 38.18,
    "Catamarca": 0.93,
    "Chaco": 2.46,
    "Chubut": 1.29,
    "Córdoba": 8.36,
    "Corrientes": 2.64,
    "Entre Ríos": 3.1,
    "Formosa": 1.32,
    "Jujuy": 1.76,
    "La Pampa": 0.78,
    "La Rioja": 0.83,
    "Mendoza": 4.45,
    "Misiones": 2.78,
    "Neuquén": 1.54,
    "Rio Negro": 1.63,
    "Salta": 3.13,
    "San Juan": 1.78,
    "San Luis": 1.18,
    "Santa Cruz": 0.73,
    "Santa Fe": 7.72,
    "Santiago del Estero": 2.31,
    "Tierra del Fuego, Antártida e Islas del Atlántico Sur": 0.4,
    "Tucumán": 3.77
})

#

df["estrato_cod"] = df["estrato_cod"].astype("Int64")

porcentaje_poblacional_prov = {
    1: 6.8, 2: 38.18, 3: 0.93, 4: 2.46, 5: 1.29,
    6: 8.36, 7: 2.64, 8: 3.1, 9: 1.32, 10: 1.76,
    11: 0.78, 12: 0.83, 13: 4.45, 14: 2.78, 15: 1.54,
    16: 1.63, 17: 3.13, 18: 1.78, 19: 1.18, 20: 0.73,
    21: 7.72, 22: 2.31, 23: 0.4, 24: 3.77
}

cuentas = df["estrato_cod"].value_counts().sort_index()
total_respuestas = cuentas.sum()

resultados = []

for cod_prov, total_muestra in cuentas.items():

    cod_prov = int(cod_prov)   

    porcentaje_muestra = (total_muestra / total_respuestas) * 100
    porcentaje_poblacional = porcentaje_poblacional_prov.get(cod_prov)

    peso = None
    if porcentaje_poblacional is not None:
        peso = porcentaje_muestra / porcentaje_poblacional

    resultados.append({
        "provincia_cod": cod_prov,
        "total_muestra": total_muestra,
        "porcentaje_muestra": porcentaje_muestra,
        "porcentaje_poblacional": porcentaje_poblacional,
        "peso": peso
    })

df_pesos_prov = pd.DataFrame(resultados)
print(df_pesos_prov)

df_pesos_prov[["provincia_cod", "peso"]]


peso_masculino
peso_femenino

pesos_sexo = {
    1: peso_masculino,
    2: peso_femenino,
}


# ------------------------------------
# CREACIÓN DEL DATAFRAME DE PESOS POR SEXO
# ------------------------------------

df_pesos_sexo = pd.DataFrame([
    {
        "sexo_cod": 1,
        "descripcion": "masculino",
        "total_muestra": total_masculino,
        "porcentaje_muestral": porcentaje_masculino,
        "porcentaje_poblacional": 48.34,
        "peso_sexo": peso_masculino
    },
    {
        "sexo_cod": 2,
        "descripcion": "femenino",
        "total_muestra": total_femenino,
        "porcentaje_muestral": porcentaje_femenino,
        "porcentaje_poblacional": 51.62,
        "peso_sexo": peso_femenino
    },
    {
        "sexo_cod": 3,
        "descripcion": "otro",
        "total_muestra": cuentas.get(3, 0),
        "porcentaje_muestral": cuentas.get(3, 0) / total * 100 if total > 0 else 0,
        "porcentaje_poblacional": None,   # sin dato censal
        "peso_sexo": 1                    # peso neutral
    },
    {
        "sexo_cod": 4,
        "descripcion": "no identificado",
        "total_muestra": cuentas.get(4, 0),
        "porcentaje_muestral": cuentas.get(4, 0) / total * 100 if total > 0 else 0,
        "porcentaje_poblacional": None,
        "peso_sexo": 1                    # peso neutral
    }
])

print(df_pesos_sexo)


df["peso_sexo"] = df["sexo"].map(df_pesos_sexo.set_index("sexo_cod")["peso_sexo"])
df["peso_sexo"].head(10)

#####

import pandas as pd

# 1) Conteos reales
cuentas = df["sexo"].value_counts(dropna=True)
total = cuentas.sum()

# 2) Porcentajes muestrales
porcentaje_masculino = cuentas.get(1, 0) / total * 100
porcentaje_femenino = cuentas.get(2, 0) / total * 100

# 3) Porcentajes poblacionales (del Censo)
poblacional_m = 48.34
poblacional_f = 51.62

# 4) Pesos
peso_m = poblacional_m / porcentaje_masculino
peso_f = poblacional_f / porcentaje_femenino

# 5) Crear dataframe de pesos por sexo
tabla_pesos_sexo = pd.DataFrame({
    "sexo": [1, 2],
    "peso": [peso_m, peso_f]
})

print(tabla_pesos_sexo)

# 6) Asignar a cada caso
df["peso_sexo"] = df["sexo"].map({1: peso_m, 2: peso_f})

tabla_validacion = (
    df.groupby("sexo")
      .apply(lambda x: x["peso_sexo"].sum())
      .rename("peso_total")
)

tabla_validacion = tabla_validacion / tabla_validacion.sum() * 100
print(tabla_validacion)

##########provincia


# Conteo muestral por provincia
cuentas_prov = df["estrato_cod"].value_counts(normalize=False)

# Porcentaje muestral por provincia
porcentaje_muestral_prov = cuentas_prov / cuentas_prov.sum() * 100

# Convertir dict a series para operar más fácil
poblacional_prov = pd.Series(porcentaje_poblacional_prov)

# Alinear índices
porcentaje_muestral_prov = porcentaje_muestral_prov.sort_index()
poblacional_prov = poblacional_prov.sort_index()

# Pesos = porcentaje poblacional / porcentaje muestral
pesos_prov = poblacional_prov / porcentaje_muestral_prov

# Crear DataFrame de pesos por provincia
tabla_pesos_prov = pd.DataFrame({
    "provincia_cod": pesos_prov.index,
    "peso": pesos_prov.values
})

print(tabla_pesos_prov)

# Asignar peso a cada caso
df["peso_provincia"] = df["estrato_cod"].map(pesos_prov)

validacion = (
    df.groupby("estrato_cod")["peso_provincia"]
      .sum() / df["peso_provincia"].sum() * 100
)

print(validacion)

df["peso_final"] = df["peso_sexo"] * df["peso_provincia"]
print(df["peso_final"].head(24))
