import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#CARGAMOS CSV
ARCHIVO = r"d:\lmonten\Desktop\MET ANALISIS OP\tp final OLEGO\dataset_tracking_electoral_-ultimo.csv"

try:
    df = pd.read_csv(ARCHIVO, parse_dates=['Fecha'])
    print("Archivo cargado bien.")
except:
    print("No se encontró el archivo.")
    df = pd.DataFrame(columns=['Fecha','Encuesta','Estrato','Sexo','Edad','Nivel Educativo','Cantidad de Integrantes en el Hogar','Imagen del Candidato','Voto','Voto Anterior'])

# 1. EXPLORACION Y NORMALIZACIÓN INICIAL

## 1.1 VISTAS RÁPIDAS
print("Primeras filas:")
print(df.head())
print(df.tail())

print("Tamaño del archivo:")
print(df.shape)

print("Tipos de datos:")
print(df.info())

print("Resumen numérico:")
print(df.describe())

print("NaNs por columna:")
print(df.isnull().sum())


## 1.2 NORMALIZACIÓN DE NOMBRES DE COLUMNAS
def normalizar_nombres_columnas(df):

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

# 2.PROCESAMIENTO Y LIMPIEZA DE DATOS
## 2.1 FECHA
df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce') 
print("Tipos de datos tras convertir fecha:")
print(df.dtypes)

# Filtro: eliminamos filas sin fecha 
filas_antes = len(df)

# Borrar filas con NaN en cualquiera de estas 3 columnas
df = df.dropna(subset=['fecha', 'imagen_del_candidato', 'voto','voto_anterior'])
filas_eliminadas = filas_antes - len(df)
print(f"Filas eliminadas por NaN en fecha, imagen_del_candidato o voto: {filas_eliminadas}")

## 2.2 SEXO
def recodificar_sexo(df):
        
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

df = recodificar_sexo(df)  # aplica los cambios y guarda el resultado
print(df["sexo"])

## 2.2 GESTIÓN DE DUPLICADOS BASADA EN ID
# CORRECCIÓN: Se asegura que la columna 'encuesta' exista después de la normalización.
def eliminar_duplicados_por_id(df, columna_id='encuesta'):
    filas_antes = len(df)

    # borrar filas con el mismo ID, dejando la primera
    df = df.drop_duplicates(subset=[columna_id])

    filas_despues = len(df)

    print("Filas eliminadas por ID repetido:", filas_antes - filas_despues)
    return df
df = eliminar_duplicados_por_id(df, columna_id="encuesta")

print(df.tail())


## 2.3 LIMPIEZA DE COLUMNA IMAGEN DE CANDIDATO
def limpiar_columna_numerica(df, columna):
    print("LIMPIANDO LA COLUMNA:", columna)

    # valores originales
    print("Primeros valores ORIGINALES:")
    print(df[columna].head())

    # paso 1: convertir a número
    df[columna] = pd.to_numeric(
        df[columna].astype(str).str.replace("%", "").str.strip(),
        errors="coerce"
    )

    print("\nDespués de pasar todo a número:")
    print(df[columna].head())

    # paso 2: poner NaN a valores fuera de 0-100
    df.loc[(df[columna] < 0) | (df[columna] > 100), columna] = np.nan

    print("\nDespués de quitar valores fuera de 0-100:")
    print(df[columna].head())

    # paso 3: imputar NaN con la media
    media = df[columna].mean()
    cant_nan = df[columna].isna().sum()

    print("Cantidad de NaN encontrados:", cant_nan)
    print("Valor de la media para imputar:", media)

    df[columna] = df[columna].fillna(media)

    print("Después de imputar NaN con la media:")
    print(df[columna].head())

    # paso 4: redondear y convertir entero SÍ O SÍ
    df[columna] = df[columna].round(0).astype("int64")

    print("Después de redondear a ENTERO:")
    print(df[columna].head())

    print(" FIN DE LIMPIEZA ")

    return df

df = limpiar_columna_numerica(df, "imagen_del_candidato")

## 2.4 LIMPIEZA COLUMNA VOTO
def limpiar_texto(valor):
    if pd.isna(valor):
        return "desconocido"
    valor = str(valor)
    valor = valor.lower()
    valor = valor.strip()
    return valor

df["voto"] = df["voto"].apply(limpiar_texto)
df["voto_anterior"] = df["voto_anterior"].apply(limpiar_texto)

# 3. TRATAMIENTO DE VARIABLES CATEGÓRICAS (VOTO)
# deteccion de candidatos y recategorizacion
def detectar_tipo_voto_uno(voto_col):
    respuestas = voto_col.astype(str).str.lower().str.strip()

    # 3.1 Detectar multicandidato (tiene prioridad)
    nombres_candidatos = [
        "milei", "massa", "bullrich", "grabois",
        "del caño", "del cano", "randazzo"
    ]

    es_multi = respuestas.apply(
        lambda x: any(n in x for n in nombres_candidatos)
    ).any()

    if es_multi:
        return "multicandidato"

    # 3.2 Detectar binario (mucho más estricto)
    positivos = [
        "si lo voto", "sí lo voto",
        "lo votaria", "lo votaría",
        "si", "sí", "voto"
    ]

    negativos = [
        "no lo voto", "no lo votaria", "no lo votaría",
        "no votaria", "no votaría", "no voto", "no"
    ]

    def es_bin(x):
        x = x.strip()
        
        if len(x.split()) > 4:  
            return False  # si la frase es larga, no es binaria pura
        return any(p == x for p in positivos) or any(n == x for n in negativos)

    es_binario = respuestas.apply(es_bin).any()

    if es_binario:
        return "binario"

    return "desconocido"
quedetecta = detectar_tipo_voto_uno(df["voto"])
print(quedetecta)
df["voto"].head(20)

def mapear_voto_auto(df):
    
    tipo = detectar_tipo_voto_uno(df["voto"])
    print(f"Tipo detectado: {tipo}")

    # ==========================================================
    # CASO MULTICANDIDATO
    # ==========================================================
    if tipo == "multicandidato":

        map_voto = {
            "milei": 1,
            "massa": 2,
            "bullrich": 3,
            "grabois": 4,
            "del caño": 5, "del cano": 5,
            "randazzo": 6,
            "otro": 7,
            "ninguno": 9, "ns/nc": 9, "no sabe": 9, "desconocido": 9
        }

        def mapear_multi(x):
            x = str(x).lower().strip()
            for k, v in map_voto.items():
                if k in x:
                    return v
            return 0

        df["voto_mapeado"] = df["voto"].apply(mapear_multi)
        return df

    # CASO BINARIO
    if tipo == "binario":

        positivos = [
            "si lo voto", "sí lo voto", "lo voto",
            "lo votaría", "lo votaria", "si", "sí"
        ]

        negativos = [
            "no lo voto", "no lo votaría", "no lo votaria",
            "no votaría", "no votaria", "no voto", "no"
        ]

        def mapear_binario(x):
            x = str(x).lower().strip()
            if any(n == x for n in negativos):
                return 2
            if any(p == x for p in positivos):
                return 1
            return 0

        df["voto_mapeado"] = df["voto"].apply(mapear_binario)
        return df


    # CASO DESCONOCIDO
  
    df["voto_mapeado"] = 0
    return df

dfmap = mapear_voto_auto(df)
print(dfmap)

df["voto_mapeado"]


## 4. RECATEGORIZACIÓN ANTES DEL TRACKING

df['voto'].astype(str).str.lower().str.strip().value_counts(dropna=False)


print(df.columns.tolist())

df.mapeado = pd.DataFrame({
    "voto": [
        "Milei", "javier milei", "Sergio Massa", "Bullrich",
        "Juan Grabois", "Del Cano", "Randazzo", "ns/nc", "Ninguno", "otros"
    ],
    "voto_anterior": [
        "massa", "Patricia Bullrich", "del cano", "no sabe",
        "desconocido", "Juan Grabois", "Randazzo", "otros", "milei", None
    ]
})

# 4.1 MAPEO

map_voto = {
    "milei": 1,
    "javier milei": 1,

    "massa": 2,
    "sergio massa": 2,

    "bullrich": 3,
    "patricia bullrich": 3,

    "grabois": 4,
    "juan grabois": 4,

    "del cano": 5,
    "del caño": 5,

    "randazzo": 6,

    "otros": 7,

    "desconocido": 9,
    "ns/nc": 9,
    "no sabe": 9,
    "ninguno": 9
}


# 4.2 CREA COLUMNAS NUEVAS 

df["voto_normalizado"] = df["voto"].astype(str).str.lower()
df["voto_mapeado"] = df["voto_normalizado"].map(map_voto).fillna(0).astype(int)

df["voto_anterior_normalizado"] = df["voto_anterior"].astype(str).str.lower()
df["voto_anterior_mapeado"] = df["voto_anterior_normalizado"].map(map_voto).fillna(0).astype(int)


# 5. PONDERACIÓN
# 5.1 PESOS POR SEXO

df["sexo"].head(20)

df["sexo"].value_counts().sum()

cuentas = df["sexo"].value_counts(dropna=True)

# total de respuestas válidas
total = cuentas.sum()

# total masculinos (codificados como 1)
total_masculino = cuentas.get(1, 0)

# Porcentaje de masculino sobre total
porcentaje_masculino = total_masculino / total *100

print("Total respuestas:", total)
print("Total masculino (1):", total_masculino)
print("Porcentaje masculino:", porcentaje_masculino)


porcentaje_poblacional = 48.34   # hicimos a mano porcentaje Poblacional con CENSO2022
peso_masculino = porcentaje_masculino / porcentaje_poblacional 
print(peso_masculino)

total_femenino = cuentas.get(2,0)
porcentaje_poblacionalfemenino = 51.62   # hicimos a mano porcentaje Poblacional con CENSO2022
porcentaje_femenino = total_femenino / total *100
peso_femenino = porcentaje_femenino / porcentaje_poblacionalfemenino
print (peso_femenino)

df_pesos_sexo = {
    1: peso_masculino, 
    2: peso_femenino,
}
print(df_pesos_sexo)


# 5.2 PESOS POR PROVINCIA (ESTRATO)

df["estrato"].head(24)
cuentas_prov = df["estrato"].value_counts(dropna=True)


def recodificar_estrato(df):
   
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
df=(recodificar_estrato(df)).dropna(subset=["estrato_cod"])

print(df["estrato_cod"].head(50))
df["estrato_cod"].astype("Int64")


total_respuestas = cuentas_prov.sum()

print("Total respuestas válidas:", total_respuestas)
print(cuentas_prov)

# calculo de pesos por provincia
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

df["estrato_cod"] = df["estrato_cod"].astype("Int64")

porcentaje_poblacional_prov = {
    1: 6.8, 2: 38.18, 3: 0.93, 4: 2.46, 5: 1.29,
    6: 8.36, 7: 2.64, 8: 3.1, 9: 1.32, 10: 1.76,
    11: 0.78, 12: 0.83, 13: 4.45, 14: 2.78, 15: 1.54,
    16: 1.63, 17: 3.13, 18: 1.78, 19: 1.18, 20: 0.73,
    21: 7.72, 22: 2.31, 23: 0.4, 24: 3.77
}

cuentas_estrato = df["estrato_cod"].value_counts().sort_index()
total_respuestas = cuentas_estrato.sum()
df["estrato_cod"].head(24)

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
print(df_pesos_prov.head(24))
df_pesos_prov[["provincia_cod", "peso"]]

pesoprovincial = dict(zip(df_pesos_prov["provincia_cod"], df_pesos_prov["peso"]))
print(pesoprovincial)


df["peso_sexo"] = df["sexo"].map(df_pesos_sexo)
df["peso_prov"] = df["estrato_cod"].map(pesoprovincial)
df["peso_final"] = df["peso_sexo"] * df["peso_prov"]
df["peso_final"].head(24).dropna()


# 5.3 APLICACIÓN DE PESOS A VARIABLES IMAGEN DEL CANDIDATO Y VOTO (MULTICANDIDATO)
variables = ["imagen_del_candidato"]

for var in variables:
    df[var + "_pond"] = df[var] * df["peso_final"]
    print(f"\nPrimeras filas de la variable ponderada {var}_pond:")
    print(df[[var, var + "_pond"]].head())

df["estrato"] = pd.to_numeric(df["estrato"], errors='coerce').fillna(0).astype(int)

print(df.head())



# TRACKING ELECTORAL
# tracking simple 
def tracking_7_dias(df):
    df["fecha"] = pd.to_datetime(df["fecha"])

    # Agrupamos cada 7 días
    tracking_simple = (
        df.groupby([pd.Grouper(key="fecha", freq="7D")])
        .agg(
            imagen_media=("imagen_del_candidato_pond", "mean"),
            casos=("imagen_del_candidato", "count")
        ).dropna()
        .reset_index()
    )

    return tracking_simple

tracking_simple = tracking_7_dias(df)

print(tracking_simple.head(15))
plt.figure(figsize=(10, 5))
plt.plot(tracking_simple["fecha"], tracking_simple["imagen_media"], marker='o')
plt.title("Tracking de Imagen del Candidato Ponderada - Semanal")
plt.xlabel("Fecha")
plt.ylabel("Imagen Media")
plt.grid(True)
plt.show()
##### fin de tracking de imagen candidato ponderada #####


########## TRACKING de CANDIDATO BINARIO ###########
for c in df["voto_mapeado"].unique():
    df[f"voto_{c}"] = (df["voto_mapeado"] == c).astype(int)
# lista de valores detectados (1 y 2)
categorias = sorted(df["voto_mapeado"].dropna().unique().tolist())
print("Categorias detectadas en voto_mapeado:", categorias)

# tracking semanal (cada 7 días)
# columnas para cada valor de voto_mapeado (binario: 1=SI, 2=NO)
for c in [1, 2]:
    df[f"voto_{c}"] = (df["voto_mapeado"] == c).astype(int)

tracking_bin = (
    df.set_index("fecha")
      .groupby(pd.Grouper(freq="7D"))
      [["voto_1", "voto_2"]]
      .sum()
)

print(tracking_bin.head(15))

plt.figure(figsize=(12,5))
plt.plot(tracking_bin.index, tracking_bin["voto_1"], marker="o", linewidth=2, label="Sí (1)")
plt.plot(tracking_bin.index, tracking_bin["voto_2"], marker="o", linewidth=2, label="No (2)")
plt.title("Tracking semanal: Sí vs No")
plt.xlabel("Fecha")
plt.ylabel("Cantidad de casos")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()


########## TRACKING POR CANDIDATO - MULTICANDIDATO ###########

df["voto_mapeado"]
for c in df["voto_mapeado"].unique():
    df[f"voto_{c}"] = (df["voto_mapeado"] == c).astype(int)
candidatos = df["voto_mapeado"].dropna().unique().tolist()
print("Candidatos detectados:", candidatos)

# tracking semanal (cada 7 días)
tracking_por_candidato = (
    df.set_index("fecha")
      .groupby(pd.Grouper(freq="7D"))
      [[f"voto_{c}" for c in candidatos]]
      .sum()
)
print(tracking_por_candidato.head(15))

for c in candidatos:
    serie = tracking_por_candidato[f"voto_{c}"]

    plt.figure(figsize=(10,4))
    plt.plot(serie.index, serie.values)
    plt.title(f"Tracking semanal, candidato: {c}")
    plt.xlabel("Fecha")
    plt.ylabel("Cantidad de casos")
    plt.grid(True)
    plt.show()

###### re-mapeo para CANDIDATOS
MAPEO_CANDIDATOS = {
    1: "Javier Milei",
    2: "Sergio Massa",
    3: "Patricia Bullrich",
    4: "Juan Grabois",
    5: "Del Caño",
    6: "Randazzo",
    7: "Otros",
    9: "NS/NC/Nulo",
    
}

candidatos_a_trackear = sorted([c for c in df["voto_mapeado"].unique() if c in MAPEO_CANDIDATOS and c != 9])
print("Códigos de candidatos a trackear:", candidatos_a_trackear)

for c in candidatos_a_trackear:
    # Nombre de la columna: 'voto_pond_1', 'voto_pond_2', etc.
    df[f"voto_pond_{c}"] = np.where(df["voto_mapeado"] == c, df["peso_final"], 0)

# CANTIDAD PONDERADA DE VOTOS por semana.
tracking_ponderado = (
    df.set_index("fecha")
    .groupby(pd.Grouper(freq="7D")) 
    [[f"voto_pond_{c}" for c in candidatos_a_trackear] + ["peso_final"]] # Incluimos peso_final total para normalizar
    .sum()
)

tracking_ponderado["total_voto_valido_pond"] = tracking_ponderado[[f"voto_pond_{c}" for c in candidatos_a_trackear]].sum(axis=1)

# normalización a Porcentaje (%)
tracking_porcentaje = pd.DataFrame()
for c in candidatos_a_trackear:
    tracking_porcentaje[MAPEO_CANDIDATOS[c]] = (
        tracking_ponderado[f"voto_pond_{c}"] / tracking_ponderado["total_voto_valido_pond"]
    ) * 100

print("\nTracking de Porcentaje de Voto Ponderado (Primeras 15 semanas):")
print(tracking_porcentaje.head(15))

#GRAFICO

# gráfico consolidado 
plt.figure(figsize=(12, 7))
plt.style.use('seaborn-v0_8-darkgrid')

for nombre_candidato in tracking_porcentaje.columns:
    plt.plot(
        tracking_porcentaje.index, 
        tracking_porcentaje[nombre_candidato].values, 
        label=nombre_candidato,
        marker='o', 
        markersize=4,
        linewidth=2
    )

plt.title("Tracking Electoral Ponderado (Voto Válido) - Frecuencia Semanal", fontsize=16)
plt.xlabel("Fecha de Cierre Semanal", fontsize=12)
plt.ylabel("Intención de Voto Ponderada (%)", fontsize=12)
plt.legend(title="Candidato", loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout(rect=[0, 0, 0.85, 1]) # Ajusta para que la leyenda quepa
plt.xticks(rotation=45, ha='right')
plt.show()

# gráficos Indndividuales 
for nombre_candidato in tracking_porcentaje.columns:
    serie = tracking_porcentaje[nombre_candidato]
    
    plt.figure(figsize=(10, 4))
    plt.plot(serie.index, serie.values, marker='o', linestyle='-', color='purple')
    plt.title(f"Tracking Semanal de {nombre_candidato} (Ponderado)", fontsize=14)
    plt.xlabel("Fecha", fontsize=11)
    plt.ylabel("Intención de Voto Ponderada (%)", fontsize=11)
    plt.ylim(0, max(serie.max() * 1.2, 10)) # Asegura un buen rango Y
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

## VAIABLE IMAGEN DEL CANDIDATO PONDERADA

# promedio diario
df_imagen = (
    df.groupby('fecha')['imagen_del_candidato_pond']
      .mean()
      .rename('imagen_media_diaria')
      .to_frame()
)

# 2. tracking semanal cada 7 días
tracking_imagen = (
    df_imagen
        .resample("7D")
        .mean()
        .rename(columns={"imagen_media_diaria": "imagen_media_semanal"})
)


print("Tracking de Imagen del Candidato X:")
print(tracking_imagen.head(15))
#  gráfico de tracking de imagen del candidato
plt.figure(figsize=(10, 5))
plt.plot(
    tracking_imagen.index, 
    tracking_imagen["imagen_media_semanal"].values, 
    marker='o', linestyle='-', color='green'
)
plt.title("Tracking Semanal de Imagen del Candidato (Ponderado)", fontsize=16)
plt.xlabel("Fecha", fontsize=12)
plt.ylabel("Imagen Media Ponderada", fontsize=12)
plt.ylim(0, 400)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()




