# trabajo-final-OP_tracking-electoral  
**Seguimient(Tracking) Electoral de un Candidato**

Este proyecto desarrolla un algoritmo de análisis y visualización de datos electorales para realizar **tracking de imagen e intención de voto de un candidato** a lo largo del tiempo. Se trabaja con un dataset simulado de encuestas y se aplican técnicas de limpieza, normalización, agregación temporal, interpolación, estadísticas descriptivas e inferenciales.

---

## Datos Utilizados

El dataset contiene una muestra representativa con las siguientes variables:

| Variable                          | Tipo         | Descripción                                               |
|-----------------------------------|--------------|-----------------------------------------------------------|
| `fecha`                           | Fecha        | Fecha en la que fue realizada la encuesta (`yyyy-mm-dd`)  |
| `encuesta`                        | Categórica   | Identificador único de la encuesta                        |
| `estrato`                         | Categórica   | Estrato socioeconómico (bajo, medio, alto)                |
| `sexo`                            | Categórica   | Género del encuestado                                     |
| `edad`                            | Numérica     | Edad del encuestado                                       |
| `nivel_educativo`                 | Categórica   | Máximo nivel educativo alcanzado                          |
| `cantidad_integrantes_en_el_hogar`| Numérica     | Cantidad de personas en el hogar                          |
| `imagen_del_candidato`            | Numérica     | Puntaje (0 a 100) sobre la imagen del candidato           |
| `voto`                            | Categórica   | Intención de voto (positivo/negativo/nsnc)                |
| `voto_anterior`                   | Categórica   | A quién votó el encuestado en la elección anterior        |
