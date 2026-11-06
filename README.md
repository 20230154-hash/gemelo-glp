# Gemelo digital experimental de GLP

Aplicación Streamlit para simular concentraciones de GLP, aplicar un filtro de Kalman y comparar Random Forest con Gradient Boosting. El entregable web se obtiene al publicarla en Streamlit Community Cloud.

## Colores y alarma virtual

La misma regla se aplica a la tarjeta de predicción, los puntos del gráfico y la tabla de predicciones:

- Verde: concentración menor que el umbral.
- Rojo: concentración mayor o igual que el umbral.
- Umbral académico inicial: 147 ppm. Se puede cambiar en la barra lateral.

Con el umbral inicial, 100 y 146 ppm son verdes; 147 y 180 ppm son rojos. La prueba manual sirve para comprobar la interfaz. Sus valores se identifican como manuales y no sustituyen la predicción del modelo.

El escenario inicial de simulación permite observar predicciones en ambos colores. Las concentraciones son sintéticas. El umbral de 147 ppm es una referencia académica configurable, no un umbral de seguridad certificado. La alarma es exclusivamente virtual.

## Publicar y obtener el enlace

1. Descargue el ZIP y use «Extraer todo» en Windows.
2. Abra las carpetas extraídas hasta encontrar `app.py`, `requirements.txt` y este `README.md`.
3. Entre a https://github.com e inicie sesión o cree una cuenta.
4. Cree un repositorio llamado `gemelo-glp`. Seleccione Public y active la opción para añadir un README. Pulse Create repository.
5. En el repositorio, seleccione Add file > Upload files. Arrastre los tres archivos extraídos. Deben quedar directamente en la raíz del repositorio.
6. Guarde la carga con Commit changes.
7. Entre a https://share.streamlit.io e inicie sesión. Vincule su cuenta de GitHub si se solicita.
8. Seleccione Create app. Si aparece la pregunta sobre tener una aplicación, elija Yup, I have an app.
9. Seleccione su repositorio `usuario/gemelo-glp`, la rama `main` y Main file path `app.py`.
10. Opcionalmente, elija un nombre disponible para App URL.
11. En Advanced settings, seleccione Python 3.12 y guarde. Esta versión se usó para verificar la aplicación.
12. Pulse Deploy y espere a que termine la instalación.
13. Abra la web resultante. En Share > Copy link copie la dirección de la aplicación. Esa dirección termina en `.streamlit.app` y es el enlace que se entrega.
14. Compruebe que el enlace abre en una ventana de incógnito. Si pide acceso, revise Settings > Sharing > Who can view this app y seleccione This app is public and searchable.

Las bibliotecas se instalan automáticamente desde `requirements.txt`. Para esta publicación no necesita Anaconda ni ejecutar comandos en su computadora.

## Comprobar el entregable

1. Mantenga el umbral en 147 ppm.
2. En «Prueba de colores de la alarma», introduzca 100. El número debe verse verde y mostrar SIN ALARMA.
3. Introduzca 180. El número debe verse rojo y mostrar ALARMA ACTIVADA.
4. Introduzca 147. El número debe verse rojo, pues alcanzar el umbral ya activa la alarma.
5. Revise el gráfico de predicción y la tabla: los valores que alcanzan o superan el umbral son rojos y los valores inferiores son verdes.
6. En «Gemelo digital y luz de emergencia virtual», mueva «Instante de reproducción dentro de la prueba». La tarjeta usa la predicción del modelo para ese instante y cambia de color.
7. El umbral cambia los colores inmediatamente. Para aplicar cambios a la simulación o al horizonte, pulse «Generar señales y entrenar modelos».
8. Entregue el enlace de la web. Como evidencia adicional puede guardar una captura con la alarma en rojo y otra en verde.

## Ejecutar en una computadora opcionalmente

Python 3.12 y las dependencias de `requirements.txt` son suficientes:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Alcance y método

- Simulador ilustrativo de una zona; no usa mediciones reales ni un modelo físico validado.
- Filtro de Kalman escalar causal.
- RandomForestRegressor e HistGradientBoostingRegressor para predecir concentración a t+h.
- Comparación en el último 25 % temporal de una trayectoria sintética; no representa validación experimental ni una estimación suficiente de generalización.
- Exportación CSV de señales, métricas y predicciones. Las predicciones exportadas incluyen umbral, estado y activación de la alarma virtual.
- MQTT TLS opcional para publicar el estado virtual. Se mantiene desactivado por defecto y no es necesario para el entregable.

Un sistema real de detección debe usar sensores calibrados y alarmas certificadas independientes. No conectar una sirena, relé o luz física al resultado de esta simulación.

## Documentación oficial

- Publicación: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- Compartir la aplicación: https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app
- Cargar archivos en GitHub: https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository
