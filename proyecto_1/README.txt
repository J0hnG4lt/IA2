Regresión Lineal Múltiple

Integrantes:
	Erick Silva			11-10906
	Leslie Rodrigues	10-10613
	Georvic Tur			12-11402


Requerimientos
	
	Como se va a trabajar con NumPy y Matplotlib, se recomienda ejecutar la
	siguiente orden en el directorio actual:
		
		sudo pip install requirements.txt
		
	Luego de haber instalado exitosamente los requerimientos, se pueden correr
	los scripts.

Organización de los módulos
	
	parseInput.py
		Aquí se implementa el preprocesamiento del dataset de la tercera pregunta.
	customValues.txt
		Aquí se establecen los valores numéricos a usar con los atributos nominales.
	
	datasets/dataActividad3.csv
		Este es el dataset preprocesado por parseInput a partir de data.txt
	datasets/data.txt
		Este es el dataset de la tercera pregunta
	datasets/x01_copia.txt
		Dataset del peso
	datasets/x08_copia.txt
		Dataset de homicidios
	
	preguntas/proyecto.py
		Aquí se implementan todas las funciones usadas en las preguntas.
	
	preguntas/pregunta_dos_parte_a.py
	preguntas/pregunta_dos_parte_b.py
	preguntas/pregunta_dos_crimen_parte_a.py
	preguntas/pregunta_dos_crimen_parte_b.py
	preguntas/pregunta_tres.py


Instrucciones
	
	Los scripts se deben correr en la carpeta "preguntas"
		
		cd preguntas
		
	Para evaluar la parte 'a' y 'b' de la segunda pregunta con el dataset de peso, 
	hay que correr los siguientes scripts:
		
		python pregunta_dos_parte_a.py
		
		python pregunta_dos_parte_b.py
		
		Se puede modificar dentro de ellos los valores del aprendizaje y el
		número de decimales a mostrar en las gráficas.
		
	Para evaluar la parte 'a' y 'b' de la segunda pregunta con el dataset de 
	homicidios, hay que correr los siguientes scripts:
	
		python pregunta_dos_crimen_parte_a.py
		
		python pregunta_dos_crimen_parte_b.py
	
	Para evaluar el dataset de ventas de la tercera pregunta, hay que seguir los
	siguientes pasos desde la carpeta proyecto_1:
		
		python parseInput.py
		cd preguntas
		python pregunta_tres.py
	

