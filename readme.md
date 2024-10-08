# Buscador de películas semántico

Con este proyecto aprenderás las bases de git y docker. También tendrás un acercamiento al uso de vectores para NLP o procesamiento del lenguaje natural con hugging face, esta es la tecnología detrás de herramientas como chatgpt, Llama, o incluso google.

## Objetivos - Entrega preliminar

- Crear un repositorio haciendo uso de git.
- Crear un archivo readme explicando como ejecutar el proyecto.
- Subir los cambios necesarios al repositorio para ejecutar el proyecto.
- Crear un contenedor con docker para ejecutar por consola el proyecto.
    - ¿Cómo guardamos los datos luego de aplicar la similitud por coseno?
- Crear una función que cree una nueva columna que va a tener información relevante para los embeddings, ¿tal vez es importante tener el valor ganado de la película, o el nombre del director?.

### Rubrica
| Funcionalidad (2.5)   | Documentación    (2.5)   |
| ------------ | ------------ | 
| El código funciona según las instrucciones, desde un contenedor | Existe documentación clara en formato MarkDown de cómo ejecutar el proyecto. | 
| El proyecto está en un repositorio de git con acceso al profesor. | El código está correctamente documentado. | 
|Existe una función para aumentar el contexto de los embeddings | |
|El proyceto una vez iniciado permite realizar varias búsquedas y para su ejecución con un comando específico||

## Objetivos - Entrega final
- Crear pruebas unitarias.

- Usar las mejores prácticas teniendo en cuenta el nivel de acople y desacople de los módulos, manejo de errores, etc.
- Usar y documentar al menos un patrón de diseño.
- Documentar las funciones y el proyecto (estructura).

### Rúbrica
| Funcionalidad (2.0)   | Documentación    (0.5)   | Pruebas y calidad (1.5) | Estructura (1.0) |
| ------------ | ------------ | ------------ | ------------ | 
| El código funciona según las instrucciones, desde un contenedor docker | Existe documentación clara en formato MarkDown de cómo ejecutar el proyecto. | El test coverage es de **mínimo** 70% | El proyecto está estrucurdo en carpetas separadas lógicamente |
|  |  | Se usaron buenas prácticas en el código (patrones de diseño, principios solid, decoupling ) |
|||Hacer uso de excepciones para manejar posibles errores||


## Bonus (+ en la nota final de la entrega)
- Usar otro dataset diferente para hacer búsqueda semántica, pueden ser datasets de productos, de preguntas y respuestas, etc. ***+0.1***
- Crear un EDA completo de los datos usados, y **sustentarlo** en la clase en 10 minutos. ***+0.1***


## Super bonus (+ en la nota defenitiva de la materia)
- Hacer uso de una base de datos vectorial para realizar búsquedas, y **sustentarlo** en la clase en 10 minutos. ***+0.1***.

- Desplegar el contenedor a la nube para ser consumido desde cualquier lugar (por consola o por interfaz), y **sustentarlo** en la clase en 15 minutos. ***+0.1***.

---
> Nota: 
>* Una vez entregado algo no se reciben cambios. Si se hacen cambios en el repo **TODO** se califica sobre la norma (-0.5 por cada día de retraso).
>* Si no se entrega a tiempo, los bonus valen la mitad.
>* Sobre el bonus, si varias personas utilizan la misma base de datos vectorial, y justifican que *no hubo fraude*, **se dividirá el bonus entre la cantidad de personas**.