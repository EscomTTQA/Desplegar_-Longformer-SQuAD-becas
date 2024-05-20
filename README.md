# Desplegar_-Longformer-SQuAD-becas
El despliegue del modelo de aprendizaje automático para el proyecto BecarIA implica seguir una serie de pasos clave. Guardar el modelo, subir los archivos a un repositorio de Hugging Face y generar la API para su uso posterior.

# BecarIA: desplegar modelo de aprendizaje automático para su posterior uso como una API
Pasos para desplegar el modelo en Hugging Face que se usa para el proyecto BecarIA. [Prueba el modelo aquí](https://huggingface.co/BecarIA/Longformer-SQuAD-becas-1).
## ¿Cómo desplegar el modelo en Hugging Face?
El proceso consiste en guardar el modelo, que para el caso de este proyecto ya se ha hecho un ajuste fino a un modelo pre-entrenado llamado Longformer y tomado desde Hugging Face. Posteriormente, la forma en que se guarda el modelo genera archivos que contienen:
* La arquitectura del modelo.
* Los pesos del modelo.
* La configuración de entrenamiento.
* El estado del optimizador.

Luego, usar CLI (Command Line Interface) que proporciona Hugging Face y permite interactuar con Hugging Face Hub directamente desde una terminal. Desde ahí se inicia sesión con el token de acceso de usuario que se crea desde la página de [Hugging Face en el apartado de tokens](https://huggingface.co/settings/tokens). Una vez iniciada sesión en huggingface-cli se necesita crear un repositorio que almacenará los archivos del Longformer ajustado. Además, se crea un programa de Python que permite subir o cargar el modelo sobre el repositorio creado.

## Prerrequisitos
1. **Abrir Hugging Face e iniciar sesión**: en [Hugging Face](https://huggingface.co/join).
2. **Instalar `transformers` y `huggingface_hub`**:

```bash
pip install transformers huggingface_hub 
```
4. **Crear token de acceso de usuario**:
Desde la página de [Hugging Face en el apartado de tokens](https://huggingface.co/settings/tokens). Es importante que este token sea de tipo write.   
5. **Guardar el modelo**: 
Una vez que ya se ha ajustado el modelo, es necesario exponerlo, para eso hay que guardarlo y se realizó de la siguiente forma: 

```python
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
```

## Proceso para desplegar el modelo

### Iniciar sesión desde huggingface-cli
Abrir la terminal y escribir la siguiente linea:

```bash
huggingface-cli login
```
Luego, se pide el token de acceso, simplemente se copia desde la página de Hugging Face, se pega sobre la terminal y dar enter. Muestra algo como lo siguiente:

```bash
Token can be pasted using 'Right-Click'.
Enter your token (input will not be visible):
Add token as git credential? (Y/n) Y
Token is valid (permission: write).
Your token has been saved in your configured git credential helpers (manager,store).
Your token has been saved to C:\Users\user\.cache\huggingface\token
Login successful
```
    
Para saber que todo se ha iniciado correctamente se ejecuta:
    
```bash
huggingface-cli whoami
```
Eso te muestra el nombre de usuario de Hugging Face que inició sesión.

### Crear repositorio de Hugging Face:
Sobre el mismo CLI ejecutamos la siguiente línea:

```bash
huggingface-cli repo create <Nombre_del_repositorio>
```

Al hacerlo, muestra las siguientes líneas:
```bash
git version 2.40.0.windows.1
git-lfs/3.3.0 (GitHub; windows amd64; go 1.19.3; git 77deabdf)

You are about to create BecarIA/Longformer-SQuAD-becas-1
Proceed? [Y/n] Y

Your repo now lives at:
  https://huggingface.co/BecarIA/Longformer-SQuAD-becas-1

You can clone it locally with the command below, and commit/push as usual.

  git clone https://huggingface.co/BecarIA/Longformer-SQuAD-becas-1
```
  
El nombre del repositorio se tomará también como el nombre del modelo que se sube. Esto crea el repositorio y lo podemos visualizar al dar clic en el perfil (parte superior derecha de [Hugging Face](https://huggingface.co/)).

### Crear aplicación de Python para desplegar modelo sobre el repositorio creado
En la sección del perfil se podrá visualizar el repositorio y al dar clic se observa que está vacío. Se necesita copiar la ruta relativa de este repositorio, que se visualiza de esta forma: 

![image](https://github.com/EscomTTQA/Desplegar_-Longformer-SQuAD-becas/assets/167526018/3d5ae5dc-54dd-4218-aa83-1ea8f84b0426)


Una vez copiada se crea el archivo .py que permite subir el modelo al repositorio:

```python
from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import os

# Obtiene la ruta absoluta del directorio actual
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Carpeta del modelo dentro del directorio actual
MODEL_FOLDER = 'longformer-base-4096-spanish-finetuned-squad_9_10_trainer3'

# Ruta completa al modelo
MODEL_PATH = os.path.join(CURRENT_DIR, MODEL_FOLDER)

#Carga la arquitectura del modelo en una varable python
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_PATH)

#Sube los archivos relacionados a la arquitectura del modelo a la dirección del repositorio que se ha creado
model.push_to_hub("nombreUsuario/nombreRepositorio")

#Carga los pesos del modelo
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

#Sube los archivos relacionados a los pesos del modelo a la dirección del repositorio que se ha creado
tokenizer.push_to_hub("nombreUsuario/nombreRepositorio")
```
La ejecución de este código puede tardar algunos minutos. Luego, en la dirección del repositorio creado se pueden visualizar los archivos necesarios para que el modelo funcione. Incluso, debido a que la plataforma es muy robusta para el manejo y despliegue de modelos de Machine Learning y a los archivos que se suben, se puede probar el modelo desde Hugging Face:

![image](https://github.com/EscomTTQA/Desplegar_-Longformer-SQuAD-becas/assets/167526018/6f94d62d-2792-4ab8-8e17-9458aceae5f0)


Vemos como está trabajando el modelo para poder responder:

![image](https://github.com/EscomTTQA/Desplegar_-Longformer-SQuAD-becas/assets/167526018/19cbf76c-eb36-48be-b696-14be67d96425)


La respuesta que da:

![image](https://github.com/EscomTTQA/Desplegar_-Longformer-SQuAD-becas/assets/167526018/458db53a-4392-4a77-bb7e-2705a29d66a3)


Este proceso se describe también en: [Subiendo modelos](https://huggingface.co/docs/hub/models-uploading#uploading-models)

### Usar el modelo puesto en el repositorio como una API
#### Saber el endpoint de la API del modelo
Para poder hacer peticiones se requiere la URL a la que se le hacen las peticiones. Hugging Face indica que para eso podemos usar la ruta:
```
https://api-inference.huggingface.co/models/<MODEL_ID>
```
Donde `<MODEL_ID>` es la ruta relativa del repositorio donde está alojado el modelo.
Además, se indica que para modelos de QA (modelos que realizan tareas de pregunta y respuesta), como es el caso de este proyecto, la estructura o contenido que se debe seguir es:

```python
import requests
headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://api-inference.huggingface.co/models/nombreUsuario/nombreRepo"
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()
data = query(
    {
        "inputs": {
            "question": "What's my name?",
            "context": "My name is Clara and I live in Berkeley.",
        }
    }
)
```

La documentación del uso de la API se encuentra en: [Parámetros detallados](https://huggingface.co/docs/api-inference/en/detailed_parameters#question-answering-task)

#### Hacer peticiones a la API desde Postman
Para probar el modelo se requiere una aplicación o herramienta que permita el manejo de APIs. Para este ejemplo se usa Postman de la siguiente forma:
1. **Se configura la autorización**: con un tipo llamado "Bearer token" y luego, se coloca el token de acceso de usuario.
2. **Se coloca la URL de la API**:

```
https://api-inference.huggingface.co/models/<MODEL_ID>
```
3. **Se coloca el cuerpo del JSON**:

```
{
    "inputs": {
        "question": "¿Qué monto ofrece la Beca de Transporte Institucional B?",
        "context": "La Beca de Transporte Institucional B ofrece un monto de $2,500.00 por periodo escolar. La beca está dirigida a estudiantes con gastos de transporte de más de $400.00 hasta $800.00 mensuales relacionados con sus actividades académicas. Esta beca es compatible con el programa BEIFI.\nPara aplicar a la beca debes acceder a la plataforma Sibec, revisar las convocatoria y los requisitos específicos, completar la solicitud en línea y adjuntar la documentación requerida.\nLos requisitos para solicitar y renovar la beca son: ser ciudadano/a mexicano/a, estar inscrito en modalidad escolarizada en nivel superior, ser alumno regular o alumno irregular con un máximo de dos unidades de aprendizaje en situación de adeudo, tener un gasto mensual de transporte del domicilio a la unidad académica de más de $400.00 hasta $800.00 mensuales, provenir de una familia cuyos ingresos mensuales no superen los cuatro salarios mínimos per cápita vigentes al momento de la solicitud y es necesario contar con una cuenta bancaria activa en BBVA a su nombre, permitiendo recibir depósitos y transferencias electrónicas mayores al monto total de la beca, sin límite de depósitos al mes.\nLas convocatorias de becas se publican antes del inicio de cada semestre, generalmente en agosto. El proceso de selección de becas usualmente toma de 1 a 2 meses tras cerrar la convocatoria. La fecha de registro de solicitud de beca será del 25 de septiembre al 01 de octubre de 2023. La fecha de carga de documentación requerida en el SIBec será del 25 de septiembre al 01 de octubre de 2023. La fecha de validación de requisitos es del 02 al 23 de octubre de 2023 y la fecha de publicación de resultados el 13 de noviembre del 2023. La beca es renovable, ya que el periodo de duración abarca los dos periodos escolares del ciclo 2023-2024 (2024/1 y 2024/2), con validación al inicio del periodo escolar 2024/2.\nLa asignación de becas se prioriza por un gasto de más de $400.00 hasta $800.00 mensual de transporte. Además, se evalúa el promedio general del estudiante como parte del proceso de selección, residencia en zonas prioritarias, discapacidad visual, auditiva o motriz y determinación de afrodescendencia. En esta convocatoria, la priorización fundamental está determinada por la carga de créditos: mientras más carga de créditos falten por acreditar al alumno, mayor consideración tendrá.\nLos documentos requeridos para la solicitud son los siguientes: el acuse de la solicitud de beca y la carta compromiso expedida por el SIBec, formato de reporte de ingresos y egresos del aspirante y de los familiares/responsables de la manutención del aspirante, expedido por el sistema y debidamente requisitado, acompañado de comprobantes de percepciones y documentos de identificación de los firmantes. Los estudiantes de nuevo ingreso deben presentar el certificado de estudios del nivel educativo anterior, y todos los solicitantes deben proporcionar un comprobante de domicilio (no mayor a 3 meses), una identificación oficial (INE o credencial expedida por la DAE con firma). Además, la Dirección de Administración Escolar anexará al SIBec una constancia de becas donde se especifique el promedio general del estudiante (generada por gestión escolar). Toda la documentación debe ser escaneada a color (no fotos) y cargada electrónicamente en el Sistema de Becas del IPN (SIBec).\nLa documentación debe ser nombrada de la siguiente manera: Boleta_TipoDeBecaSolicitada.pdf (por ejemplo: 2030123456_Institucional.pdf). Todos los documentos deben ser consolidados en un solo archivo PDF con un tamaño máximo de 3 MB y una resolución de 150 dpi.\nRecibirás la notificación de aceptación de la beca a través del SIBec. Después de que se publiquen los resultados, deberás ingresar al SIBec para confirmar tu selección y completar los pasos necesarios para el registro bancario. El monto de la beca se deposita en la cuenta bancaria registrada en la plataforma. Los fondos de la beca se distribuyen normalmente antes de que finalice el semestre en curso, generalmente poco antes del fin del mismo.\nLos motivos para perder la beca incluyen no actualizar los datos bancarios en el SIBec, no registrar la cuenta bancaria a tiempo, rechazo de depósitos de beca, renuncia del beneficiario, proporcionar información falsa, incumplir las normas de la institución, el fallecimiento del estudiante, dejar de ser alumno regular o cambiar de modalidad de estudio.\nSi tu solicitud de beca es rechazada, puedes solicitar una reconsideración siguiendo lo establecido en el Reglamento General de Becas del Instituto Politécnico Nacional. El recurso de consideración permite a los estudiantes presentar argumentos o documentación adicional para reevaluar su solicitud. Para iniciar este proceso, debes seguir los procedimientos y cumplir con los plazos especificados en el reglamento.\nUna vez que se ha completado y enviado la solicitud de beca, los datos proporcionados no pueden ser modificados en la plataforma SIBec. Antes de enviar tu solicitud es importante revisar todos los datos y documentos, ya que cualquier información incorrecta o incompleta podría influir en la decisión final respecto a tu beca.\nLa información detallada sobre la beca se puede encontrar en la Gaceta Politécnica Convocatoria General de Becas: https://www.InstitutoPolitecnicoNacional.mx/daes/servicios/becas.html. Además, información actualizada y anuncios se pueden encontrar en las página de Facebook Becas ESCOM: https://www.facebook.com/BecasEscom."
    }
}
```
4. **Se configura el Header correcto**:
```
key = Content-Type
value = application/json
```

![image](https://github.com/EscomTTQA/Desplegar_-Longformer-SQuAD-becas/assets/167526018/283ae97d-2f20-4676-a0eb-4f074c530d85)


Como resultado se obtiene:

![image](https://github.com/EscomTTQA/Desplegar_-Longformer-SQuAD-becas/assets/167526018/a788bba2-a85f-4bc0-80d1-893dd0e9363b)


