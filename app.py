import streamlit as st
import pandas as pd
from pydantic import BaseModel, Field
from typing import List
import io
import os

# --- Librerías de PDF y Gemini ---
from pdfminer.six import extract_text_to_fp, PDFPage, PDFResourceManager, PDFDevice, process_pdf
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams

from google import genai
from google.genai import types
from google.genai.errors import APIError

# --- 1. MODELO ESTRUCTURADO DE DATOS (PYDANTIC) ---
# Define el esquema que garantiza la estructura de los datos extraídos por Gemini (Las 28 columnas)

class AvaluoData(BaseModel):
    """Schema para la extracción estructurada de datos de avalúos."""
    n_avaluo: str = Field(description="N AVALUO: Número de Avalúo (ej: VAL2023-217).")
    anio: str = Field(description="AÑO: Año del informe.")
    provincia: str = Field(description="PROVINCIA.")
    parroquia: str = Field(description="PARROQUIA.")
    direccion_inmueble: str = Field(description="DIRECCION del inmueble a avaluar.")
    descripcion: str = Field(description="DESCRIPCION: Tipo de bien (ej: VIVIENDA, DEPARTAMENTO, TERRENO URBANO).")
    area_terreno: float = Field(description="AREA DE TERRENO: M2 de terreno.")
    area_construccion: float = Field(description="AREA DE CONSTRUCCION: Suma de la columna cantidad (excluyendo el valor de la fila terreno).")
    val_terreno_m2: float = Field(description="VAL. TERRENO C/M2: Valor unitario del terreno.")
    valor_mercado_construccion: float = Field(description="VALOR DE MERCADO CONSTRUCCIONES (Dato para el cálculo de J).")
    val_comercial_inmueble: float = Field(description="VAL. COMERCIAL DEL INMUEBLE: Valor total de mercado.")

    # Referencia 1
    ref1_dir: str = Field(description="REF(1) DIRECCION: Dirección de la primera referencia.")
    ref1_contacto: str = Field(description="REF(1) NUM CONTACTO: Número de contacto de la primera referencia.")
    ref1_area_constr: float = Field(description="REF(1) AREA DE CONTRUCCION: Área de construcción de la primera referencia.")
    ref1_val_terreno: float = Field(description="REF(1) VAL. TERRENO: Valor comercial del terreno de la primera referencia.")
    ref1_val_terreno_m2: float = Field(description="REF(1) VAL. TERRENO C/M2: Valor/m² terreno de la primera referencia.")

    # Referencia 2
    ref2_dir: str = Field(description="REF(2) DIRECCION: Dirección de la segunda referencia.")
    ref2_contacto: str = Field(description="REF(2) NUM CONTACTO: Número de contacto de la segunda referencia.")
    ref2_area_constr: float = Field(description="REF(2) AREA DE CONTRUCCION: Área de construcción de la segunda referencia.")
    ref2_val_terreno: float = Field(description="REF(2) VAL. TERRENO: Valor comercial del terreno de la segunda referencia.")
    ref2_val_terreno_m2: float = Field(description="REF(2) VAL. TERRENO C/M2: Valor/m² terreno de la segunda referencia.")

    # Referencia 3
    ref3_dir: str = Field(description="REF(3) DIRECCION: Dirección de la tercera referencia.")
    ref3_contacto: str = Field(description="REF(3) NUM CONTACTO: Número de contacto de la tercera referencia.")
    ref3_area_constr: float = Field(description="REF(3) AREA DE CONTRUCCION: Área de construcción de la tercera referencia.")
    ref3_val_terreno: float = Field(description="REF(3) VAL. TERRENO: Valor comercial del terreno de la tercera referencia.")
    ref3_val_terreno_m2: float = Field(description="REF(3) VAL. TERRENO C/M2: Valor/m² terreno de la tercera referencia.")

    # Geografía
    latitud: float = Field(description="LATITUD: Coordenada geográfica de latitud.")
    longitud: float = Field(description="LONGITUD: Coordenada geográfica de longitud.")

# --- 2. FUNCIONES DE LÓGICA CENTRAL ---

def get_pdf_text(uploaded_file):
    """Extrae texto plano de un archivo PDF subido (usando pdfminer.six)."""
    output_string = io.StringIO()
    manager = PDFResourceManager()
    converter = TextConverter(manager, output_string, laparams=LAParams())

    uploaded_file.seek(0) # Vuelve al inicio del archivo
    with io.BytesIO(uploaded_file.read()) as f:
        process_pdf(manager, converter, f)

    converter.close()
    return output_string.getvalue()


@st.cache_data(show_spinner="Procesando avalúo con Gemini... esto puede tardar unos segundos.")
def extract_data_from_pdf_and_cache(pdf_bytes, file_name, model_schema):
    """
    Función CACHEADA que llama a la API de Gemini para la extracción estructurada.
    """
    # La clave se lee del archivo secrets.toml, inyectada en os.environ
    if 'GEMINI_API_KEY' not in os.environ:
        st.error("Error: La clave GEMINI_API_KEY no está configurada. Verifica los Secrets de Streamlit.")
        return None, None

    try:
        client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
        model = 'gemini-2.5-flash'

        pdf_text = get_pdf_text(pdf_bytes)

        # --- PROMPT DE INSTRUCCIONES CRÍTICAS ---
        prompt_instructions = f"""
        Eres un experto extractor de datos. Lee el texto del informe de avalúo y extrae la información requerida, ajustándola EXACTAMENTE al formato JSON del esquema proporcionado.

        **Instrucciones de Extracción Específicas:**
        1.  **N AVALUO y AÑO:** Extraer de la codificación y del campo AÑO.
        2.  **DESCRIPCION:** Solo usar las palabras VIVIENDA, DEPARTAMENTO, INMUEBLE COMERCIAL, TERRENO URBANO, TERRENO RURAL, TERRENO AGRICOLA, EDIFICIO, CASA.
        3.  **AREA CONSTRUCCION:** Es la suma de la columna 'cantidad' de la sección L. valoración, excluyendo la fila 'terreno'.
        4.  **Referencias (REF1, REF2, REF3):** Extraer los datos de las tres tablas del subtema K. REFERENCIAS PARA VALORACIÓN.
        5.  Extrae valores numéricos sin comas ni símbolos de moneda (ej: 1250000.50).

        --- TEXTO DEL DOCUMENTO DE AVALÚO ---
        {pdf_text}
        """

        # Llamada a la API con respuesta estructurada JSON
        response = client.models.generate_content(
            model=model,
            contents=prompt_instructions,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=model_schema,
            ),
        )

        extracted_data = AvaluoData.model_validate_json(response.text).model_dump()
        return extracted_data, pdf_text

    except APIError as e:
        st.error(f"Error de la API de Gemini: {e}. Verifica clave o cuotas.")
        return None, None
    except Exception as e:
        st.error(f"Error de procesamiento del archivo o parsing JSON: {e}")
        return None, None


def main():
    """Configuración de la interfaz de usuario de Streamlit."""

    # --- Configuración Inicial ---
    st.set_page_config(layout="wide", page_title="IDP de Avalúos con Gemini")
    st.title("🤖 IDP de Avalúos con Gemini")
    st.markdown("Herramienta centralizada para la extracción, almacenamiento y búsqueda de datos de informes PDF.")
    st.divider()

    # Obtiene las columnas finales de Excel del modelo Pydantic
    COLUMNAS_FINALES = [
        "N AVALUO", "AÑO", "PROVINCIA", "PARROQUIA", "DIRECCION", "DESCRIPCION", 
        "AREA DE TERRENO", "AREA DE CONSTRUCCION", "VAL. TERRENO C/M2", "VAL. CONSTRUCCION C/M2", 
        "VAL. COMERCIAL DEL INMUEBLE", "REF(1) DIRECCION", "REF(1) NUM CONTACTO", 
        "REF(1) AREA DE CONSTRUCCION", "REF(1) VAL. TERRENO", "REF(1) VAL. TERRENO C/M2", 
        "REF(2) DIRECCION", "REF(2) NUM CONTACTO", "REF(2) AREA DE CONSTRUCCION", 
        "REF(2) VAL. TERRENO", "REF(2) VAL. TERRENO C/M2", "REF(3) DIRECCION", 
        "REF(3) NUM CONTACTO", "REF(3) AREA DE CONSTRUCCION", "REF(3) VAL. TERRENO", 
        "REF(3) VAL. TERRENO C/M2", "LATITUD", "LONGITUD"
    ]

    # --- Inicialización de la Base de Datos en Session State ---
    if 'data_db' not in st.session_state:
        st.session_state.data_db = pd.DataFrame(columns=COLUMNAS_FINALES)
    if 'rag_corpus' not in st.session_state:
        st.session_state.rag_corpus = {}

    # --- Lógica de Carga y Extracción ---
    with st.sidebar:
        st.header("1. Cargar y Procesar")
        uploaded_files = st.file_uploader(
            "Sube uno o más PDFs (Carga por lote)",
            type=["pdf"],
            accept_multiple_files=True
        )

        if st.button("▶️ Iniciar Extracción"):
            if not uploaded_files:
                st.warning("Sube al menos un PDF para comenzar.")
            else:
                new_db_entries = []
                new_rag_entries = {}
                progress_bar = st.progress(0, text="Iniciando procesamiento...")

                for i, file in enumerate(uploaded_files):
                    progress_text = f"Procesando archivo {i + 1} de {len(uploaded_files)}: {file.name}"
                    progress_bar.progress((i + 1) / len(uploaded_files), text=progress_text)

                    # Llamada a la función CACHEADA
                    extracted_data, full_text = extract_data_from_pdf_and_cache(file, file.name, AvaluoData)

                    if extracted_data:
                        # 1. Realizar el cálculo de VAL. CONSTRUCCION C/M2 (Columna J)
                        try:
                            area_total_constr = extracted_data.pop('area_construccion') 
                            val_mercado_constr = extracted_data.pop('valor_mercado_construccion') 

                            val_constr_m2 = round(val_mercado_constr / area_total_constr, 2) if area_total_constr and area_total_constr > 0 else 0.0
                        except Exception:
                            val_constr_m2 = 0.0 # Usar 0.0 si falla el cálculo
                            area_total_constr = extracted_data.get('area_construccion', 0.0)


                        # 2. Reestructurar los datos al orden final del Excel
                        datos_finales = {
                            "N AVALUO": extracted_data.get('n_avaluo', ''), "AÑO": extracted_data.get('anio', ''), "PROVINCIA": extracted_data.get('provincia', ''), 
                            "PARROQUIA": extracted_data.get('parroquia', ''), "DIRECCION": extracted_data.get('direccion_inmueble', ''), "DESCRIPCION": extracted_data.get('descripcion', ''),
                            "AREA DE TERRENO": extracted_data.get('area_terreno', 0.0), "AREA DE CONSTRUCCION": area_total_constr, 
                            "VAL. TERRENO C/M2": extracted_data.get('val_terreno_m2', 0.0), "VAL. CONSTRUCCION C/M2": val_constr_m2, 
                            "VAL. COMERCIAL DEL INMUEBLE": extracted_data.get('val_comercial_inmueble', 0.0), 
                            "REF(1) DIRECCION": extracted_data.get('ref1_dir', ''), "REF(1) NUM CONTACTO": extracted_data.get('ref1_contacto', ''), 
                            "REF(1) AREA DE CONSTRUCCION": extracted_data.get('ref1_area_constr', 0.0), "REF(1) VAL. TERRENO": extracted_data.get('ref1_val_terreno', 0.0), 
                            "REF(1) VAL. TERRENO C/M2": extracted_data.get('ref1_val_terreno_m2', 0.0), 
                            "REF(2) DIRECCION": extracted_data.get('ref2_dir', ''), "REF(2) NUM CONTACTO": extracted_data.get('ref2_contacto', ''), 
                            "REF(2) AREA DE CONSTRUCCION": extracted_data.get('ref2_area_constr', 0.0), "REF(2) VAL. TERRENO": extracted_data.get('ref2_val_terreno', 0.0), 
                            "REF(2) VAL. TERRENO C/M2": extracted_data.get('ref2_val_terreno_m2', 0.0), 
                            "REF(3) DIRECCION": extracted_data.get('ref3_dir', ''), "REF(3) NUM CONTACTO": extracted_data.get('ref3_contacto', ''), 
                            "REF(3) AREA DE CONSTRUCCION": extracted_data.get('ref3_area_constr', 0.0), "REF(3) VAL. TERRENO": extracted_data.get('ref3_val_terreno', 0.0), 
                            "REF(3) VAL. TERRENO C/M2": extracted_data.get('ref3_val_terreno_m2', 0.0), 
                            "LATITUD": extracted_data.get('latitud', 0.0), "LONGITUD": extracted_data.get('longitud', 0.0)
                        }

                        new_db_entries.append(datos_finales)
                        key = extracted_data.get('n_avaluo', file.name)
                        new_rag_entries[key] = full_text

                # 3. Actualizar la base de datos
                if new_db_entries:
                    new_df = pd.DataFrame(new_db_entries, columns=COLUMNAS_FINALES)
                    st.session_state.data_db = pd.concat([st.session_state.data_db, new_df], ignore_index=True)
                    st.session_state.rag_corpus.update(new_rag_entries)
                    st.success(f"🎉 Éxito: Se procesaron {len(new_db_entries)} informe(s).")

                progress_bar.empty()

    # --- Sección de Resultados y Descarga ---
    st.header("Base de Datos Procesada")

    if st.session_state.data_db.empty:
        st.info("No hay datos. Sube y procesa los archivos en la barra lateral.")
    else:
        st.dataframe(st.session_state.data_db, use_container_width=True)

        @st.cache_data
        def convert_df_to_excel(df):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Avalúos')
            return output.getvalue()

        excel_data = convert_df_to_excel(st.session_state.data_db)

        st.download_button(
            label="💾 Descargar Base de Datos (Excel)",
            data=excel_data,
            file_name="Base_de_Datos_Avalúos.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.divider()

    # --- Sección de Búsqueda RAG ---
    st.header("2. Búsqueda Inteligente (RAG)")
    rag_query = st.text_input(
        "Haz una pregunta abierta (ej: ¿Cuál fue el valor unitario de la construcción del último avalúo procesado?)",
    )

    if rag_query and st.session_state.rag_corpus:

        # 1. Recuperación del contexto (simple: se envía todo el texto de los avalúos)
        full_context = "\n---\n".join(st.session_state.rag_corpus.values())

        rag_prompt = f"""
        Basándote ÚNICAMENTE en el siguiente CONTEXTO de informes, responde a la pregunta.
        Si la información no está en el contexto, indica que no pudiste encontrarla.

        --- PREGUNTA ---
        {rag_query}

        --- CONTEXTO ---
        {full_context}
        """

        try:
            client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
            rag_response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=rag_prompt
            )

            st.markdown("### Respuesta Generada por Gemini")
            st.markdown(rag_response.text)

        except Exception as e:
            st.error(f"Error en la búsqueda RAG: {e}")

if __name__ == '__main__':
    main()
