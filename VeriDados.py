# -*- coding: utf-8 -*-
"""
Ferramenta de Auditoria e Comparação de Extrações de Dados
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
import csv 
from sqlalchemy import create_engine, text
import io
import re
from io import BytesIO 
from typing import List, Dict, Any, Optional, Tuple, Set

# --- Configuração da Página Streamlit ---
st.set_page_config(layout="wide", page_title="VeriDados", page_icon=":bar_chart:")

st.title("Ferramenta de Auditoria e Comparação de Dados")
st.markdown("""
**Atenção:** Ao carregar e analisar dados, especialmente se forem sensíveis, assegure-se de que tem as permissões necessárias e de que cumpre todas as políticas de segurança, privacidade e confidencialidade de dados relevantes da sua organização ou contexto.
""")

# --- Inicialização de Flags e Estados no Session State (no topo do script) ---
if 'viewing_all_keys_description' not in st.session_state:
    st.session_state.viewing_all_keys_description = None
if 'all_keys_data_to_show' not in st.session_state:
    st.session_state.all_keys_data_to_show = None
if 'expanded_view_rendered_this_cycle' not in st.session_state:
    st.session_state.expanded_view_rendered_this_cycle = False

# --- Funções Auxiliares de Carregamento e Análise Individual ---

def make_unique_columns(columns: List[str]) -> List[str]:
    """
    Garante que os nomes das colunas em uma lista sejam únicos,
    adicionando sufixos como .1, .2 etc. a nomes duplicados.
    """
    seen = {}
    new_cols = []
    for col_name in columns:
        original_col_name = col_name # Salva o nome original para o caso de ser um placeholder
        if not isinstance(col_name, str) or not col_name.strip(): # Se for placeholder ou não string
            # Não tenta adicionar sufixo a placeholders já numerados se não for necessário
            # Mas se o placeholder em si estiver duplicado, ele será numerado
            pass

        if col_name not in seen:
            seen[col_name] = 0
            new_cols.append(col_name)
        else:
            seen[col_name] += 1
            new_cols.append(f"{original_col_name}.{seen[col_name]}")
    return new_cols

def load_data(uploaded_file: Optional[st.runtime.uploaded_file_manager.UploadedFile]) -> Optional[pd.DataFrame]:
    if uploaded_file is None:
        return None
    file_name = uploaded_file.name
    df = None
    # Definir temp_db_path aqui para que esteja no escopo do finally principal da função
    temp_db_path = None 
    # Flag para saber se o arquivo .db temporário foi criado e precisa ser removido
    db_temp_file_created = False

    try:
        if file_name.endswith(".csv"):
            # ... (seu código para CSV aqui, como já está e funcionando) ...
            # Cole o seu bloco de código para CSV aqui
            uploaded_file.seek(0)
            content_as_bytes = uploaded_file.getvalue()
            detected_sep_info = ""
            final_sep_used = None
            try:
                sample_text_sniffer = content_as_bytes[:8192].decode('utf-8', errors='replace')
                dialect = csv.Sniffer().sniff(sample_text_sniffer)
                detected_sep = dialect.delimiter
                detected_sep_info = f"CSV Sniffer sugeriu o delimitador: '{detected_sep}'. "
            except (csv.Error, UnicodeDecodeError) as e:
                detected_sep_info = f"CSV Sniffer não pôde determinar o delimitador (erro: {e}). "
                detected_sep = None
            potential_separators = []
            if detected_sep:
                potential_separators.append(detected_sep)
            for s_common in ['|', ';', ',', '\t']: 
                if s_common not in potential_separators:
                    potential_separators.append(s_common)
            last_parser_error = None
            for sep_candidate in potential_separators:
                try:
                    file_stream = io.BytesIO(content_as_bytes)
                    current_df_attempt = pd.read_csv(file_stream, sep=sep_candidate, engine='python', quoting=csv.QUOTE_NONE, on_bad_lines='warn')
                    if not current_df_attempt.empty:
                        if current_df_attempt.shape[1] > 1: 
                            df = current_df_attempt
                            final_sep_used = sep_candidate
                            break 
                        elif df is None: 
                            df = current_df_attempt
                            final_sep_used = sep_candidate
                    elif df is None : 
                        df = current_df_attempt 
                        final_sep_used = sep_candidate
                except pd.errors.ParserError as e:
                    last_parser_error = e
                except Exception as e: 
                    last_parser_error = e
            if df is not None and not df.empty and final_sep_used:
                if df.shape[1] > 1 and all(str(col).startswith("Unnamed:") for col in df.columns):
                    st.warning(f"CSV '{file_name}' carregado com delimitador '{final_sep_used}', mas os nomes das colunas sugerem que o delimitador pode estar incorreto (Ex: {df.columns[:3].tolist()}).")
                elif df.shape[1] > 1 :
                    st.success(f"CSV '{file_name}' carregado com delimitador '{final_sep_used}'. Colunas: {df.shape[1]}.")
                else: 
                    st.info(f"CSV '{file_name}' carregado com delimitador '{final_sep_used}' (1 coluna). Verifique se este é o delimitador correto.")
            elif df is not None and df.empty and final_sep_used:
                 st.warning(f"CSV '{file_name}' carregado com delimitador '{final_sep_used}', mas resultou em DataFrame vazio.")
            elif df is None: 
                err_msg = f"Falha ao carregar CSV '{file_name}' após tentar os delimitadores: {potential_separators}."
                if last_parser_error: err_msg += f" Último erro de parser: {last_parser_error}"
                st.error(err_msg)
                return None


        elif file_name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file, engine="openpyxl")
            st.success(f"Arquivo Excel '{file_name}' carregado com sucesso.")
        elif file_name.endswith(".parquet"): 
            df = pd.read_parquet(uploaded_file, engine="pyarrow")
            st.success(f"Arquivo Parquet '{file_name}' carregado com sucesso.")
        elif file_name.endswith(".db"):
            temp_db_path = f"./temp_{file_name}" # Define o caminho
            db_temp_file_created = True # Marca que o arquivo será criado

            with open(temp_db_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            conn_str = f"sqlite:///{temp_db_path}"
            engine_db = None # Inicializa engine_db como None
            try:
                engine_db = create_engine(conn_str)
                query = text("SELECT name FROM sqlite_master WHERE type='table';")
                with engine_db.connect() as connection:
                    table_names_df = pd.read_sql_query(query, connection)
                
                if not table_names_df.empty:
                    if len(table_names_df) > 1:
                        table_to_load = st.selectbox(
                            f"Selecione a tabela para carregar do arquivo '{file_name}':", 
                            table_names_df["name"], 
                            key=f"db_table_select_{file_name}"
                        )
                    else:
                        table_to_load = table_names_df["name"][0]
                    
                    if table_to_load:
                        df = pd.read_sql_table(table_to_load, engine_db)
                        st.success(f"Tabela '{table_to_load}' do arquivo .db '{file_name}' carregada com sucesso.")
                    else:
                        st.error(f"Nenhuma tabela selecionada ou disponível no arquivo .db: {file_name}")
                        # Não precisa 'return None' aqui, o 'finally' cuidará da limpeza se necessário
                        # e o 'df' continuará None, levando ao tratamento de erro no final da função geral.
                else:
                    st.error(f"Nenhuma tabela encontrada no arquivo .db: {file_name}")
            except Exception as e_db_ops:
                st.error(f"Erro ao operar com o banco de dados SQLite '{file_name}': {e_db_ops}")
                df = None # Garante que df seja None
            finally:
                if engine_db: # Só chama dispose se a engine foi criada
                    engine_db.dispose()
            
            # Se df ainda é None após tentar carregar o .db, significa que algo falhou
            if df is None:
                # Mensagem de erro mais genérica, pois os erros específicos já foram mostrados
                # st.error(f"Falha ao processar ou carregar dados do arquivo .db: {file_name}") # Opcional
                return None # Retorna None para indicar falha no carregamento do .db

        else: # Formato não suportado
            st.error(f"Formato de arquivo não suportado: {file_name}. Formatos suportados: .csv, .xlsx, .xls, .parquet, .db")
            return None
        
        # --- Processamento comum para todos os DataFrames carregados com sucesso ---
        if df is not None:
            new_column_names = []
            for i, col in enumerate(df.columns):
                if isinstance(col, str) and col.strip():
                    new_column_names.append(col.strip())
                else:
                    new_column_names.append(f"coluna_sem_nome_{i}")
            df.columns = new_column_names
            df.columns = make_unique_columns(df.columns.tolist())
        
        return df # Retorna o DataFrame processado (ou None se falhou antes)

    except Exception as e_outer: # Captura qualquer outro erro não previsto no carregamento
        st.error(f"Erro crítico inesperado ao carregar o arquivo {file_name}: {e_outer}")
        # st.exception(e_outer) # Para depuração
        return None
    finally:
        # Bloco finally para remover o arquivo .db temporário, se foi criado
        if db_temp_file_created and temp_db_path and os.path.exists(temp_db_path):
            try:
                os.remove(temp_db_path)
                # st.info(f"Arquivo temporário '{temp_db_path}' removido.") # Opcional
            except OSError as e_remove:
                st.warning(f"Não foi possível remover o arquivo temporário '{temp_db_path}': {e_remove}")

def get_distinct_values_report(df: pd.DataFrame, column: str, limit: int = 30, 
                               analyze_as_multivalue: bool = False, 
                               internal_delimiter: str = '|') -> Tuple[np.ndarray, int]:
    if column not in df.columns:
        st.warning(f"Coluna '{column}' não encontrada no DataFrame.")
        return np.array([]), 0

    series_to_process = df[column].copy() 

    if analyze_as_multivalue and internal_delimiter:
        series_to_process = series_to_process.fillna('').astype(str)
        
        def split_and_strip_for_distinct(text, delim):
            if not text.strip(): return []
            return [item.strip() for item in text.split(delim)]

        exploded_series = series_to_process.apply(lambda x: split_and_strip_for_distinct(x, internal_delimiter)).explode()
        processed_items = exploded_series[exploded_series.str.strip() != '']
        
        if processed_items.empty:
            return np.array([]), 0
        distinct_values = processed_items.unique()
    else:
        processed_items = series_to_process.dropna()
        if processed_items.empty:
            return np.array([]), 0
        # Tenta converter para string para evitar erro com tipos mistos no unique()
        try:
            distinct_values = processed_items.astype(str).unique()
        except Exception: # Fallback se a conversão falhar (improvável após dropna, mas seguro)
            distinct_values = processed_items.unique()
            
    count = len(distinct_values)
    if hasattr(distinct_values, 'to_numpy'):
        distinct_values_np = distinct_values.to_numpy()
    else: 
        distinct_values_np = np.asarray(distinct_values)
        
    return distinct_values_np[:limit if count > limit else count], count

def get_null_empty_info_df(df: pd.DataFrame) -> pd.DataFrame:
    null_info_list = []
    for col in df.columns:
        null_count = df[col].isnull().sum()
        # Considera string vazia ou apenas espaços como "vazio"
        empty_count = df[col].apply(lambda x: isinstance(x, str) and not x.strip()).sum()
        total_null_empty = null_count + empty_count
        total_rows = len(df)
        percentage = (total_null_empty / total_rows) * 100 if total_rows > 0 else 0
        if total_null_empty > 0:
            null_info_list.append({
                "Coluna": col,
                "Nulos/Vazios": total_null_empty, # Nome da coluna simplificado
                "Percentual (%)": round(percentage, 2)
            })
    if not null_info_list:
        return pd.DataFrame(columns=["Coluna", "Nulos/Vazios", "Percentual (%)"])
    return pd.DataFrame(null_info_list)

def get_capitalization_inconsistencies_report(df: pd.DataFrame) -> Dict[str, Dict[str, List[str]]]:
    all_inconsistencies: Dict[str, Dict[str, List[str]]] = {}
    for col in df.select_dtypes(include=['object', 'string']).columns:
        series = df[col].dropna().astype(str)
        # Ignora strings vazias ou que só contêm espaços da análise de capitalização
        series_cleaned = series[series.str.strip() != '']
        if series_cleaned.empty:
            continue
            
        value_counts_lower = series_cleaned.str.lower().value_counts()
        inconsistent_groups: Dict[str, List[str]] = {}
        for val_lower, count in value_counts_lower.items():
            if not val_lower.strip(): # Ignora se o valor em minúsculo for vazio (após strip)
                continue
            original_values = series_cleaned[series_cleaned.str.lower() == val_lower].unique()
            if len(original_values) > 1: # Mais de uma variação original para o mesmo valor minúsculo
                inconsistent_groups[val_lower] = sorted(list(original_values))
        if inconsistent_groups:
            all_inconsistencies[col] = inconsistent_groups
    return all_inconsistencies

def get_date_format_inconsistencies_report(df: pd.DataFrame) -> Dict[str, List[str]]:
    all_formats: Dict[str, List[str]] = {}
    # Regexes para formatos de data comuns (incluindo hora opcional)
    # YYYY-MM-DD ou YYYY/MM/DD
    # DD-MM-YYYY ou DD/MM/YYYY
    # MM-DD-YYYY ou MM/DD/YYYY
    # Adicionados formatos com hora HH:MM:SS e HH:MM
    patterns = {
        "YYYY-MM-DD": r"^\d{4}-\d{2}-\d{2}( \d{2}:\d{2}(:\d{2})?)?$",
        "YYYY/MM/DD": r"^\d{4}/\d{2}/\d{2}( \d{2}:\d{2}(:\d{2})?)?$",
        "DD/MM/YYYY": r"^\d{2}/\d{2}/\d{4}( \d{2}:\d{2}(:\d{2})?)?$",
        "DD-MM-YYYY": r"^\d{2}-\d{2}-\d{4}( \d{2}:\d{2}(:\d{2})?)?$",
        "MM/DD/YYYY": r"^\d{2}/\d{2}/\d{4}( \d{2}:\d{2}(:\d{2})?)?$",
        "MM-DD-YYYY": r"^\d{2}-\d{2}-\d{4}( \d{2}:\d{2}(:\d{2})?)?$", # Corrigido o regex original para MM-DD-YYYY
    }
    for col in df.select_dtypes(include=['object', 'string']).columns:
        # Processar apenas valores não nulos e não vazios
        series = df[col].dropna()
        series = series[series.astype(str).str.strip() != ''] # Remove strings vazias ou só com espaços
        
        if series.empty:
            continue

        formats_found_in_col: Set[str] = set()
        # Tenta converter para datetime para uma verificação mais robusta
        try:
            pd.to_datetime(series, errors='raise')
            # Se chegou aqui, todos os valores são convertíveis para data por pd.to_datetime (com seu parser flexível)
            # Isso não necessariamente significa que têm o *mesmo* formato visual.
            # Continuamos com a análise de regex para formatos visuais.
        except (ValueError, TypeError):
            # Se pd.to_datetime falha, indica que a coluna não é consistentemente uma data ou tem formatos muito variados
            # ou não é data de forma alguma. A análise por regex abaixo tentará pegar padrões.
            pass

        # Heurística: se pelo menos 5% dos valores não nulos e não vazios corresponderem a algum padrão de data
        # Isso ajuda a focar em colunas que provavelmente contêm datas.
        num_potential_dates = sum(series.astype(str).apply(lambda x: any(re.match(p, x) for p in patterns.values())))
        
        if len(series) > 0 and (num_potential_dates / len(series)) > 0.05:
            for item in series.unique(): # Analisa apenas valores únicos para eficiência
                s_item = str(item).strip()
                if not s_item: continue # Pula se o item único for uma string vazia

                matched_fmt = False
                for fmt_name, pattern in patterns.items():
                    if re.match(pattern, s_item):
                        formats_found_in_col.add(fmt_name)
                        matched_fmt = True
                        break # Para no primeiro padrão que corresponder para este item
                # Se nenhum dos padrões principais corresponder, mas pd.to_datetime conseguiria parsear, 
                # podemos adicionar um formato genérico "Outro (parseável)"
                if not matched_fmt:
                    try:
                        pd.to_datetime(s_item)
                        formats_found_in_col.add("Outro (parseável por Pandas)")
                    except (ValueError, TypeError):
                        pass # Não é parseável, não adiciona

            if len(formats_found_in_col) > 1: # Se mais de um formato visual foi detectado
                all_formats[col] = sorted(list(formats_found_in_col))
    return all_formats

def get_leading_trailing_spaces_report(df: pd.DataFrame) -> Dict[str, List[str]]:
    all_spaces: Dict[str, List[str]] = {}
    for col in df.select_dtypes(include=['object', 'string']).columns:
        # Processa apenas strings não nulas
        series = df[col].dropna().astype(str)
        # Filtra valores que são diferentes após o strip(), mas não são apenas espaços em branco
        space_issues = series[(series.str.strip() != series) & (series.str.strip() != '')].unique().tolist()
        if space_issues:
            all_spaces[col] = space_issues[:10] # Amostra de até 10 valores
    return all_spaces

def get_basic_stats_df(df: pd.DataFrame) -> pd.DataFrame:
    stats_list = []
    numeric_cols = df.select_dtypes(include=np.number).columns
    if not numeric_cols.empty:
        for col in numeric_cols:
            if df[col].notna().sum() == 0: # Ignora colunas que são todas NaN
                continue
            
            # Define as funções de agregação padrão
            agg_functions = ["count", "mean", "min", "max", "std"]
            
            # Realiza as agregações padrão
            stats_series = df[col].agg(agg_functions)
            stats = stats_series.to_dict()
            
            # Calcula e adiciona a contagem de nulos separadamente
            stats['nulos'] = df[col].isnull().sum()
            
            stats["Coluna"] = col # Adiciona o nome da coluna ao dicionário de estatísticas

            # Arredonda os valores float
            for key, value in stats.items():
                if isinstance(value, (float, np.floating)): 
                    stats[key] = round(value, 2)
            stats_list.append(stats)
            
        if not stats_list: # Caso todas as colunas numéricas sejam apenas NaN
            return pd.DataFrame(columns=["Coluna", "count", "mean", "min", "max", "std", "nulos"])    
        
        # Garante a ordem das colunas no DataFrame final
        # Certifique-se de que todas as chaves esperadas existem antes de tentar reordenar
        final_df = pd.DataFrame(stats_list)
        expected_cols = ["Coluna", "count", "mean", "min", "max", "std", "nulos"]
        # Filtra para apenas colunas que realmente existem no DataFrame e estão na lista esperada
        cols_to_show = [c for c in expected_cols if c in final_df.columns]
        return final_df[cols_to_show]
        
    return pd.DataFrame(columns=["Coluna", "count", "mean", "min", "max", "std", "nulos"])

def analyze_multivalue_column(df: pd.DataFrame, column_name: str, delimiter: str) -> Optional[Dict[str, Any]]:
    if column_name not in df.columns:
        st.warning(f"Coluna '{column_name}' não encontrada no DataFrame.")
        return None
    if not delimiter: 
        st.warning("O delimitador interno não pode ser uma string vazia.")
        return None

    series_to_analyze = df[column_name].fillna('').astype(str)
    
    def split_and_strip(text, delim):
        if not text.strip(): 
            return []
        return [item.strip() for item in text.split(delim)]

    list_series = series_to_analyze.apply(lambda x: split_and_strip(x, delimiter))
    exploded_series = list_series.explode()
    exploded_series_cleaned = exploded_series[exploded_series.str.strip() != ''] 
    
    if exploded_series_cleaned.empty:
        return {
            "total_items": 0,
            "unique_items_count": 0,
            "all_unique_items": np.array([]), 
            "frequency_df": pd.DataFrame(columns=['item', 'contagem'])
        }

    total_items = len(exploded_series_cleaned) 
    unique_items_list = exploded_series_cleaned.unique()
    unique_items_count = len(unique_items_list)
    
    frequency_df = exploded_series_cleaned.value_counts().reset_index()
    frequency_df.columns = ['item', 'contagem']
    
    return {
        "total_items": total_items,
        "unique_items_count": unique_items_count,
        "all_unique_items": np.asarray(unique_items_list), 
        "frequency_df": frequency_df
    }

# --- Funções de Conversão ---
def convert_df_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Dados") -> bytes: # Adicionado sheet_name
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name[:31]) # Limita nome da aba a 31 chars
    return output.getvalue()

# --- Funções de Geração de Relatório ---
def generate_excel_report(
    df1: Optional[pd.DataFrame], df1_name: str, df2: Optional[pd.DataFrame] = None,
    df2_name: Optional[str] = None, comparison_results: Optional[Dict[str, Any]] = None,
    individual_analysis1: Optional[Dict[str, Any]] = None, individual_analysis2: Optional[Dict[str, Any]] = None
) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:     
        if df1 is not None and individual_analysis1 is not None: 
            sheet_name1 = f"Analise {df1_name[:20]}" 
            current_row = 0
            df_report1_overview = pd.DataFrame([{
                "Nome do Arquivo": df1_name, "Número de Linhas": df1.shape[0],
                "Número de Colunas": df1.shape[1], "Colunas": ", ".join(df1.columns.tolist()) 
            }])
            df_report1_overview.to_excel(writer, sheet_name=sheet_name1, index=False, startrow=current_row)
            current_row += len(df_report1_overview) + 2
            
            if "null_info" in individual_analysis1 and not individual_analysis1["null_info"].empty:
                 pd.DataFrame([{"Seção": "Nulos e Vazios"}]).to_excel(writer, sheet_name=sheet_name1, index=False, startrow=current_row, header=False)
                 current_row += 1
                 individual_analysis1["null_info"].to_excel(writer, sheet_name=sheet_name1, index=False, startrow=current_row) 
                 current_row += len(individual_analysis1["null_info"]) + 2

            if "stats" in individual_analysis1 and not individual_analysis1["stats"].empty:
                pd.DataFrame([{"Seção": "Estatísticas Básicas (Numéricas)"}]).to_excel(writer, sheet_name=sheet_name1, index=False, startrow=current_row, header=False)
                current_row += 1
                individual_analysis1["stats"].to_excel(writer, sheet_name=sheet_name1, index=False, startrow=current_row)
                current_row += len(individual_analysis1["stats"]) + 2
            # Adicionar mais seções aqui se necessário

        if df2 is not None and df2_name is not None and individual_analysis2 is not None: 
            sheet_name2 = f"Analise {df2_name[:20]}"
            current_row = 0 # Reset current_row para a nova aba
            df_report2_overview = pd.DataFrame([{
                "Nome do Arquivo": df2_name, "Número de Linhas": df2.shape[0],
                "Número de Colunas": df2.shape[1], "Colunas": ", ".join(df2.columns.tolist()) 
            }])
            df_report2_overview.to_excel(writer, sheet_name=sheet_name2, index=False, startrow=current_row)
            current_row += len(df_report2_overview) + 2

            if "null_info" in individual_analysis2 and not individual_analysis2["null_info"].empty:
                 pd.DataFrame([{"Seção": "Nulos e Vazios"}]).to_excel(writer, sheet_name=sheet_name2, index=False, startrow=current_row, header=False)
                 current_row +=1
                 individual_analysis2["null_info"].to_excel(writer, sheet_name=sheet_name2, index=False, startrow=current_row)
                 current_row += len(individual_analysis2["null_info"]) + 2
            
            if "stats" in individual_analysis2 and not individual_analysis2["stats"].empty:
                pd.DataFrame([{"Seção": "Estatísticas Básicas (Numéricas)"}]).to_excel(writer, sheet_name=sheet_name2, index=False, startrow=current_row, header=False)
                current_row +=1
                individual_analysis2["stats"].to_excel(writer, sheet_name=sheet_name2, index=False, startrow=current_row)
                current_row += len(individual_analysis2["stats"]) + 2
            # Adicionar mais seções aqui se necessário

        if comparison_results and df1 is not None and df2 is not None:
            sheet_name_diff = "Diferencas"
            start_row_diff = 0
            if "structure_comparison" in comparison_results and not comparison_results["structure_comparison"].empty:
                pd.DataFrame([{"Seção": "Comparação de Estrutura (Colunas e Tipos)"}]).to_excel(writer, sheet_name=sheet_name_diff, index=False, startrow=start_row_diff, header=False)
                start_row_diff +=1
                comparison_results["structure_comparison"].to_excel(writer, sheet_name=sheet_name_diff, index=False, startrow=start_row_diff) 
                start_row_diff += len(comparison_results["structure_comparison"]) + 2
            
            if "row_col_counts" in comparison_results:
                pd.DataFrame([{"Seção": "Contagem de Linhas e Colunas"}]).to_excel(writer, sheet_name=sheet_name_diff, index=False, startrow=start_row_diff, header=False)
                start_row_diff +=1
                counts_df = pd.DataFrame([comparison_results["row_col_counts"]])
                counts_df.to_excel(writer, sheet_name=sheet_name_diff, index=False, startrow=start_row_diff)
                start_row_diff += len(counts_df) + 2 

            if "key_check_summary" in comparison_results and comparison_results["key_check_summary"]: 
                 pd.DataFrame([{"Seção": "Verificação de Chaves"}]).to_excel(writer, sheet_name=sheet_name_diff, index=False, startrow=start_row_diff, header=False)
                 start_row_diff +=1
                 key_summary_df = pd.DataFrame(comparison_results["key_check_summary"])
                 if "Amostra" in key_summary_df.columns:
                     key_summary_df["Amostra"] = key_summary_df["Amostra"].apply(lambda x: ", ".join(map(str, x[:5])) + ('...' if len(x) > 5 else '') if isinstance(x, list) else x)

                 key_summary_df.to_excel(writer, sheet_name=sheet_name_diff, index=False, startrow=start_row_diff)
                 start_row_diff += len(key_summary_df) + 2

            if "value_diff_summary" in comparison_results and not comparison_results["value_diff_summary"].empty:
                pd.DataFrame([{"Seção": "Colunas com Valores Distintos (Chaves Correspondentes)"}]).to_excel(writer, sheet_name=sheet_name_diff, index=False, startrow=start_row_diff, header=False)
                start_row_diff +=1
                comparison_results["value_diff_summary"].to_excel(writer, sheet_name=sheet_name_diff, index=False, startrow=start_row_diff)
    return output.getvalue()

def generate_markdown_report(
    df1: Optional[pd.DataFrame], df1_name: str, df2: Optional[pd.DataFrame] = None,
    df2_name: Optional[str] = None, comparison_results: Optional[Dict[str, Any]] = None,
    individual_analysis1: Optional[Dict[str, Any]] = None, individual_analysis2: Optional[Dict[str, Any]] = None
) -> bytes:
    md_content = f"# Relatório de Auditoria de Dados\n\n" # Removido (UNESTAC)
    md_content += f"Data da Geração: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n"
    if df1 is not None and individual_analysis1 is not None: 
        md_content += f"## Análise do Arquivo 1: {df1_name}\n"
        md_content += f"- **Número de Linhas:** {df1.shape[0]}\n"
        md_content += f"- **Número de Colunas:** {df1.shape[1]}\n"
        md_content += f"- **Colunas:** `{', '.join(df1.columns.tolist())}`\n\n" 
        if "null_info" in individual_analysis1 and not individual_analysis1["null_info"].empty:
            md_content += "### Nulos/Vazios\n"
            md_content += individual_analysis1["null_info"].to_markdown(index=False) + "\n\n"
        if "stats" in individual_analysis1 and not individual_analysis1["stats"].empty:
            md_content += "### Estatísticas Básicas (Numéricas)\n"
            md_content += individual_analysis1["stats"].to_markdown(index=False) + "\n\n"
        # Adicionar mais seções aqui, se necessário
    if df2 is not None and df2_name is not None and individual_analysis2 is not None: 
        md_content += f"## Análise do Arquivo 2: {df2_name}\n"
        md_content += f"- **Número de Linhas:** {df2.shape[0]}\n"
        md_content += f"- **Número de Colunas:** {df2.shape[1]}\n"
        md_content += f"- **Colunas:** `{', '.join(df2.columns.tolist())}`\n\n"
        if "null_info" in individual_analysis2 and not individual_analysis2["null_info"].empty:
            md_content += "### Nulos/Vazios\n"
            md_content += individual_analysis2["null_info"].to_markdown(index=False) + "\n\n"
        if "stats" in individual_analysis2 and not individual_analysis2["stats"].empty:
            md_content += "### Estatísticas Básicas (Numéricas)\n"
            md_content += individual_analysis2["stats"].to_markdown(index=False) + "\n\n"
        # Adicionar mais seções aqui, se necessário

    if comparison_results and df1 is not None and df2 is not None :
        md_content += f"## Comparação Entre Arquivos: {df1_name} vs {df2_name}\n\n"
        if "structure_comparison" in comparison_results and not comparison_results["structure_comparison"].empty:
            md_content += "### Comparação de Estrutura (Colunas e Tipos)\n"
            md_content += comparison_results["structure_comparison"].to_markdown(index=False) + "\n"
            if "unique_cols_df1" in comparison_results and comparison_results["unique_cols_df1"]:
                md_content += f"- Colunas Exclusivas em `{df1_name}`: `{', '.join(comparison_results['unique_cols_df1'])}`\n"
            if "unique_cols_df2" in comparison_results and comparison_results["unique_cols_df2"]:
                md_content += f"- Colunas Exclusivas em `{df2_name}`: `{', '.join(comparison_results['unique_cols_df2'])}`\n\n"
        
        if "row_col_counts" in comparison_results:
            md_content += "### Diferença no Número Total de Linhas e Colunas\n"
            rc_counts = comparison_results["row_col_counts"]
            md_content += f"- `{df1_name}`: {rc_counts['Arquivo 1 Linhas']} linhas, {rc_counts['Arquivo 1 Colunas']} colunas\n"
            md_content += f"- `{df2_name}`: {rc_counts['Arquivo 2 Linhas']} linhas, {rc_counts['Arquivo 2 Colunas']} colunas\n\n"

        if "key_check_summary" in comparison_results and comparison_results["key_check_summary"]:
            md_content += "### Verificação de Chaves\n"
            key_summary_df = pd.DataFrame(comparison_results["key_check_summary"])
            if "Amostra" in key_summary_df.columns: 
                 key_summary_df["Amostra (até 5)"] = key_summary_df["Amostra"].apply(lambda x: str(x[:5]) if isinstance(x, list) else x)
                 # Seleciona colunas para o markdown, garantindo que 'Amostra (até 5)' exista
                 cols_for_md = ["Descrição", "Contagem"]
                 if "Amostra (até 5)" in key_summary_df.columns:
                     cols_for_md.append("Amostra (até 5)")
                 md_content += key_summary_df[cols_for_md].to_markdown(index=False) + "\n\n"

            else:
                 md_content += key_summary_df.to_markdown(index=False) + "\n\n"

        if "value_diff_summary" in comparison_results and not comparison_results["value_diff_summary"].empty:
            md_content += "### Colunas com Valores Distintos (para chaves correspondentes)\n"
            md_content += comparison_results["value_diff_summary"].to_markdown(index=False) + "\n\n"
        elif "selected_key1" in comparison_results and comparison_results["selected_key1"] and \
             "selected_key2" in comparison_results and comparison_results["selected_key2"]:
            md_content += "### Colunas com Valores Distintos (para chaves correspondentes)\n"
            md_content += "Nenhuma divergência encontrada ou não haviam chaves correspondentes/comuns para comparar.\n\n"
            
    return md_content.encode("utf-8")

# --- Funções de Comparação ---
def compare_structures(df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    cols1 = set(df1.columns); cols2 = set(df2.columns)
    common_cols = sorted(list(cols1.intersection(cols2)))
    unique_cols_df1 = sorted(list(cols1 - cols2)); unique_cols_df2 = sorted(list(cols2 - cols1))
    type_comparison = []
    for col in common_cols:
        type1 = str(df1[col].dtype); type2 = str(df2[col].dtype)
        type_comparison.append({"Coluna": col, "Tipo Arq1": type1, "Tipo Arq2": type2, "Tipos Iguais?": type1 == type2})
    
    # Adicionar colunas únicas ao DataFrame de comparação de estrutura
    for col in unique_cols_df1:
        type_comparison.append({"Coluna": col, "Tipo Arq1": str(df1[col].dtype), "Tipo Arq2": "N/A (Exclusiva Arq1)", "Tipos Iguais?": False})
    for col in unique_cols_df2:
        type_comparison.append({"Coluna": col, "Tipo Arq1": "N/A (Exclusiva Arq2)", "Tipo Arq2": str(df2[col].dtype), "Tipos Iguais?": False})

    return pd.DataFrame(type_comparison), unique_cols_df1, unique_cols_df2

def compare_row_column_counts(df1: pd.DataFrame, df2: pd.DataFrame) -> Dict[str, int]:
    rows1, cols1_count = df1.shape; rows2, cols2_count = df2.shape
    return {"Arquivo 1 Linhas": rows1, "Arquivo 1 Colunas": cols1_count, 
            "Arquivo 2 Linhas": rows2, "Arquivo 2 Colunas": cols2_count,
            "Diferença Linhas (Arq1 - Arq2)": rows1 - rows2, 
            "Diferença Colunas (Arq1 - Arq2)": cols1_count - cols2_count}

def check_key_presence(
    df1: pd.DataFrame, df2: pd.DataFrame,
    key_col_df1: str, key_col_df2: str,
    df1_name: str = "Arq1", df2_name: str = "Arq2"
) -> Tuple[Optional[Set[str]], Optional[Set[str]], Optional[List[str]], Optional[List[str]], Optional[List[str]], List[Dict[str, Any]]]:
    summary: List[Dict[str, Any]] = []
    if not key_col_df1 or key_col_df1 == "Não Selecionado" or \
       not key_col_df2 or key_col_df2 == "Não Selecionado":
        msg = "Colunas chave não selecionadas para um ou ambos os arquivos."
        summary.append({"Descrição": msg, "Contagem": "N/A", "Amostra": []})
        return None, None, None, None, None, summary
    if key_col_df1 not in df1.columns:
        msg = f"Coluna chave '{key_col_df1}' não encontrada no Arquivo 1 ({df1_name})."
        summary.append({"Descrição": msg, "Contagem": "N/A", "Amostra": []})
        return None, None, None, None, None, summary
    if key_col_df2 not in df2.columns:
        msg = f"Coluna chave '{key_col_df2}' não encontrada no Arquivo 2 ({df2_name})."
        summary.append({"Descrição": msg, "Contagem": "N/A", "Amostra": []})
        return None, None, None, None, None, summary

    # Tratar NaNs e converter chaves para string para comparação robusta
    keys_df1_set = set(df1[key_col_df1].dropna().astype(str).unique())
    keys_df2_set = set(df2[key_col_df2].dropna().astype(str).unique())

    # Remover chaves que são strings vazias após conversão e dropna
    keys_df1_set = {k for k in keys_df1_set if k.strip()}
    keys_df2_set = {k for k in keys_df2_set if k.strip()}

    missing_in_df2_list = sorted(list(keys_df1_set - keys_df2_set))
    missing_in_df1_list = sorted(list(keys_df2_set - keys_df1_set))
    common_keys_list = sorted(list(keys_df1_set.intersection(keys_df2_set)))

    amostra_ui_limite = 10 

    summary = [
        {"Descrição": f"Chaves ÚNICAS em `{key_col_df1}` ({df1_name}) (não vazias)", "Contagem": len(keys_df1_set), "Amostra": sorted(list(keys_df1_set))[:amostra_ui_limite]},
        {"Descrição": f"Chaves ÚNICAS em `{key_col_df2}` ({df2_name}) (não vazias)", "Contagem": len(keys_df2_set), "Amostra": sorted(list(keys_df2_set))[:amostra_ui_limite]},
        {"Descrição": f"Chaves COMUNS entre `{df1_name}` e `{df2_name}`", "Contagem": len(common_keys_list), "Amostra": common_keys_list[:amostra_ui_limite]},
        {"Descrição": f"Chaves de `{df1_name}` NÃO ENCONTRADAS em `{df2_name}`", "Contagem": len(missing_in_df2_list), "Amostra": missing_in_df2_list[:amostra_ui_limite]},
        {"Descrição": f"Chaves de `{df2_name}` NÃO ENCONTRADAS em `{df1_name}`", "Contagem": len(missing_in_df1_list), "Amostra": missing_in_df1_list[:amostra_ui_limite]}
    ]
    return keys_df1_set, keys_df2_set, missing_in_df2_list, missing_in_df1_list, common_keys_list, summary

# --- Interface Streamlit ---
st.sidebar.header("Configurações da Análise")
# Para o primeiro uploader de arquivo
uploaded_file_1 = st.sidebar.file_uploader(
    label="**Arquivo 1 (Obrigatório)**", 
    type=["csv", "xlsx", "xls", "parquet", "db"], 
    key="file1",
    help="Arraste e solte o arquivo ou clique em 'Procurar arquivos'. Limite 200MB. Formatos: CSV, Excel, Parquet, DB."
)

# Para o segundo uploader de arquivo
uploaded_file_2 = st.sidebar.file_uploader(
    label="**Arquivo 2 (Opcional para Comparação)**", 
    type=["csv", "xlsx", "xls", "parquet", "db"], 
    key="file2",
    help="Arraste e solte o arquivo ou clique em 'Procurar arquivos'. Limite 200MB. Formatos: CSV, Excel, Parquet, DB."
)

if 'df1' not in st.session_state: st.session_state.df1 = None
if 'df2' not in st.session_state: st.session_state.df2 = None
if 'report_data_df1' not in st.session_state: st.session_state.report_data_df1 = {}
if 'report_data_df2' not in st.session_state: st.session_state.report_data_df2 = {}
if 'report_data_comparison' not in st.session_state: st.session_state.report_data_comparison = {}
if 'uploaded_file_1_name' not in st.session_state: st.session_state.uploaded_file_1_name = None
if 'uploaded_file_2_name' not in st.session_state: st.session_state.uploaded_file_2_name = None
# Adicionar estado para colunas chave selecionadas para persistência
if 'key1_sel_sidebar' not in st.session_state: st.session_state.key1_sel_sidebar = "Não Selecionado"
if 'key2_sel_sidebar' not in st.session_state: st.session_state.key2_sel_sidebar = "Não Selecionado"


process_button = st.sidebar.button("Processar Arquivos", type="primary")

if process_button:
    st.session_state.df1 = None; st.session_state.df2 = None
    st.session_state.report_data_df1 = {}; st.session_state.report_data_df2 = {}
    st.session_state.report_data_comparison = {}
    st.session_state.uploaded_file_1_name = None; st.session_state.uploaded_file_2_name = None
    st.session_state.viewing_all_keys_description = None
    st.session_state.all_keys_data_to_show = None
    st.session_state.expanded_view_rendered_this_cycle = False
    # Não resetar st.session_state.key1_sel_sidebar e key2_sel_sidebar aqui, 
    # pois queremos que eles persistam se o usuário apenas reprocessar. 
    # Eles são atualizados pelos selectboxes.

    if uploaded_file_1:
        with st.spinner(f"Carregando Arquivo 1: {uploaded_file_1.name}..."):
            st.session_state.df1 = load_data(uploaded_file_1)
        if st.session_state.df1 is not None:
             st.session_state.uploaded_file_1_name = uploaded_file_1.name
    else: # Adicionado para garantir que se o arquivo 1 for removido, df1 seja None
        st.session_state.df1 = None
        st.session_state.uploaded_file_1_name = None
        st.warning("Arquivo 1 é obrigatório para processamento.")


    if uploaded_file_2:
        with st.spinner(f"Carregando Arquivo 2: {uploaded_file_2.name}..."):
            st.session_state.df2 = load_data(uploaded_file_2)
        if st.session_state.df2 is not None:
            st.session_state.uploaded_file_2_name = uploaded_file_2.name
    else: # Adicionado para garantir que se o arquivo 2 for removido, df2 seja None
        st.session_state.df2 = None
        st.session_state.uploaded_file_2_name = None


def display_individual_analysis(df: pd.DataFrame, file_name: str, key_prefix: str, report_data_dict: Dict[str, Any]):
    st.header(f"Análise do Arquivo: {file_name}")
    with st.expander("Visão Geral e Amostra de Dados", expanded=True):
        st.write(f"**Linhas:** {df.shape[0]}, **Colunas:** {df.shape[1]}")
        st.dataframe(df.head())
    report_data_dict["overview"] = {"Nome": file_name, "Linhas": df.shape[0], "Colunas": df.shape[1], "Amostra": df.head()}
    report_data_dict["column_list"] = df.columns.tolist()

    st.subheader("1. Informações das Colunas")
    cols_info = [{"Nome da Coluna": col, 
                  "Tipo de Dado (Pandas)": str(df[col].dtype),
                  "Valores Não Nulos": int(df[col].count()), # Convertido para int nativo
                  "Valores Únicos": int(df[col].nunique())} # Convertido para int nativo
                 for col in df.columns]
    st.dataframe(pd.DataFrame(cols_info))
    report_data_dict["column_info"] = pd.DataFrame(cols_info) # Salvar no relatório

    st.subheader("2. Análise de Valores Distintos")
    col_distinct_opts = ["Selecione uma coluna"] + df.columns.tolist()
    if f"{key_prefix}_distinct_col_selected" not in st.session_state:
        st.session_state[f"{key_prefix}_distinct_col_selected"] = "Selecione uma coluna"
    
    st.session_state[f"{key_prefix}_distinct_col_selected"] = st.selectbox(
        f"Selecione a coluna para visualizar valores distintos ({file_name}):", 
        col_distinct_opts, 
        key=f"{key_prefix}_distinct_col_sb", 
        index=col_distinct_opts.index(st.session_state[f"{key_prefix}_distinct_col_selected"])
    )
    col_distinct = st.session_state[f"{key_prefix}_distinct_col_selected"]

    analyze_mv_for_distinct = False
    # Estado de sessão para o delimitador
    if f"{key_prefix}_internal_delimiter_distinct" not in st.session_state:
        st.session_state[f"{key_prefix}_internal_delimiter_distinct"] = "|" # Valor padrão
    
    internal_delimiter_for_distinct_input = "|" # Valor padrão caso o checkbox não esteja marcado ou não seja string

    if col_distinct != "Selecione uma coluna":
        if pd.api.types.is_string_dtype(df[col_distinct]) or pd.api.types.is_object_dtype(df[col_distinct]):
            analyze_mv_for_distinct = st.checkbox("Analisar itens individuais (para campos multivalor)",  # Texto do checkbox atualizado
                                              key=f"{key_prefix}_analyze_mv_distinct_checkbox", value=False)
            if analyze_mv_for_distinct:
                # Mostrar input do delimitador SOMENTE se o checkbox estiver marcado
                st.session_state[f"{key_prefix}_internal_delimiter_distinct"] = st.text_input(
                    "Delimitador dos itens individuais:",
                    value=st.session_state[f"{key_prefix}_internal_delimiter_distinct"],
                    key=f"{key_prefix}_delimiter_distinct_input"
                )
                internal_delimiter_for_distinct_input = st.session_state[f"{key_prefix}_internal_delimiter_distinct"]
            
        limit_distinct = st.slider(f"Limite de exemplos distintos a exibir ({col_distinct}):", 5, 100, 10, key=f"{key_prefix}_distinct_limit")

        # Garantir que o delimitador só seja usado se a análise multivalor estiver ativa e o delimitador não for vazio
        use_delimiter = analyze_mv_for_distinct and internal_delimiter_for_distinct_input.strip() != ""
        delimiter_to_use = internal_delimiter_for_distinct_input if use_delimiter else None

        if analyze_mv_for_distinct and not delimiter_to_use:
            st.warning("Por favor, forneça um delimitador válido para a análise de itens individuais.")
            # Você pode optar por não prosseguir com a análise ou analisar sem explodir os valores
            # Aqui, vamos prosseguir, mas a função get_distinct_values_report não usará o delimitador
            # e o resultado será como se analyze_mv_for_distinct fosse False.

        distinct_vals, total_distinct_count = get_distinct_values_report(
            df, 
            col_distinct, 
            limit_distinct,
            analyze_as_multivalue=use_delimiter, # Passa True apenas se o delimitador for válido e a opção estiver marcada
            internal_delimiter=delimiter_to_use    # Passa o delimitador efetivo
        )

        st.write(f"Total de valores distintos em '{col_distinct}' {'(como itens individuais usando "' + delimiter_to_use + '")' if use_delimiter else ''}: **{total_distinct_count}**")
        if total_distinct_count > 0:
            st.write(f"Exibindo até {len(distinct_vals)} exemplos:")
            distinct_vals_str = "\n".join(map(str, distinct_vals.tolist() if isinstance(distinct_vals, np.ndarray) else list(distinct_vals)))
            st.text_area("Valores distintos (amostra):", value=distinct_vals_str, height=150, key=f"{key_prefix}_distinct_values_display", disabled=True)
        
        report_data_dict.setdefault("distinct_values_selected", {})[col_distinct] = {
            "total": total_distinct_count, 
            "sample": list(distinct_vals),
            "is_multivalue_analysis": use_delimiter, # Atualizado para refletir se o delimitador foi realmente usado
            "delimiter_used": delimiter_to_use if use_delimiter else None # Guarda o delimitador usado
        }

    st.subheader("3. Análise de Nulos e Vazios")
    null_info_df = get_null_empty_info_df(df)
    if not null_info_df.empty: st.dataframe(null_info_df)
    else: st.success("Nenhuma coluna com valores nulos ou strings vazias/com apenas espaços encontrada.")
    report_data_dict["null_info"] = null_info_df

    st.subheader("4. Verificação de Inconsistências de Padronização")
    with st.expander("Inconsistências de Capitalização"):
        cap_incons = get_capitalization_inconsistencies_report(df)
        if cap_incons:
            for col, inconsistencies in cap_incons.items():
                st.write(f"**Coluna `{col}`:**")
                for lower_val, originals in inconsistencies.items():
                    st.caption(f"  - Valor base (minúsculo): `{lower_val}`, Variações: `{', '.join(originals)}`")
        else: st.info("Nenhuma inconsistência de capitalização detectada (ignorando strings vazias/com espaços).")
        report_data_dict["capitalization"] = cap_incons
    with st.expander("Formatos de Data Mistos (Beta)"):
        date_fmts = get_date_format_inconsistencies_report(df)
        if date_fmts:
            for col, formats in date_fmts.items():
                st.write(f"**Coluna `{col}`:** Formatos detectados: `{', '.join(formats)}`")
        else: st.info("Nenhuma coluna com múltiplos formatos de data (baseado em regex e parse do Pandas) detectada.")
        report_data_dict["date_formats"] = date_fmts
    with st.expander("Valores com Espaços no Início/Fim (Amostra)"):
        space_iss = get_leading_trailing_spaces_report(df)
        if space_iss:
            for col, values in space_iss.items():
                st.write(f"**Coluna `{col}`:**")
                sample_display = [f"'{v}'" for v in values[:5]] # Mostrar aspas para evidenciar
                st.caption(f"  Amostra: {', '.join(sample_display)}")
        else: st.info("Nenhum valor com espaços extras (que não sejam apenas espaços) detectado.")
        report_data_dict["leading_trailing_spaces"] = space_iss

    st.subheader("5. Estatísticas Descritivas (Colunas Numéricas)")
    stats_df = get_basic_stats_df(df)
    if not stats_df.empty: st.dataframe(stats_df)
    else: st.info("Nenhuma coluna numérica com dados para estatísticas encontrada.")
    report_data_dict["stats"] = stats_df

    st.subheader("6. Análise Detalhada de Campos Multivalor")
    with st.expander("Analisar Coluna com Múltiplos Valores Internos", expanded=False):
        string_object_cols = [col for col in df.columns if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col])]
        if not string_object_cols:
            st.info("Nenhuma coluna de texto/objeto disponível para análise multivalor neste arquivo.")
        else:
            # Estado de sessão para a coluna multivalor e delimitador
            if f"{key_prefix}_mv_col_detail_selected" not in st.session_state:
                st.session_state[f"{key_prefix}_mv_col_detail_selected"] = "Selecione uma coluna"
            if f"{key_prefix}_mv_delimiter_detail_input" not in st.session_state:
                st.session_state[f"{key_prefix}_mv_delimiter_detail_input"] = "|"

            mv_col_options_detail = ["Selecione uma coluna"] + string_object_cols
            st.session_state[f"{key_prefix}_mv_col_detail_selected"] = st.selectbox(
                "Coluna para análise detalhada de multivalores:", 
                mv_col_options_detail, 
                key=f"{key_prefix}_mv_col_detail_sb",
                index=mv_col_options_detail.index(st.session_state[f"{key_prefix}_mv_col_detail_selected"])
            )
            mv_selected_col_detail = st.session_state[f"{key_prefix}_mv_col_detail_selected"]

            st.session_state[f"{key_prefix}_mv_delimiter_detail_input"] = st.text_input(
                "Delimitador interno dos valores:", 
                value=st.session_state[f"{key_prefix}_mv_delimiter_detail_input"], 
                key=f"{key_prefix}_mv_delimiter_detail_ti"
            )
            mv_delimiter_detail = st.session_state[f"{key_prefix}_mv_delimiter_detail_input"]

            if mv_selected_col_detail != "Selecione uma coluna" and mv_delimiter_detail:
                if st.button("Analisar Campo Multivalor Selecionado", key=f"{key_prefix}_mv_analyze_btn_detail"):
                    with st.spinner(f"Analisando '{mv_selected_col_detail}'..."):
                        mv_analysis_results = analyze_multivalue_column(df, mv_selected_col_detail, mv_delimiter_detail)
                    st.session_state[f"{key_prefix}_mv_analysis_results_{mv_selected_col_detail}"] = mv_analysis_results

            analysis_key_in_session = f"{key_prefix}_mv_analysis_results_{mv_selected_col_detail}"
            if analysis_key_in_session in st.session_state and st.session_state[analysis_key_in_session]:
                mv_analysis_results = st.session_state[analysis_key_in_session]
                
                st.write(f"**Resultados para a coluna '{mv_selected_col_detail}' (delimitador: '{mv_delimiter_detail}'):**")
                
                col1_metric, col2_metric = st.columns(2)
                with col1_metric:
                    st.metric("Total de Itens Individuais", mv_analysis_results["total_items"])
                with col2_metric:
                    st.metric("Número de Itens Únicos", mv_analysis_results["unique_items_count"])
                
                all_unique_items_list = mv_analysis_results["all_unique_items"]

                if all_unique_items_list.size > 0:
                    st.write("Visualização de Itens Únicos:")
                    num_options = [10, 20, 30, 50, 100, 200, 500, len(all_unique_items_list)]
                    valid_options = sorted(list(set(opt for opt in num_options if opt <= len(all_unique_items_list))))
                    if not valid_options or (len(all_unique_items_list) not in valid_options and len(all_unique_items_list) > 0) :
                         if len(all_unique_items_list) > 0 : valid_options.append(len(all_unique_items_list))
                         valid_options = sorted(list(set(valid_options))) # Remover duplicatas e ordenar
                    
                    if not valid_options and len(all_unique_items_list) > 0:
                        valid_options = [len(all_unique_items_list)]
                    
                    default_index = min(2, len(valid_options) -1) if len(valid_options) > 2 else (len(valid_options) -1 if valid_options else 0)

                    if valid_options: 
                        num_to_display_mv = st.selectbox(
                            "Número de itens únicos a exibir:",
                            options=valid_options,
                            index=default_index, 
                            format_func=lambda x: f"Exibir {x} itens" if x != len(all_unique_items_list) or x==0 else f"Exibir Todos ({x})",
                            key=f"{key_prefix}_mv_num_unique_to_show"
                        )
                        items_to_display_sample = all_unique_items_list[:num_to_display_mv]
                        df_unique_items_sample = pd.DataFrame(items_to_display_sample, columns=[f"Itens Únicos Visíveis ({mv_selected_col_detail})"])
                        display_height = max(100, min(400, (len(items_to_display_sample) + 1) * 35 + 10)) if len(items_to_display_sample) > 0 else 100
                        st.dataframe(df_unique_items_sample, height=display_height)
                    else:
                        st.info("Não há itens únicos para exibir.")
                
                st.write("Tabela de Frequência dos Itens Individuais:")
                freq_df_to_export = mv_analysis_results["frequency_df"]
                st.dataframe(freq_df_to_export)
                
                report_data_dict.setdefault("multivalue_column_analysis", {}).setdefault(file_name, {})[mv_selected_col_detail] = mv_analysis_results
                
                if not freq_df_to_export.empty:
                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        csv_export_bytes = freq_df_to_export.to_csv(index=False, sep=';', encoding='utf-8-sig').encode('utf-8-sig')
                        st.download_button(
                            label=f"Freq. '{mv_selected_col_detail}' (.csv)",
                            data=csv_export_bytes,
                            file_name=f"frequencia_{key_prefix}_{mv_selected_col_detail}.csv",
                            mime="text/csv",
                            key=f"{key_prefix}_mv_download_freq_csv_detail" 
                        )
                    with col_dl2:
                        excel_export_bytes = convert_df_to_excel_bytes(freq_df_to_export, sheet_name=f"Freq {mv_selected_col_detail[:20]}") 
                        st.download_button(
                            label=f"Freq. '{mv_selected_col_detail}' (.xlsx)",
                            data=excel_export_bytes,
                            file_name=f"frequencia_{key_prefix}_{mv_selected_col_detail}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key=f"{key_prefix}_mv_download_freq_excel_detail" 
                        )
            elif mv_selected_col_detail != "Selecione uma coluna" and not mv_delimiter_detail:
                st.warning("Por favor, forneça um delimitador interno.")

# --- Execução da Análise e Exibição ---
if st.session_state.df1 is not None and st.session_state.uploaded_file_1_name:
    with st.spinner(f"Analisando Arquivo 1: {st.session_state.uploaded_file_1_name}..."):
        display_individual_analysis(st.session_state.df1, st.session_state.uploaded_file_1_name, "df1", st.session_state.report_data_df1)
if st.session_state.df2 is not None and st.session_state.uploaded_file_2_name:
    with st.spinner(f"Analisando Arquivo 2: {st.session_state.uploaded_file_2_name}..."):
        display_individual_analysis(st.session_state.df2, st.session_state.uploaded_file_2_name, "df2", st.session_state.report_data_df2)

# --- Comparação entre Arquivos ---
if st.session_state.df1 is not None and st.session_state.df2 is not None and \
   st.session_state.uploaded_file_1_name and st.session_state.uploaded_file_2_name:
    
    st.header("Comparação Entre Arquivos")
    # Opções de comparação já estão na sidebar, apenas o título aqui
    # st.sidebar.subheader("Opções de Comparação") # Removido pois já tem header geral

    df1_for_comp = st.session_state.df1
    df2_for_comp = st.session_state.df2
    name1 = st.session_state.uploaded_file_1_name
    name2 = st.session_state.uploaded_file_2_name

    st.subheader("1. Comparação de Estrutura (Colunas e Tipos)")
    type_comp_df, unique1, unique2 = compare_structures(df1_for_comp, df2_for_comp)
    st.session_state.report_data_comparison["structure_comparison"] = type_comp_df
    st.session_state.report_data_comparison["unique_cols_df1"] = unique1
    st.session_state.report_data_comparison["unique_cols_df2"] = unique2
    st.dataframe(type_comp_df)
    if unique1: st.info(f"Colunas exclusivas em **{name1}**: `{', '.join(unique1)}`")
    if unique2: st.info(f"Colunas exclusivas em **{name2}**: `{', '.join(unique2)}`")

    st.subheader("2. Comparação de Contagem de Linhas e Colunas")
    counts_comp = compare_row_column_counts(df1_for_comp, df2_for_comp)
    st.session_state.report_data_comparison["row_col_counts"] = counts_comp
    
    # Exibir contagens como métricas para melhor visualização
    col_count1, col_count2, col_count3, col_count4 = st.columns(4)
    col_count1.metric(f"Linhas {name1}", counts_comp["Arquivo 1 Linhas"])
    col_count2.metric(f"Colunas {name1}", counts_comp["Arquivo 1 Colunas"])
    col_count3.metric(f"Linhas {name2}", counts_comp["Arquivo 2 Linhas"])
    col_count4.metric(f"Colunas {name2}", counts_comp["Arquivo 2 Colunas"])
    if counts_comp["Diferença Linhas (Arq1 - Arq2)"] != 0:
        st.metric("Diferença Linhas (Arq1 - Arq2)", counts_comp["Diferença Linhas (Arq1 - Arq2)"])
    if counts_comp["Diferença Colunas (Arq1 - Arq2)"] != 0:
        st.metric("Diferença Colunas (Arq1 - Arq2)", counts_comp["Diferença Colunas (Arq1 - Arq2)"])


    st.subheader("3. Verificação de Chaves")
    key_opts1 = ["Não Selecionado"] + df1_for_comp.columns.tolist()
    key_opts2 = ["Não Selecionado"] + df2_for_comp.columns.tolist()
    common_cols_for_keys = list(set(df1_for_comp.columns).intersection(set(df2_for_comp.columns)))
    default_key_names = ["id", "chave", "codigo", "identificador", "numero_tombo", "chave_primaria", "id_delegacia", "id_procedimento"] 
    def_k1, def_k2 = "Não Selecionado", "Não Selecionado" # Inicializa como Não Selecionado

    # Tenta encontrar chaves padrão com o mesmo nome em ambos
    for common_key_name in default_key_names:
        if common_key_name.lower() in [c.lower() for c in common_cols_for_keys]:
            # Encontra os nomes exatos das colunas (case-sensitive)
            actual_col_name_df1 = next((c for c in df1_for_comp.columns if c.lower() == common_key_name.lower()), None)
            actual_col_name_df2 = next((c for c in df2_for_comp.columns if c.lower() == common_key_name.lower()), None)
            if actual_col_name_df1 and actual_col_name_df2:
                 def_k1 = actual_col_name_df1
                 def_k2 = actual_col_name_df2
                 break 
    # Fallback se não encontrou chave padrão com mesmo nome, tenta a primeira chave padrão em cada df
    if def_k1 == "Não Selecionado":
        for key_name_pattern in default_key_names:
            for col_df1 in df1_for_comp.columns:
                if key_name_pattern.lower() in col_df1.lower():
                    def_k1 = col_df1; break
            if def_k1 != "Não Selecionado": break
    if def_k2 == "Não Selecionado":
        for key_name_pattern in default_key_names:
            for col_df2 in df2_for_comp.columns:
                if key_name_pattern.lower() in col_df2.lower():
                    def_k2 = col_df2; break
            if def_k2 != "Não Selecionado": break
    # Último fallback: primeira coluna comum, se houver
    if def_k1 == "Não Selecionado" and common_cols_for_keys: def_k1 = common_cols_for_keys[0] 
    if def_k2 == "Não Selecionado" and common_cols_for_keys: def_k2 = common_cols_for_keys[0]
    # Se ainda não selecionado, usa a primeira coluna do respectivo df
    if def_k1 == "Não Selecionado" and len(df1_for_comp.columns) > 0: def_k1 = df1_for_comp.columns[0]
    if def_k2 == "Não Selecionado" and len(df2_for_comp.columns) > 0: def_k2 = df2_for_comp.columns[0]

    idx1 = key_opts1.index(def_k1) if def_k1 in key_opts1 else 0
    idx2 = key_opts2.index(def_k2) if def_k2 in key_opts2 else 0
    
    # Usar o estado de sessão para os selectboxes de chave
    st.session_state.key1_sel_sidebar = st.sidebar.selectbox(f"Coluna Chave Arq.1 ({name1}):", key_opts1, index=idx1, key="k1s_sidebar_sb") 
    st.session_state.key2_sel_sidebar = st.sidebar.selectbox(f"Coluna Chave Arq.2 ({name2}):", key_opts2, index=idx2, key="k2s_sidebar_sb")
    
    key1_sel = st.session_state.key1_sel_sidebar
    key2_sel = st.session_state.key2_sel_sidebar

    st.session_state.report_data_comparison["selected_key1"] = key1_sel if key1_sel != "Não Selecionado" else None
    st.session_state.report_data_comparison["selected_key2"] = key2_sel if key2_sel != "Não Selecionado" else None

    if key1_sel != "Não Selecionado" and key2_sel != "Não Selecionado":
        k1_s, k2_s, m_in_2, m_in_1, common_k_list, key_summary_list = check_key_presence(
            df1_for_comp, df2_for_comp, key1_sel, key2_sel, name1, name2
        )
        
        st.session_state.report_data_comparison["key_check_summary"] = key_summary_list
        if k1_s is not None: st.session_state.report_data_comparison["keys_df1_full"] = sorted(list(k1_s))
        if k2_s is not None: st.session_state.report_data_comparison["keys_df2_full"] = sorted(list(k2_s))
        if m_in_2 is not None: st.session_state.report_data_comparison["missing_in_df2_full"] = m_in_2
        if m_in_1 is not None: st.session_state.report_data_comparison["missing_in_df1_full"] = m_in_1
        if common_k_list is not None: st.session_state.report_data_comparison["common_keys_full"] = common_k_list

        st.markdown("#### Sumário da Verificação de Chaves:")
        if not key_summary_list or (len(key_summary_list) == 1 and key_summary_list[0]['Contagem'] == 'N/A'):
            st.info(key_summary_list[0]['Descrição'] if key_summary_list and key_summary_list[0]['Contagem'] == 'N/A' else "Nenhum sumário de verificação de chaves para exibir.")
        else:
            for item_index, item_summary in enumerate(key_summary_list):
                if not isinstance(item_summary, dict) or 'Descrição' not in item_summary:
                    st.warning(f"Item de sumário inválido no índice {item_index}.")
                    continue

                descricao_original = item_summary.get('Descrição')
                contagem_total = item_summary.get('Contagem', 0)
                amostra_para_ui = item_summary.get('Amostra', [])
                display_description = descricao_original if descricao_original is not None else "Descrição Indisponível"
                
                st.write(f"**{display_description}**: {contagem_total if contagem_total != 'N/A' else 'N/A'}")

                if amostra_para_ui and isinstance(contagem_total, int) and contagem_total > 0 and isinstance(amostra_para_ui, list):
                    st.caption(f"  Amostra (até {len(amostra_para_ui)} de {contagem_total}): `{', '.join(map(str, amostra_para_ui))}`")

                lista_completa_correspondente = None
                if descricao_original is not None: # Mapeamento para obter a lista completa correta
                    if f"Chaves ÚNICAS em `{key1_sel}` ({name1}) (não vazias)" == descricao_original:
                        lista_completa_correspondente = st.session_state.report_data_comparison.get("keys_df1_full")
                    elif f"Chaves ÚNICAS em `{key2_sel}` ({name2}) (não vazias)" == descricao_original:
                        lista_completa_correspondente = st.session_state.report_data_comparison.get("keys_df2_full")
                    elif f"Chaves COMUNS entre `{name1}` e `{name2}`" == descricao_original:
                        lista_completa_correspondente = st.session_state.report_data_comparison.get("common_keys_full")
                    elif f"Chaves de `{name1}` NÃO ENCONTRADAS em `{name2}`" == descricao_original:
                        lista_completa_correspondente = st.session_state.report_data_comparison.get("missing_in_df2_full")
                    elif f"Chaves de `{name2}` NÃO ENCONTRADAS em `{name1}`" == descricao_original:
                        lista_completa_correspondente = st.session_state.report_data_comparison.get("missing_in_df1_full")
                
                if lista_completa_correspondente is not None and \
                   isinstance(contagem_total, int) and \
                   contagem_total > len(amostra_para_ui) and contagem_total > 0 : 
                    
                    clean_desc_for_btn_key = re.sub(r'[^a-zA-Z0-9_]', '', display_description)[:30] # Aumentado e underscore permitido
                    btn_key = f"view_all_btn_{item_index}_{clean_desc_for_btn_key}"
                    button_label = f"Ver/Exportar todas as {contagem_total} chaves para: \"{display_description}\""
                    
                    if st.button(button_label, key=btn_key):
                        if descricao_original is not None: 
                            st.session_state.viewing_all_keys_description = descricao_original
                            col_name_df = f"Chaves ({descricao_original.replace('`','').replace('(não vazias)','').strip()})" 
                            st.session_state.all_keys_data_to_show = pd.DataFrame(lista_completa_correspondente, columns=[col_name_df])
                            st.session_state.expanded_view_rendered_this_cycle = False 
                            st.rerun() # Força o rerender para mostrar a seção expandida
                        else:
                            st.error("Erro: Não foi possível obter a descrição da categoria de chaves.")
                            st.session_state.viewing_all_keys_description = None
                            st.session_state.all_keys_data_to_show = None
                            st.session_state.expanded_view_rendered_this_cycle = False
        
        if st.session_state.get('viewing_all_keys_description') and \
           st.session_state.get('all_keys_data_to_show') is not None:

            if not st.session_state.get('expanded_view_rendered_this_cycle', False):
                st.session_state.expanded_view_rendered_this_cycle = True # Flag para evitar re-renderização em loop

                st.markdown("---")
                active_desc = st.session_state.viewing_all_keys_description
                df_para_exibir_e_exportar = st.session_state.all_keys_data_to_show

                st.subheader(f"Lista Completa: {active_desc if active_desc else 'N/D'}")
                if df_para_exibir_e_exportar is not None and not df_para_exibir_e_exportar.empty:
                    st.dataframe(df_para_exibir_e_exportar, height=300)
                    
                    nome_arquivo_limpo = re.sub(r'[^a-zA-Z0-9_.-]', '_', str(active_desc if active_desc else "lista_chaves"))
                    nome_arquivo_download = f"lista_chaves_{nome_arquivo_limpo[:50]}.xlsx"
                    
                    bytes_do_excel = convert_df_to_excel_bytes(df_para_exibir_e_exportar, sheet_name=f"Chaves {nome_arquivo_limpo[:20]}")
                    st.download_button(
                        label=f"Download Lista Completa (.xlsx)",
                        data=bytes_do_excel,
                        file_name=nome_arquivo_download,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"download_btn_expanded_{nome_arquivo_limpo}" 
                    )
                else:
                    st.info("Não há dados para exibir ou exportar para esta seleção.")

                if st.button("Fechar Visualização da Lista Completa", key="close_expanded_keys_view"):
                    st.session_state.viewing_all_keys_description = None
                    st.session_state.all_keys_data_to_show = None
                    st.session_state.expanded_view_rendered_this_cycle = False
                    st.rerun() # Força o rerender para ocultar a seção
        
        st.markdown("---") # Separador após a seção de verificação de chaves
        
        st.subheader("4. Comparação de Valores (para Chaves Iguais e Colunas Comuns)")
        try:
            if common_k_list and len(common_k_list) > 0: # Procede apenas se houver chaves comuns
                # Garantir que as colunas chave são do mesmo tipo (string) antes do merge
                df1_c = df1_for_comp.copy()
                df2_c = df2_for_comp.copy()
                df1_c[key1_sel] = df1_c[key1_sel].astype(str)
                df2_c[key2_sel] = df2_c[key2_sel].astype(str)
                
                # Usar apenas as linhas com chaves comuns para o merge
                df1_common_keys = df1_c[df1_c[key1_sel].isin(common_k_list)]
                df2_common_keys = df2_c[df2_c[key2_sel].isin(common_k_list)]

                # Evitar múltiplas correspondências se as chaves não forem únicas em cada df
                # (pega a primeira ocorrência da chave em cada df para o merge)
                df1_unique_common = df1_common_keys.drop_duplicates(subset=[key1_sel], keep='first')
                df2_unique_common = df2_common_keys.drop_duplicates(subset=[key2_sel], keep='first')

                merged_df = pd.merge(df1_unique_common, df2_unique_common, left_on=key1_sel, right_on=key2_sel, suffixes=("_a1", "_a2"), how="inner")
                
                if not merged_df.empty:
                    diff_summary_list = []
                    # Colunas comuns para comparação (excluindo as chaves usadas no merge)
                    cols_to_compare_values = [c for c in common_cols_for_keys if c != key1_sel and c != key2_sel]
                    
                    for col_original_name in cols_to_compare_values:
                        c_a1 = col_original_name + "_a1"
                        c_a2 = col_original_name + "_a2"
                        if c_a1 in merged_df.columns and c_a2 in merged_df.columns:
                            # Comparar como string para capturar diferenças de formatação, tipo, etc.
                            # Tratar NaNs de forma que NaN != NaN seja False, mas NaN != valor seja True
                            # Preencher NaNs com um placeholder único que não deve existir nos dados
                            nan_placeholder = "__NaN_Placeholder_Unique__" 
                            series_a1 = merged_df[c_a1].fillna(nan_placeholder).astype(str)
                            series_a2 = merged_df[c_a2].fillna(nan_placeholder).astype(str)
                            
                            comparison_series = series_a1 != series_a2
                            are_different_count = comparison_series.sum()

                            if are_different_count > 0:
                                percentage_diff = (are_different_count / len(merged_df) * 100)
                                diff_summary_list.append({"Coluna": col_original_name, "Divergências": are_different_count, "% Divergências": f"{percentage_diff:.2f}"})
                    
                    if diff_summary_list:
                        value_diff_df = pd.DataFrame(diff_summary_list)
                        st.dataframe(value_diff_df)
                        st.session_state.report_data_comparison["value_diff_summary"] = value_diff_df
                    else: st.success("Nenhuma divergência de valores encontrada nas colunas comuns para chaves iguais (comparando como texto).")
                else: st.info("Nenhuma chave comum após merge para comparar valores de colunas (verifique unicidade das chaves).")
            else:
                st.info("Nenhuma chave comum encontrada entre os dois arquivos para comparar valores de colunas.")

        except Exception as e:
            st.error(f"Erro durante a comparação de valores de colunas: {e}")
            # st.exception(e) # Para depuração, mostra o traceback completo

    else: 
        st.info("Selecione as colunas chave em ambos os arquivos na barra lateral para habilitar a verificação de chaves e a comparação de valores.")

# --- Exportação de Relatórios ---
if st.session_state.df1 is not None and st.session_state.uploaded_file_1_name:
    st.header("Exportação de Relatórios Consolidados")
    base_filename = f"auditoria_{st.session_state.uploaded_file_1_name.split('.')[0]}"
    if st.session_state.df2 is not None and st.session_state.uploaded_file_2_name:
        base_filename += f"_vs_{st.session_state.uploaded_file_2_name.split('.')[0]}"

    col_rep1, col_rep2 = st.columns(2)
    with col_rep1:
        excel_bytes = generate_excel_report(
            st.session_state.df1, st.session_state.uploaded_file_1_name,
            st.session_state.df2, st.session_state.uploaded_file_2_name,
            st.session_state.report_data_comparison if st.session_state.df2 is not None else None,
            st.session_state.report_data_df1,
            st.session_state.report_data_df2 if st.session_state.df2 is not None else None
        )
        st.download_button(
            label="Download Relatório Excel (.xlsx)",
            data=excel_bytes,
            file_name=f"{base_filename}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_excel_report"
        )
    with col_rep2:
        md_bytes = generate_markdown_report(
            st.session_state.df1, st.session_state.uploaded_file_1_name,
            st.session_state.df2, st.session_state.uploaded_file_2_name,
            st.session_state.report_data_comparison if st.session_state.df2 is not None else None,
            st.session_state.report_data_df1,
            st.session_state.report_data_df2 if st.session_state.df2 is not None else None
        )
        st.download_button(
            label="Download Relatório Markdown (.md)",
            data=md_bytes,
            file_name=f"{base_filename}.md",
            mime="text/markdown",
            key="download_markdown_report"
        )
else:
    st.info("Carregue e processe pelo menos um arquivo para gerar relatórios.")

st.sidebar.markdown("---") # Linha separadora
# Usando HTML para centralizar o conteúdo
st.sidebar.markdown(
    """
    <div style='text-align: center;'>
        <strong>VeriDados</strong><br>
        <small>Versão 1.0</small><br>
        <a href='http://www.brunnoml.com.br' target='_blank'>www.brunnoml.com.br</a>
    </div>
    """,
    unsafe_allow_html=True
)