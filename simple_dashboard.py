"""
Dashboard simplificado para Pairs Trading
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

from data.simple_data import SimpleDataManager, update_ibov_data
from strategy.cointegration import CointegrationTester

# Configuração da página
st.set_page_config(
    page_title="Pairs Trading - Simples",
    page_icon="📈", 
    layout="wide"
)

@st.cache_data(ttl=300)  # Cache por 5 minutos
def load_data_manager():
    """Carrega o gerenciador de dados (com cache)."""
    return SimpleDataManager()

@st.cache_data(ttl=300)
def get_database_info():
    """Informações do banco (com cache)."""
    dm = load_data_manager()
    return dm.get_database_info()

def main():
    st.title("🎯 Pairs Trading - Sistema Simplificado")
    st.markdown("**Cointegração no mercado brasileiro com dados do Yahoo Finance**")
    
    # Sidebar
    st.sidebar.title("📊 Controles")
    
    # Botão de atualizar dados
    if st.sidebar.button("🔄 Atualizar Dados Yahoo", type="primary"):
        with st.spinner("Baixando dados do Yahoo Finance..."):
            results = update_ibov_data("2y")
            success_count = sum(results.values())
            total_count = len(results)
            
            if success_count > 0:
                st.sidebar.success(f"✅ {success_count}/{total_count} símbolos atualizados!")
                st.rerun()
            else:
                st.sidebar.error("❌ Falha na atualização")
    
    # Status do banco
    info = get_database_info()
    
    if info['total_symbols'] == 0:
        st.warning("🚨 Nenhum dado encontrado! Clique em 'Atualizar Dados Yahoo' primeiro.")
        return
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Símbolos", info['total_symbols'])
    
    with col2:
        st.metric("Registros", f"{info['total_records']:,}")
    
    with col3:
        st.metric("Período", f"{info['min_date']} a {info['max_date']}")
    
    with col4:
        st.metric("Tamanho DB", f"{info['database_size']:.1f} MB")
    
    st.divider()
    
    # Abas principais
    tab1, tab2, tab3 = st.tabs(["🔍 Análise de Pares", "📊 Dados", "⚙️ Config"])
    
    with tab1:
        render_pair_analysis()
    
    with tab2:
        render_data_tab()
    
    with tab3:
        render_config_tab()

def render_pair_analysis():
    """Aba de análise de pares."""
    st.header("🔍 Análise de Cointegração")
    
    dm = load_data_manager()
    symbols = dm.get_available_symbols()
    
    if len(symbols) < 2:
        st.warning("Precisa de pelo menos 2 símbolos para análise de pares")
        return
    
    # Seleção de pares
    col1, col2 = st.columns(2)
    
    with col1:
        symbol1 = st.selectbox("Primeira ação", symbols, index=0)
    
    with col2:
        symbol2 = st.selectbox("Segunda ação", symbols, index=1 if len(symbols) > 1 else 0)
    
    if symbol1 == symbol2:
        st.warning("Selecione duas ações diferentes")
        return
    
    # Período de análise
    col1, col2 = st.columns(2)
    with col1:
        days_back = st.slider("Dias para análise", 252, 1000, 500)
    
    with col2:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
    
    # Botão de análise
    if st.button("🧮 Testar Cointegração", type="primary"):
        with st.spinner(f"Testando cointegração {symbol1} vs {symbol2}..."):
            
            # Carrega dados
            pair_data = dm.get_pair_data(symbol1, symbol2, 
                                       start_date.strftime('%Y-%m-%d'),
                                       end_date.strftime('%Y-%m-%d'))
            
            if pair_data.empty:
                st.error("Não foi possível carregar dados para o par selecionado")
                return
            
            if len(pair_data) < 100:
                st.warning(f"Poucos dados disponíveis: {len(pair_data)} dias")
                return
            
            # Testa cointegração
            tester = CointegrationTester()
            result = tester.test_pair_cointegration(pair_data[symbol1], pair_data[symbol2])
            
            # Mostra resultados
            display_cointegration_results(result, symbol1, symbol2, pair_data)

def display_cointegration_results(result, symbol1, symbol2, pair_data):
    """Exibe resultados da cointegração."""
    
    st.subheader(f"📊 Resultados: {symbol1} vs {symbol2}")
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        is_coint = result.get('is_cointegrated', False)
        st.metric("Cointegrado?", "✅ Sim" if is_coint else "❌ Não")
    
    with col2:
        pvalue = result.get('coint_pvalue', 1.0)
        st.metric("P-Value", f"{pvalue:.4f}")
    
    with col3:
        half_life = result.get('half_life', 0)
        st.metric("Half-Life", f"{half_life:.1f} dias")
    
    with col4:
        correlation = result.get('correlation', 0)
        st.metric("Correlação", f"{correlation:.3f}")
    
    # Gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        # Preços normalizados
        fig = go.Figure()
        
        # Normaliza preços para base 100
        norm1 = pair_data[symbol1] / pair_data[symbol1].iloc[0] * 100
        norm2 = pair_data[symbol2] / pair_data[symbol2].iloc[0] * 100
        
        fig.add_trace(go.Scatter(x=norm1.index, y=norm1, name=symbol1, line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=norm2.index, y=norm2, name=symbol2, line=dict(color='red')))
        
        fig.update_layout(title="Preços Normalizados (Base 100)", 
                         xaxis_title="Data", yaxis_title="Preço Normalizado")
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Spread
        hedge_ratio = result.get('hedge_ratio', 1.0)
        intercept = result.get('intercept', 0)
        spread = pair_data[symbol1] - hedge_ratio * pair_data[symbol2] - intercept
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=spread.index, y=spread, name='Spread', line=dict(color='green')))
        fig.add_hline(y=spread.mean(), line_dash="dash", line_color="red", annotation_text="Média")
        
        fig.update_layout(title="Spread do Par", 
                         xaxis_title="Data", yaxis_title="Spread")
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Z-Score
    if not spread.empty:
        z_score = (spread - spread.mean()) / spread.std()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=z_score.index, y=z_score, name='Z-Score', line=dict(color='purple')))
        
        # Linhas de sinal
        fig.add_hline(y=2, line_dash="dash", line_color="green", annotation_text="Entry Long")
        fig.add_hline(y=-2, line_dash="dash", line_color="green", annotation_text="Entry Short")
        fig.add_hline(y=0.5, line_dash="dot", line_color="orange", annotation_text="Exit")
        fig.add_hline(y=-0.5, line_dash="dot", line_color="orange")
        
        fig.update_layout(title="Z-Score do Spread", 
                         xaxis_title="Data", yaxis_title="Z-Score")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Sinal atual
        current_z = z_score.iloc[-1]
        
        if abs(current_z) >= 2:
            signal = "ENTRY LONG" if current_z <= -2 else "ENTRY SHORT" 
            color = "green" if current_z <= -2 else "red"
            st.markdown(f"""
            <div style="background-color: {color}; color: white; padding: 1rem; border-radius: 0.5rem; text-align: center; font-weight: bold;">
                🚨 SINAL: {signal} (Z-Score: {current_z:.2f})
            </div>
            """, unsafe_allow_html=True)
        elif abs(current_z) <= 0.5:
            st.info(f"💭 Sem sinal - Z-Score neutro: {current_z:.2f}")
        else:
            st.warning(f"⏳ Aguardando - Z-Score: {current_z:.2f}")

def render_data_tab():
    """Aba de dados."""
    st.header("📊 Dados Disponíveis")
    
    dm = load_data_manager()
    symbols = dm.get_available_symbols()
    
    if not symbols:
        st.warning("Nenhum símbolo disponível")
        return
    
    # Tabela de símbolos
    st.subheader("📋 Símbolos no Banco")
    
    # Cria dataframe com informações dos símbolos
    symbol_info = []
    
    for symbol in symbols[:20]:  # Mostra apenas os primeiros 20
        data = dm.get_price_data(symbol)
        if not data.empty:
            symbol_info.append({
                'Símbolo': symbol,
                'Registros': len(data),
                'Primeira Data': data.index.min().strftime('%Y-%m-%d'),
                'Última Data': data.index.max().strftime('%Y-%m-%d'),
                'Último Preço': f"R$ {data['close'].iloc[-1]:.2f}"
            })
    
    if symbol_info:
        df = pd.DataFrame(symbol_info)
        st.dataframe(df, use_container_width=True)
        
        if len(symbols) > 20:
            st.info(f"Mostrando 20 de {len(symbols)} símbolos disponíveis")
    
    # Gráfico de exemplo
    if symbols:
        st.subheader("📈 Visualização de Preço")
        
        selected_symbol = st.selectbox("Selecione um símbolo", symbols)
        
        if selected_symbol:
            data = dm.get_price_data(selected_symbol)
            
            if not data.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data.index, y=data['close'], 
                                       name=selected_symbol, line=dict(color='blue')))
                
                fig.update_layout(title=f"Preço de {selected_symbol}",
                                xaxis_title="Data", yaxis_title="Preço (R$)")
                
                st.plotly_chart(fig, use_container_width=True)

def render_config_tab():
    """Aba de configuração."""
    st.header("⚙️ Configurações")
    
    # Configurações de dados
    st.subheader("💾 Gerenciamento de Dados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        period = st.selectbox("Período para download", 
                            options=["1y", "2y", "5y", "max"],
                            index=1)
        
        if st.button("🔄 Atualizar com Período Personalizado"):
            with st.spinner(f"Baixando dados ({period})..."):
                results = update_ibov_data(period)
                success_count = sum(results.values())
                total_count = len(results)
                st.success(f"✅ {success_count}/{total_count} símbolos atualizados!")
    
    with col2:
        if st.button("🗑️ Limpar Banco de Dados", type="secondary"):
            if st.button("⚠️ Confirmar Limpeza"):
                dm = load_data_manager()
                dm.clear_database()
                st.success("🗑️ Banco de dados limpo!")
                st.rerun()
    
    # Parâmetros de cointegração
    st.subheader("🧮 Parâmetros de Cointegração")
    
    st.info("""
    **Interpretação dos resultados:**
    
    - **P-Value < 0.05**: Par é cointegrado (estatisticamente significante)
    - **Half-Life**: Tempo médio para o spread retornar à média (em dias)
    - **Correlação**: Correlação linear entre as ações (-1 a 1)
    - **Z-Score**: Número de desvios-padrão da média
        - |Z| > 2: Sinal de entrada
        - |Z| < 0.5: Sinal de saída
    """)

if __name__ == "__main__":
    main()