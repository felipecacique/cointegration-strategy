"""
Versão simplificada do sistema de dados.
Apenas baixa OHLCV do Yahoo Finance e salva no SQLite.
"""
import yfinance as yf
import pandas as pd
import sqlite3
import os
from datetime import datetime, timedelta
from typing import List
from simple_logger import logger

class SimpleDataManager:
    """Gerenciador de dados simplificado."""
    
    def __init__(self, db_path: str = "database/simple_data.db"):
        self.db_path = db_path
        self._create_database()
    
    def _create_database(self):
        """Cria banco de dados simples."""
        # Cria diretório se não existir
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # Tabela simples: só OHLCV
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prices (
                    symbol TEXT,
                    date DATE,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    PRIMARY KEY (symbol, date)
                )
            """)
            
            # Tabela para controle de atualizações
            conn.execute("""
                CREATE TABLE IF NOT EXISTS updates (
                    symbol TEXT PRIMARY KEY,
                    last_update TIMESTAMP
                )
            """)
            
            conn.commit()
    
    def download_symbol(self, symbol: str, period: str = "2y") -> bool:
        """
        Baixa dados de um símbolo e SUBSTITUI no banco.
        
        Args:
            symbol: Símbolo da ação (ex: 'PETR4.SA')
            period: Período dos dados ('1y', '2y', '5y', 'max')
        
        Returns:
            True se sucesso, False se erro
        """
        try:
            logger.info(f"Baixando dados para {symbol}")
            
            # Baixa dados do Yahoo Finance
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, auto_adjust=True)  # auto_adjust=True já inclui splits/dividendos
            
            if data.empty:
                logger.warning(f"Nenhum dado retornado para {symbol}")
                return False
            
            # Prepara dados para inserção
            data = data.reset_index()
            data['symbol'] = symbol
            data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
            
            # Seleciona apenas as colunas que precisamos
            data = data[['symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            data.columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
            
            # Remove dados antigos e insere novos (SUBSTITUI tudo)
            with sqlite3.connect(self.db_path) as conn:
                # Deleta dados antigos do símbolo
                conn.execute("DELETE FROM prices WHERE symbol = ?", (symbol,))
                
                # Insere dados novos
                data.to_sql('prices', conn, if_exists='append', index=False)
                
                # Atualiza timestamp
                conn.execute("""
                    INSERT OR REPLACE INTO updates (symbol, last_update) 
                    VALUES (?, ?)
                """, (symbol, datetime.now()))
                
                conn.commit()
            
            logger.info(f"✅ {symbol}: {len(data)} registros salvos")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro baixando {symbol}: {e}")
            return False
    
    def download_universe(self, symbols: List[str], period: str = "2y") -> dict:
        """
        Baixa dados para uma lista de símbolos.
        
        Args:
            symbols: Lista de símbolos
            period: Período dos dados
            
        Returns:
            Dicionário com resultados {symbol: success}
        """
        results = {}
        total = len(symbols)
        
        logger.info(f"🚀 Iniciando download de {total} símbolos")
        
        for i, symbol in enumerate(symbols, 1):
            print(f"📊 {i}/{total} - {symbol}")
            results[symbol] = self.download_symbol(symbol, period)
            
            # Pequena pausa para não sobrecarregar o Yahoo Finance
            import time
            time.sleep(0.1)
        
        # Estatísticas
        success_count = sum(results.values())
        fail_count = total - success_count
        
        logger.info(f"✅ Download completo: {success_count} sucessos, {fail_count} falhas")
        
        return results
    
    def get_price_data(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Recupera dados de preço para um símbolo.
        
        Args:
            symbol: Símbolo da ação
            start_date: Data inicial (YYYY-MM-DD)
            end_date: Data final (YYYY-MM-DD)
            
        Returns:
            DataFrame com dados de preço
        """
        query = "SELECT * FROM prices WHERE symbol = ?"
        params = [symbol]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        
        query += " ORDER BY date"
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)
        
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        return df
    
    def get_pair_data(self, symbol1: str, symbol2: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Recupera dados para um par de ações (para cointegração).
        
        Returns:
            DataFrame com close prices das duas ações
        """
        data1 = self.get_price_data(symbol1, start_date, end_date)
        data2 = self.get_price_data(symbol2, start_date, end_date)
        
        if data1.empty or data2.empty:
            return pd.DataFrame()
        
        # Combina dados usando close price
        pair_data = pd.DataFrame({
            symbol1: data1['close'],
            symbol2: data2['close']
        }).dropna()  # Remove datas onde alguma ação não tem dados
        
        return pair_data
    
    def get_available_symbols(self) -> List[str]:
        """Retorna lista de símbolos disponíveis no banco."""
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute("SELECT DISTINCT symbol FROM prices ORDER BY symbol").fetchall()
        
        return [row[0] for row in result]
    
    def get_database_info(self) -> dict:
        """Retorna informações sobre o banco de dados."""
        with sqlite3.connect(self.db_path) as conn:
            # Contagem total de registros
            total_records = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
            
            # Contagem de símbolos
            total_symbols = conn.execute("SELECT COUNT(DISTINCT symbol) FROM prices").fetchone()[0]
            
            # Data range
            date_range = conn.execute("""
                SELECT MIN(date) as min_date, MAX(date) as max_date 
                FROM prices
            """).fetchone()
            
            # Última atualização
            last_update = conn.execute("""
                SELECT MAX(last_update) FROM updates
            """).fetchone()[0]
        
        return {
            'total_records': total_records,
            'total_symbols': total_symbols,
            'min_date': date_range[0],
            'max_date': date_range[1],
            'last_update': last_update,
            'database_size': os.path.getsize(self.db_path) / (1024*1024) if os.path.exists(self.db_path) else 0
        }
    
    def clear_database(self):
        """Limpa todo o banco de dados."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM prices")
            conn.execute("DELETE FROM updates")
            conn.commit()
        
        logger.info("🗑️ Banco de dados limpo")

# Função helper para uso fácil
def update_ibov_data(period: str = "2y") -> dict:
    """
    Atualiza dados do IBOV de forma simples.
    
    Args:
        period: Período dos dados ('1y', '2y', '5y', 'max')
    
    Returns:
        Resultados do download
    """
    # Lista simplificada do IBOV (principais ações)
    IBOV_SYMBOLS = [
        'PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'BBAS3.SA',
        'ABEV3.SA', 'WEGE3.SA', 'MGLU3.SA', 'LREN3.SA', 'SUZB3.SA',
        'GGBR4.SA', 'CSNA3.SA', 'USIM5.SA', 'GOAU4.SA', 'B3SA3.SA',
        'RENT3.SA', 'ELET3.SA', 'CMIG4.SA', 'CSAN3.SA', 'CPFE3.SA',
        'SBSP3.SA', 'TAEE11.SA', 'EGIE3.SA', 'ENGI11.SA', 'EQTL3.SA',
        'CCRO3.SA', 'RAIL3.SA', 'AZUL4.SA', 'GOLL4.SA', 'EMBR3.SA',
        'JBSS3.SA', 'BRFS3.SA', 'MRFG3.SA', 'RADL3.SA', 'FLRY3.SA',
        'GNDI3.SA', 'HAPV3.SA', 'QUAL3.SA', 'COGN3.SA', 'YDUQ3.SA'
    ]
    
    data_manager = SimpleDataManager()
    return data_manager.download_universe(IBOV_SYMBOLS, period)

if __name__ == "__main__":
    # Teste simples
    print("🎯 Sistema de Dados Simplificado")
    print("Baixando dados do IBOV...")
    
    results = update_ibov_data("1y")
    
    data_manager = SimpleDataManager()
    info = data_manager.get_database_info()
    
    print(f"\n📊 Resumo:")
    print(f"Símbolos: {info['total_symbols']}")
    print(f"Registros: {info['total_records']:,}")
    print(f"Período: {info['min_date']} a {info['max_date']}")
    print(f"Tamanho DB: {info['database_size']:.1f} MB")