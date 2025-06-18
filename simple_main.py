"""
Sistema de Pairs Trading Simplificado
"""
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.simple_data import SimpleDataManager, update_ibov_data
from strategy.cointegration import CointegrationTester
from simple_logger import logger

def update_data(period="2y"):
    """Atualiza dados do Yahoo Finance."""
    print("🔄 Atualizando dados do Yahoo Finance...")
    print("📝 Nota: Isto SUBSTITUI todos os dados antigos (Yahoo ajusta série histórica)")
    
    results = update_ibov_data(period)
    
    # Mostra estatísticas
    data_manager = SimpleDataManager()
    info = data_manager.get_database_info()
    
    success_count = sum(results.values())
    total_count = len(results)
    
    print(f"\n✅ Atualização completa!")
    print(f"📊 {success_count}/{total_count} símbolos baixados com sucesso")
    print(f"💾 {info['total_records']:,} registros no banco")
    print(f"📅 Período: {info['min_date']} a {info['max_date']}")
    print(f"🗃️ Tamanho: {info['database_size']:.1f} MB")

def find_pairs():
    """Busca pares cointegrados."""
    print("🔍 Buscando pares cointegrados...")
    
    data_manager = SimpleDataManager()
    symbols = data_manager.get_available_symbols()
    
    if len(symbols) < 2:
        print("❌ Precisa de pelo menos 2 símbolos no banco de dados")
        print("💡 Execute: python simple_main.py --update")
        return
    
    print(f"📊 Testando {len(symbols)} símbolos...")
    
    tester = CointegrationTester()
    cointegrated_pairs = []
    
    # Testa todas as combinações
    from itertools import combinations
    
    for i, (symbol1, symbol2) in enumerate(combinations(symbols, 2)):
        if i % 50 == 0:  # Progress
            print(f"   Testado {i} pares...")
        
        # Pega dados do par
        pair_data = data_manager.get_pair_data(symbol1, symbol2)
        
        if len(pair_data) < 252:  # Precisa de pelo menos 1 ano
            continue
        
        # Testa cointegração
        result = tester.test_pair_cointegration(pair_data[symbol1], pair_data[symbol2])
        
        if result.get('is_cointegrated', False):
            cointegrated_pairs.append({
                'pair': f"{symbol1}-{symbol2}",
                'pvalue': result['coint_pvalue'],
                'half_life': result['half_life'],
                'correlation': result['correlation'],
                'hedge_ratio': result['hedge_ratio']
            })
    
    # Ordena por p-value (menor é melhor)
    cointegrated_pairs.sort(key=lambda x: x['pvalue'])
    
    print(f"\n🎯 Encontrados {len(cointegrated_pairs)} pares cointegrados!")
    
    if cointegrated_pairs:
        print("\n📋 Top 10 pares:")
        print("Pair                P-Value   Half-Life  Correlation  Hedge Ratio")
        print("-" * 65)
        
        for pair in cointegrated_pairs[:10]:
            print(f"{pair['pair']:<15} {pair['pvalue']:.4f}    {pair['half_life']:.1f}d      {pair['correlation']:.3f}       {pair['hedge_ratio']:.3f}")
    
    return cointegrated_pairs

def show_status():
    """Mostra status do sistema."""
    print("📊 Status do Sistema")
    print("=" * 50)
    
    data_manager = SimpleDataManager()
    info = data_manager.get_database_info()
    
    if info['total_symbols'] == 0:
        print("❌ Nenhum dado encontrado")
        print("💡 Execute: python simple_main.py --update")
        return
    
    print(f"✅ Símbolos no banco: {info['total_symbols']}")
    print(f"📈 Total de registros: {info['total_records']:,}")
    print(f"📅 Período dos dados: {info['min_date']} a {info['max_date']}")
    print(f"🕒 Última atualização: {info['last_update']}")
    print(f"💾 Tamanho do banco: {info['database_size']:.1f} MB")
    
    # Mostra alguns símbolos disponíveis
    symbols = data_manager.get_available_symbols()
    print(f"\n📋 Símbolos disponíveis (mostrando 10):")
    for symbol in symbols[:10]:
        print(f"  • {symbol}")
    
    if len(symbols) > 10:
        print(f"  ... e mais {len(symbols) - 10} símbolos")

def launch_dashboard():
    """Lança dashboard simples."""
    print("🚀 Lançando dashboard...")
    print("📱 Acesse: http://localhost:8501")
    
    try:
        import subprocess
        subprocess.run(["streamlit", "run", "simple_dashboard.py"])
    except FileNotFoundError:
        print("❌ Streamlit não encontrado")
        print("💡 Instale: pip install streamlit")
    except Exception as e:
        print(f"❌ Erro: {e}")

def main():
    """Função principal."""
    parser = argparse.ArgumentParser(description="Sistema de Pairs Trading Simplificado")
    
    parser.add_argument("--update", action="store_true", 
                       help="Atualizar dados do Yahoo Finance")
    parser.add_argument("--pairs", action="store_true",
                       help="Buscar pares cointegrados")
    parser.add_argument("--status", action="store_true",
                       help="Mostrar status do sistema")
    parser.add_argument("--dashboard", action="store_true",
                       help="Lançar dashboard")
    parser.add_argument("--period", default="2y",
                       help="Período dos dados (1y, 2y, 5y, max)")
    parser.add_argument("--all", action="store_true",
                       help="Executar tudo (update + pairs)")
    
    args = parser.parse_args()
    
    # Se nenhum argumento, mostra ajuda
    if not any([args.update, args.pairs, args.status, args.dashboard, args.all]):
        parser.print_help()
        return
    
    print("🎯 Sistema de Pairs Trading - Versão Simplificada")
    print("=" * 60)
    
    try:
        if args.all or args.update:
            update_data(args.period)
        
        if args.all or args.pairs:
            find_pairs()
        
        if args.status:
            show_status()
        
        if args.dashboard:
            launch_dashboard()
    
    except KeyboardInterrupt:
        print("\n⚠️ Processo interrompido pelo usuário")
    except Exception as e:
        print(f"❌ Erro: {e}")
        logger.error(f"Erro no main: {e}")

if __name__ == "__main__":
    main()