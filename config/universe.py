"""
Stock universe definition for Brazilian market pairs trading.
"""

IBOV_TICKERS = [
    'ABEV3.SA', 'AZUL4.SA', 'B3SA3.SA', 'BBAS3.SA', 'BBDC3.SA', 'BBDC4.SA',
    'BBSE3.SA', 'BPAC11.SA', 'BRAP4.SA', 'BRDT3.SA', 'BRFS3.SA',
    'BRKM5.SA', 'BRML3.SA', 'CCRO3.SA', 'CIEL3.SA', 'CMIG4.SA',
    'CMIN3.SA', 'COGN3.SA', 'CPFE3.SA', 'CRFB3.SA', 'CSAN3.SA', 'CSNA3.SA',
    'CVCB3.SA', 'CYRE3.SA', 'ECOR3.SA', 'EGIE3.SA', 'ELET3.SA', 'ELET6.SA',
    'EMBR3.SA', 'ENBR3.SA', 'ENGI11.SA', 'EQTL3.SA', 'EZTC3.SA', 'FLRY3.SA',
    'GGBR4.SA', 'GNDI3.SA', 'GOAU4.SA', 'GOLL4.SA', 'HAPV3.SA',
    'HYPE3.SA', 'IRBR3.SA', 'ITSA4.SA', 'ITUB4.SA', 'JBSS3.SA',
    'JHSF3.SA', 'KLBN11.SA', 'LREN3.SA', 'LWSA3.SA', 'MGLU3.SA',
    'MRFG3.SA', 'MRVE3.SA', 'MULT3.SA', 'NTCO3.SA', 'PCAR3.SA', 'PETR3.SA',
    'PETR4.SA', 'PRIO3.SA', 'QUAL3.SA', 'RADL3.SA', 'RAIL3.SA', 'RENT3.SA',
    'SANB11.SA', 'SBSP3.SA', 'SUZB3.SA', 'TAEE11.SA',
    'TOTS3.SA', 'UGPA3.SA', 'USIM5.SA', 'VALE3.SA', 'VIVT3.SA', 'WEGE3.SA',
    'YDUQ3.SA'
]

IBRX100_ADDITIONAL = [
    'ALOS3.SA', 'ALPA4.SA', 'ALSO3.SA', 'ALUP11.SA', 'AMER3.SA', 'ARML3.SA',
    'ASAI3.SA', 'AURE3.SA', 'BIDI11.SA', 'BPAN4.SA', 'BRAP3.SA', 'CAML3.SA',
    'CASH3.SA', 'CBAV3.SA', 'CMIN3.SA', 'COCE5.SA', 'CSMG3.SA', 'CXSE3.SA',
    'DXCO3.SA', 'ENEV3.SA', 'EUCA4.SA', 'EVEN3.SA', 'FRAS3.SA', 'GFSA3.SA',
    'GRND3.SA', 'IFCM3.SA', 'INTB3.SA', 'IRBR3.SA', 'JALL3.SA', 'KEPL3.SA',
    'LIGT3.SA', 'MDIA3.SA', 'MILS3.SA', 'MOVI3.SA', 'NEOE3.SA', 'ODPV3.SA',
    'OMGE3.SA', 'ONCO3.SA', 'ORVR3.SA', 'PETZ3.SA', 'PLPL3.SA', 'POMO4.SA',
    'POSI3.SA', 'RDOR3.SA', 'RECV3.SA', 'RRRP3.SA', 'SAPR11.SA', 'SEQL3.SA',
    'SHUL4.SA', 'SLCE3.SA', 'SMFT3.SA', 'SMTO3.SA', 'STBP3.SA', 'TEND3.SA',
    'TGMA3.SA', 'TIMS3.SA', 'TRIS3.SA', 'TUPY3.SA', 'UNIP6.SA', 'USIM3.SA',
    'VAMO3.SA', 'VBBR3.SA', 'VIIA3.SA', 'VULC3.SA', 'WEST3.SA', 'WIZS3.SA'
]

SECTOR_MAPPING = {
    'FINANCIAL': [
        'BBAS3.SA', 'BBDC3.SA', 'BBDC4.SA', 'BBSE3.SA', 'BPAC11.SA', 'BPAN4.SA',
        'ITSA4.SA', 'ITUB4.SA', 'SANB11.SA', 'B3SA3.SA', 'BIDI11.SA'
    ],
    'ENERGY': [
        'PETR3.SA', 'PETR4.SA', 'PRIO3.SA', 'ENBR3.SA', 'ELET3.SA', 'ELET6.SA',
        'CMIG4.SA', 'CPFE3.SA', 'EGIE3.SA', 'ENGI11.SA', 'EQTL3.SA', 'SBSP3.SA',
        'TAEE11.SA', 'ENEV3.SA'
    ],
    'MATERIALS': [
        'VALE3.SA', 'CSNA3.SA', 'GGBR4.SA', 'GOAU4.SA', 'USIM5.SA', 'USIM3.SA',
        'BRAP4.SA', 'BRAP3.SA', 'KLBN11.SA', 'SUZB3.SA'
    ],
    'INDUSTRIALS': [
        'WEGE3.SA', 'EMBR3.SA', 'RAIL3.SA', 'CCRO3.SA', 'ECOR3.SA', 'AZUL4.SA',
        'GOLL4.SA', 'RENT3.SA'
    ],
    'CONSUMER_STAPLES': [
        'ABEV3.SA', 'BRFS3.SA', 'JBSS3.SA', 'BEEF3.SA', 'MRFG3.SA', 'SMTO3.SA'
    ],
    'CONSUMER_DISCRETIONARY': [
        'MGLU3.SA', 'LREN3.SA', 'BTOW3.SA', 'CVCB3.SA', 'COGN3.SA', 'YDUQ3.SA',
        'MULT3.SA', 'HYPE3.SA', 'PCAR3.SA', 'ASAI3.SA', 'LWSA3.SA'
    ],
    'HEALTHCARE': [
        'FLRY3.SA', 'GNDI3.SA', 'RADL3.SA', 'HAPV3.SA', 'QUAL3.SA'
    ],
    'REAL_ESTATE': [
        'CYRE3.SA', 'EZTC3.SA', 'MRVE3.SA', 'JHSF3.SA', 'BRML3.SA', 'EVEN3.SA',
        'GFSA3.SA', 'HGTX3.SA', 'MOVI3.SA'
    ],
    'TELECOMMUNICATIONS': [
        'VIVT3.SA', 'TOTS3.SA'
    ],
    'TECHNOLOGY': [
        'MDIA3.SA', 'WIZS3.SA'
    ],
    'UTILITIES': [
        'CSAN3.SA', 'CSMG3.SA', 'SAPR11.SA', 'SULA11.SA'
    ]
}

UNIVERSE_DEFINITIONS = {
    'IBOV': IBOV_TICKERS,
    'IBRX100': IBOV_TICKERS + IBRX100_ADDITIONAL,
    'ALL': IBOV_TICKERS + IBRX100_ADDITIONAL
}

def get_universe_tickers(universe_name='IBOV'):
    """Get list of tickers for specified universe."""
    return UNIVERSE_DEFINITIONS.get(universe_name.upper(), IBOV_TICKERS)

def get_sector_tickers(sector_name):
    """Get list of tickers for specified sector."""
    return SECTOR_MAPPING.get(sector_name.upper(), [])

def get_ticker_sector(ticker):
    """Get sector for specified ticker."""
    for sector, tickers in SECTOR_MAPPING.items():
        if ticker in tickers:
            return sector
    return 'UNKNOWN'