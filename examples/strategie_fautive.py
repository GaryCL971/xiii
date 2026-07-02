"""
SPÉCIMEN — stratégie volontairement fautive, pour la démo XIII.

Ce fichier concentre les 3 fautes classiques que XIII détecte par analyse
statique. Il ne doit PAS être exécuté ni tradé : c'est l'appât de la démo.
Chaque faute ici est une faille RÉELLEMENT trouvée dans le labo en juin 2026.
"""
import numpy as np
import pandas as pd


def build_cpi_brick(dates):
    """Brique CPI... sauf qu'elle n'a jamais vu un chiffre d'inflation."""
    # TODO: brancher les vraies données CPI
    ret = pd.Series(np.random.normal(0.0005, 0.0008, len(dates)), index=dates)  # dummy
    return ret


def build_portfolio(cpi, nasdaq):
    """Aligne une série mensuelle (CPI) sur une série quotidienne (NASDAQ)."""
    df = pd.concat([cpi, nasdaq], axis=1).dropna()   # inner-join silencieux
    return df.mean(axis=1)


def signal(close):
    """Croisement de moyenne… qui regarde le futur deux fois."""
    ma = close.rolling(100, center=True).mean()      # fenêtre centrée = moitié future
    pos = (close > ma).astype(float).shift(-1)       # aligne la position de demain sur aujourd'hui
    return pos
