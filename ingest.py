"""
ingest.py — Point d'entrée pour construire l'index vectoriel.

Usage : python ingest.py
"""

from rag import build_index

if __name__ == "__main__":
    build_index()
    print("Index construit avec succès.")