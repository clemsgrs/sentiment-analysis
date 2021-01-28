#!/usr/bin/env bash
wget -O data/FinancialPhraseBank-v10.zip "https://www.researchgate.net/profile/Pekka_Malo/publication/251231364_FinancialPhraseBank-v10/data/0c96051eee4fb1d56e000000/FinancialPhraseBank-v10.zip"
unzip data/FinancialPhraseBank-v10.zip -d data/
rm -r data/__MACOSX/
rm data/FinancialPhraseBank-v10.zip