import pandas as pd

#a,K1,E1,,K2,E2,,K3,E3,,K4,E4,,K5,E5,,K6,E6
# Charger le fichier CSV original
fichier_source = 'data\DATA.csv'
fichier_destination = 'data\korean_english_dataset.csv'

# Lire le fichier CSV avec pandas
df = pd.read_csv(fichier_source)

# Liste des colonnes que tu veux prendre (groupées par paires)
colonnes_choisies1 = ['E1', 'K1']  # Groupe 1
colonnes_choisies2 = ['E2', 'K2']  # Groupe 2
# colonnes_choisies3 = ['E3', 'K3']  # Groupe 3
# colonnes_choisies4 = ['E4', 'K4']  # Groupe 4
# colonnes_choisies5 = ['E5', 'K5']  # Groupe 5
# colonnes_choisies6 = ['E6', 'K6']  # Groupe 6

# Créer une DataFrame vide pour stocker les colonnes combinées
df_combined = pd.DataFrame()

# Ajouter les colonnes de chaque groupe si elles ne sont pas vides
for group in [colonnes_choisies1, colonnes_choisies2]:
    group_df = df[group].dropna(how='all')  # Ne garder que les colonnes non vides 

    # Ajouter des guillemets autour de chaque valeur
    group_df = group_df.astype(str).apply(lambda x: x.map(lambda val: f'"{val}"' if pd.notnull(val) else val))

    # Ajouter les colonnes non vides à la DataFrame combinée
    if not group_df.empty:
        # Ouvrir le fichier en mode append, avec encodage UTF-8 pour éviter les erreurs d'encodage
        with open(fichier_destination, 'a', newline='', encoding='utf-8') as f:
            group_df.to_csv(f, header=False, index=False)

print(f"Les colonnes non vides avec des guillemets ont été ajoutées à {fichier_destination}.")