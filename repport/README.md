# Rapport d'Analyse des Clusters par Filtrage Directionnel

Ce dossier contient les scripts et ressources nécessaires pour générer un rapport détaillé sur l'analyse des clusters par filtrage directionnel des connexions.

## Contenu du dossier

- `analyze_points.py` : Script pour analyser les ensembles de points et générer les graphiques
- `update_report.py` : Script pour mettre à jour le rapport HTML avec les graphiques générés
- `generate_pdf.py` : Script pour générer le rapport final au format PDF
- `rapport.html` : Modèle du rapport en HTML
- `plots/` : Dossier où seront stockés les graphiques générés

## Comment générer le rapport

1. **Analyser les points et générer les graphiques**

   ```bash
   python analyze_points.py
   ```

   Ce script analyse les trois ensembles de points enregistrés dans le dossier `points/` et génère les graphiques suivants pour chaque ensemble :
   - Méthode du coude pour déterminer le nombre optimal de clusters
   - Visualisation des connexions à chaque étape du filtrage
   - Résultats du clustering avec notre méthode vs K-means standard
   - Comparaison des temps de convergence
   
   Les graphiques sont automatiquement enregistrés dans le dossier `plots/`.

2. **Mettre à jour le rapport HTML**

   ```bash
   python update_report.py
   ```

   Ce script vérifie que tous les graphiques nécessaires ont été générés et met à jour le rapport HTML.

3. **Générer le rapport PDF**

   ```bash
   python generate_pdf.py
   ```

   Ce script génère le rapport final au format PDF à partir du fichier HTML.

## Dépendances

Les scripts dépendent des bibliothèques Python suivantes :
- NumPy
- Matplotlib
- scikit-learn
- BeautifulSoup4 (pour la mise à jour du HTML)
- WeasyPrint (pour la génération du PDF)

## Structure du rapport

Le rapport contient les sections suivantes :
1. Introduction au problème de clustering et à l'importance de l'initialisation
2. Description détaillée de notre méthode de filtrage directionnel
3. Analyse des résultats sur trois jeux de données, avec comparaison à l'algorithme K-means standard
4. Conclusion mettant en évidence les avantages de notre approche

## Auteurs

- Boumedine Billal
- Addel Tolbat

Date : 12 avril, 2025
