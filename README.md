# MesurerRaie — mesure de raies spectrales

Application  pour ajuster et mesurer des raies spectrales à partir d'un fichier FITS 1D.

## Fonctionnalités

- Lecture d'un spectre FITS (`.fit`, `.fits`, `.fts`) avec choix de l'extension.
- Reconstruction de l'axe en longueur d'onde depuis le WCS (`CRVAL1`, `CDELT1`,
  `CRPIX1`, repli sur `CD1_1`) et conversion des unités courantes vers l'ångström.
- Ajustements : gaussienne simple, double gaussienne, profil de Voigt.
- Détection automatique de plusieurs raies d'absorption (continuum par filtre
  médian, proéminence et distance réglables).
- Ajustement robuste optionnel (`soft_l1`).
- Résultats : paramètres, incertitudes, FWHM, χ² réduit.
- Export du tableau en CSV et de la figure en PNG.
- Restrictions de la plage spectrale et lissage de détection.

## Utilisation

1. Charger un fichier FITS contenant un tableau 1D (`NAXIS = 1`).
2. Choisir le type d'analyse dans la barre latérale.
3. Restreindre éventuellement la plage spectrale autour de la raie visée.
4. Lire les résultats et télécharger le CSV ou la figure.

Si le header ne contient pas de calibration WCS, le spectre est affiché et
ajusté en pixels bruts.

## Remarques

- Aucune donnée n'est stockée : le traitement est effectué pendant la session.
- La taille maximale d'upload est fixée à 500 Mo dans
  `.streamlit/config.toml` (à adapter selon les limites de la plateforme).
- Le mode « Détection automatique » recherche des **creux d'absorption**. Pour
  une raie en émission, utiliser un ajustement sur une plage restreinte.
- `MesurerRaie_appv4.py` est le script d'origine, conservé pour référence ; il
  n'est pas utilisé par le déploiement.

## Avertissent

Cette application a été développée exclusivement à des fins pédagogiques, afin d'aider les étudiants à comprendre et à apprendre les concepts présentés. Elle n'a pas vocation à être utilisée à des fins scientifiques ni à produire des résultats destinés à la recherche scientifique.

Les résultats obtenus doivent être considérés comme des supports d'apprentissage et ne doivent pas être utilisés comme des résultats scientifiques validés.
