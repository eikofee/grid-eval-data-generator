# Génération de données pour l'évaluation de visualisation

Les fichiers sont fournis tels quels et n'ont pas été vraiment retravaillés pour être facilement lisibles.

Le fichier `nnSsmDemo.cpp` est le code du générateur dont le fonctionnement est décrit dans ma thèse (Chapitre 6).
Il a besoin de `wulip` pour fonctionner (demander à David).

La génération des données en elle-même est produite dans la fonction `generateDataSafe` qui est un peu commentée sur les grandes étapes de la génération.

A part ça, le programme permet aussi d'afficher des projections dans R^2 ou dans N^2 (coordonnées grilles).
Il peut aussi calculer les distances dans R^N en se basant sur le voisinage R/N^2, et enfin il peut exporter ces images en .ppm (bitmap), voir le fichier ci-dessous pour les convertir en png.

Le fichier `dataGenerationScript.py` permet de générer (à paramétrer/commenter/décommenter dans le .py) des ensembles de données projetées à l'aide du programme ci-dessus avec SSM (implémentation de David) et t-SNE (script python externe dans le dossier `python`, il doit être placé dans le même dossier que le programme `nnSsmDemo`).
Il peut être parcouru pour avoir des exemples d'utilisation de nnSsmDemo.
