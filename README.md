# PE_tennis_de_table
PE tennis de table 29a


Le fichier python "sequencage_audio.py" permet de découper un audio en sous-audio de durée souhaitée (ici 0.1 sec). Cela permet de réaliser les calculs de cross-corélation 
uniquement sur des durées faibles et de rejeter les parties où le rebond n'a pas lieu. 

Le calcul de cross-corélation permet de savoir, grâce au spectre du signal, si deux audios sont relativement similaires. Ce calcul est fait grâce à l'algorithme 
'autocorrelation_audio.py"

Le programme "distance_pearson.py" est une méthode alternative de la cross-corrélation qui ne se base pas sur le spectre mais directement sur l'audio :
La distance de Pearson est un coefficient de corrélation qui mesure la similarité linéaire entre deux séries de données. Plus le coefficient est proche de 1, plus les deux séries de données sont similaires. Plus il est proche de 0, plus les deux séries de données sont dissemblables. Si le coefficient est négatif, cela signifie que les deux séries de données ont une relation négative, c'est-à-dire que lorsqu'une série de données augmente, l'autre série de données diminue.

Le script lit les deux fichiers audio et calcule le coefficient de corrélation de Pearson entre les deux signaux. Si les fichiers audio ont des fréquences d'échantillonnage différentes, le script affiche un message d'erreur et s'arrête. Si les fichiers audio ont la même fréquence d'échantillonnage, le script calcule la distance de Pearson et l'affiche à l'écran.

\textbf{Partie vidéo:}
\textbf{L'objectif} de la partie vidéo est d'avoir en entrée une vidéo de rebonds de balle de pingpong, et de pouvoir retourner en sortie une liste contenant les instants de chaque rebond.\\
Ainsi pour atteindre cet objectif, on répartit le travail en \textbf{3 tâches}: la soustraction d'image, le tracking de centre de gravité et la détéction du changement de trajectoire.


