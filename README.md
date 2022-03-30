# Description du projet

Ce projet Github est un composant de la plate-forme pédagogique réalisée dans le cadre du projet Ephemer (https://lillethics.com/projet-ephemer/). Cette plate-forme est constituée de 3 composants conçus pour fonctionner ensemble :

- une app Django pour le frontend : `https://github.com/io-craft-org/ephemer`
- un oTree project pour le backend : `https://github.com/io-craft-org/ephemer-otree`
- un moteur oTree modifié : c'est ce projet !

Ce projet est un fork de oTree 5.4.1 (15/09/2021) : https://pypi.org/project/otree/5.4.1/. La modification est l'ajout de nouveaux endpoints REST permettant au front Django de démarrer des sessions, de monitorer les participants et de récupérer les résultats.

Ce projet est utilisé par `ephemer-otree` et sera récupéré et installé automatiquement comme les autres dépendances.
