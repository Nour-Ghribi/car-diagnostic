# Use Cases V2 de l'Agent IA

## 1. Objectif du document

Ce document définit les cas d'usage fonctionnels de l'agent IA dans l'architecture V2 du système de diagnostic automobile intelligent. Il a pour objectif de formaliser un périmètre réaliste, démontrable et techniquement cohérent avec une architecture `tool-based` côté serveur.

Dans cette V2, l'agent IA ne remplace ni le boîtier embarqué ni la logique d'accès OBD bas niveau. Son rôle est d'interpréter une demande utilisateur en langage naturel, d'identifier l'intention métier, de sélectionner les signaux utiles via des outils backend contrôlés, puis de produire une réponse diagnostique structurée, lisible et traçable.

Ce document sert de base commune pour :
- l'équipe backend ;
- l'équipe embarquée ;
- la conception des tools exposés à l'agent ;
- le cadrage du périmètre démontrable en V2 ;
- la préparation ultérieure des schémas d'entrée/sortie et des tests d'acceptation.

## 2. Périmètre fonctionnel de l'agent

### 2.1 Ce que l'agent doit faire en V2

L'agent IA côté serveur doit :
- recevoir un prompt utilisateur en langage naturel en français ou en anglais simple ;
- identifier l'intention principale de la demande ;
- convertir cette demande en besoin métier exprimé en concepts de diagnostic ;
- demander les données nécessaires via des tools backend ;
- exploiter des données issues du cache, de la base ou d'une lecture on-demand contrôlée ;
- raisonner sur les signaux disponibles, les codes défauts et le contexte véhicule ;
- produire un diagnostic textuel structuré, compréhensible et justifié ;
- indiquer explicitement les limites du diagnostic si certaines données sont manquantes, trop anciennes ou non supportées ;
- renvoyer une estimation de confiance ou, à défaut, une formulation prudente et nuancée.

### 2.2 Ce que l'agent ne doit pas faire en V2

L'agent ne doit pas :
- envoyer directement des commandes OBD brutes ;
- générer librement des PIDs, modes ou commandes ELM327 ;
- agir comme un outil de réparation automatique ;
- effacer des codes défauts ou réinitialiser le véhicule ;
- fournir des conclusions absolues lorsque les données sont partielles ;
- sortir du périmètre du diagnostic moteur/OBD défini pour la V2 ;
- se substituer à un technicien pour des opérations physiques ou mécaniques.

### 2.3 Principe d'architecture retenu

La V2 repose sur une séparation stricte des responsabilités :
- le boîtier embarqué lit les données du véhicule ;
- le backend reçoit, stocke et expose des tools ;
- l'agent raisonne uniquement sur des concepts métier tels que `rpm`, `engine_load`, `coolant_temp`, `dtc`, `o2_b1s1`, `stft_b1` ou `ltft_b1`.

Les détails OBD bas niveau restent encapsulés dans la couche backend. Cette contrainte améliore la sécurité, la traçabilité et la maintenabilité du système.

## 3. Liste officielle des intents V2

La liste officielle des intents retenus pour la V2 est la suivante :

- `READ_DTC`
- `CHECK_CYLINDER`
- `CHECK_ENGINE_HEALTH`
- `CHECK_SIGNAL_STATUS`
- `EXPLAIN_WARNING_LIGHT`
- `GET_VEHICLE_CONTEXT`
- `UNKNOWN`

Ces intents couvrent un socle démontrable et suffisamment riche pour illustrer la valeur d'un agent IA appliqué au diagnostic OBD réel, sans introduire des promesses trop ambitieuses pour une V2.

## 4. Détail complet de chaque intent

### Nom de l'intent
`READ_DTC`

### Description
Lecture et synthèse des codes défauts disponibles pour un véhicule. L'intent couvre en priorité les codes confirmés, avec possibilité d'inclure les codes pending et permanents selon le paramétrage retenu.

### Objectif métier
Permettre à l'utilisateur d'obtenir rapidement la liste des défauts connus par l'ECU et une reformulation intelligible de leur signification générale, sans devoir lire directement des codes bruts.

### Paramètres attendus
- `include_pending` : booléen, optionnel, par défaut `true`
- `include_permanent` : booléen, optionnel, par défaut `false`

### Exemples de prompts utilisateur
- "Lis les codes défauts"
- "Quels sont les DTC présents ?"
- "Read fault codes"
- "Montre-moi les codes moteur"
- "Pourquoi j'ai un défaut enregistré ?"

### Signaux/données nécessaires
- `dtc.stored`
- `dtc.pending` si activé
- `dtc.permanent` si activé
- horodatage d'observation
- source des données (`cache`, `db`, `on-demand`)

### Priorité V2 (P0 / P1 / P2)
`P0`

### Difficulté estimée
Faible à moyenne. La difficulté principale vient de la fraîcheur des données et de la gestion des différentes catégories de DTC, mais l'intent reste simple à démontrer.

### Exemple de sortie attendue (description textuelle, pas encore JSON strict)
L'agent indique les codes défauts détectés, précise s'ils sont confirmés ou pending, résume leur signification probable en langage clair et mentionne si aucun défaut n'a été observé. Il peut aussi signaler que les données sont anciennes ou incomplètes.

---

### Nom de l'intent
`CHECK_CYLINDER`

### Description
Analyse ciblée d'un cylindre particulier lorsqu'un utilisateur suspecte un raté d'allumage, un problème d'injection ou un comportement anormal lié à un cylindre donné.

### Objectif métier
Fournir un diagnostic orienté cylindre sans exposer directement la complexité OBD. L'agent doit évaluer si les signaux disponibles renforcent ou non la suspicion d'un problème sur un cylindre précis.

### Paramètres attendus
- `cylinder_index` : entier, requis, plage attendue `1..12`
- `bank` : entier, optionnel
- `detail_level` : enum, optionnel, valeurs `low | medium | high`, par défaut `medium`

### Exemples de prompts utilisateur
- "Check cylinder 2"
- "Vérifie le cylindre 3"
- "Le cylindre 1 a-t-il un problème ?"
- "Est-ce qu'il y a un misfire sur le cylindre 4 ?"
- "Analyse le cylindre 2"

### Signaux/données nécessaires
- `rpm`
- `engine_load`
- `coolant_temp`
- `throttle_pos` si disponible
- `stft_b1`
- `ltft_b1`
- `o2_b1s1`
- `dtc.stored`
- `dtc.pending`
- `mode06` explicitement optionnel, uniquement si supporté et utile
- informations de fraîcheur et de support des signaux

### Priorité V2 (P0 / P1 / P2)
`P0`

### Difficulté estimée
Élevée. Cet intent demande une logique de corrélation entre DTC, trims carburant, sonde O2, charge moteur et éventuellement Mode 06. Il nécessite aussi une gestion prudente des cas où les données par cylindre ne sont pas directement disponibles.

### Exemple de sortie attendue (description textuelle, pas encore JSON strict)
L'agent précise si les éléments observés vont dans le sens d'un problème probable sur le cylindre demandé, cite les indices principaux, mentionne les données absentes ou non supportées, et formule une conclusion prudente avec un niveau de confiance modéré si l'évidence est partielle.

---

### Nom de l'intent
`CHECK_ENGINE_HEALTH`

### Description
Évaluation synthétique de l'état global du moteur à partir des signaux live essentiels et des codes défauts disponibles.

### Objectif métier
Donner à l'utilisateur une vue d'ensemble simple sur la santé moteur, utile pour une démonstration V2, un premier niveau d'orientation ou une vérification rapide.

### Paramètres attendus
- `detail_level` : enum, optionnel, valeurs `low | medium | high`, par défaut `medium`

### Exemples de prompts utilisateur
- "Diagnostic global"
- "Le moteur est-il en bon état ?"
- "Check engine health"
- "Fais un bilan moteur"
- "Est-ce que le moteur semble sain ?"

### Signaux/données nécessaires
- `mil_status`
- `dtc.stored`
- `dtc.pending`
- `rpm`
- `engine_load`
- `coolant_temp`
- `vehicle_speed`
- `module_voltage` si disponible
- métadonnées de fraîcheur

### Priorité V2 (P0 / P1 / P2)
`P0`

### Difficulté estimée
Moyenne. L'intent demande surtout une bonne synthèse des données essentielles, des règles de cohérence simples et une formulation prudente.

### Exemple de sortie attendue (description textuelle, pas encore JSON strict)
L'agent résume si le moteur semble fonctionner normalement ou si des anomalies ressortent, mentionne les défauts actifs éventuels, cite les mesures clés utiles au diagnostic et propose éventuellement des vérifications complémentaires.

---

### Nom de l'intent
`CHECK_SIGNAL_STATUS`

### Description
Vérification ciblée d'un signal ou capteur spécifique : valeur observée, fraîcheur, disponibilité et plausibilité générale.

### Objectif métier
Permettre à l'utilisateur de poser une question simple sur un signal particulier sans demander un diagnostic complet. Cet intent sert aussi de brique de démonstration claire pour l'architecture orientée tools.

### Paramètres attendus
- `signal` : enum métier, requis, parmi la liste officiellement supportée en V2
- `max_age_ms` : entier optionnel

Liste recommandée des signaux supportés en V2 :
- `rpm`
- `coolant_temp`
- `engine_load`
- `throttle_pos`
- `stft_b1`
- `ltft_b1`
- `o2_b1s1`
- `vehicle_speed`
- `module_voltage`

Cette liste doit rester stable en V2 afin de simplifier le parsing d'intent, la validation des paramètres et la génération ultérieure des schémas JSON.

### Exemples de prompts utilisateur
- "Température d'eau ?"
- "Montre les RPM"
- "Le capteur O2 est-il OK ?"
- "Quelle est la charge moteur ?"
- "Check coolant temperature"

### Signaux/données nécessaires
- le signal demandé
- son unité
- son horodatage d'observation
- sa source
- éventuellement le contexte véhicule pour la validation de plausibilité

### Priorité V2 (P0 / P1 / P2)
`P1`

### Difficulté estimée
Faible à moyenne. L'intent est techniquement simple, mais demande de bien cadrer les valeurs attendues, la notion de fraîcheur et le catalogue des signaux réellement supportés.

### Exemple de sortie attendue (description textuelle, pas encore JSON strict)
L'agent indique la valeur du signal demandé, sa fraîcheur, si elle paraît cohérente ou non dans le contexte courant, et précise si la donnée n'est pas supportée ou indisponible.

---

### Nom de l'intent
`EXPLAIN_WARNING_LIGHT`

### Description
Explication d'un voyant d'alerte, en priorité le voyant moteur (`Check Engine` / `MIL`), à partir des DTC et d'un contexte minimal de télémétrie.

### Objectif métier
Répondre à une question utilisateur fréquente et concrète : comprendre pourquoi le voyant moteur est allumé et si le système dispose d'éléments suffisants pour l'expliquer.

### Paramètres attendus
- `warning_type` : enum, optionnel ; en V2, la seule valeur supportée doit être `check_engine`

### Exemples de prompts utilisateur
- "Pourquoi le voyant moteur est allumé ?"
- "Explain check engine light"
- "Pourquoi j'ai le Check Engine ?"
- "Que signifie ce voyant moteur ?"
- "Le voyant moteur vient de s'allumer, explique"

### Signaux/données nécessaires
- `mil_status`
- `dtc.stored`
- `dtc.pending`
- `freeze_frame` si disponible
- contexte live minimal : `rpm`, `coolant_temp`, `engine_load`, éventuellement `vehicle_speed`
- fraîcheur des informations

### Priorité V2 (P0 / P1 / P2)
`P1`

### Difficulté estimée
Moyenne. L'intent reste démontrable, mais dépend de la disponibilité des DTC et parfois du freeze frame. Une bonne gestion des cas incomplets est nécessaire.

### Exemple de sortie attendue (description textuelle, pas encore JSON strict)
L'agent explique que le voyant moteur est probablement lié à un ou plusieurs DTC observés, résume leur portée, mentionne les données de contexte disponibles et précise si l'explication reste partielle faute de données suffisantes.

---

### Nom de l'intent
`GET_VEHICLE_CONTEXT`

### Description
Récupération du contexte véhicule utile à l'interprétation des données : identité du véhicule, calibration disponible et capacités de diagnostic supportées.

### Objectif métier
Permettre d'identifier le véhicule et de savoir quelles données peuvent raisonnablement être exploitées par l'agent, en particulier les PIDs et modes disponibles.

### Paramètres attendus
- `include_calibration` : booléen, optionnel, par défaut `true`
- `refresh_capabilities` : booléen, optionnel, par défaut `false`

### Exemples de prompts utilisateur
- "Quel véhicule est connecté ?"
- "Get VIN"
- "Supported PIDs ?"
- "Donne-moi le contexte véhicule"
- "Quelles capacités OBD sont supportées ?"

### Signaux/données nécessaires
- `vin`
- `calibration_id` si disponible
- informations de dernière connexion
- capacités supportées (`supported_pids`, `supported_modes`, équivalent métier)
- état de fraîcheur de la découverte de capacités

### Priorité V2 (P0 / P1 / P2)
`P1`

### Difficulté estimée
Faible à moyenne. La difficulté est surtout liée à la découverte des capacités supportées et à leur mise en cache correcte.

### Exemple de sortie attendue (description textuelle, pas encore JSON strict)
L'agent indique le VIN ou l'identité connue du véhicule, précise les informations de calibration disponibles et liste les capacités de diagnostic utiles pour expliquer les limites ou possibilités du système.

---

### Nom de l'intent
`UNKNOWN`

### Description
Intent de repli utilisé lorsque la demande utilisateur est hors périmètre, ambiguë, trop large ou incompatible avec les capacités V2.

### Objectif métier
Éviter des réponses inventées ou trop ambitieuses. Permettre à l'agent de reformuler proprement ses limites et de réorienter l'utilisateur vers un cas d'usage supporté.

### Paramètres attendus
- aucun paramètre métier obligatoire

### Exemples de prompts utilisateur
- "Répare la voiture"
- "Fais un scan complet et corrige tout"
- "Optimise la consommation du véhicule"
- "Efface les défauts et redémarre le moteur"
- "Dis-moi exactement quelle pièce changer"

### Signaux/données nécessaires
- aucun signal spécifique
- éventuellement contexte minimal de session pour proposer une reformulation

### Priorité V2 (P0 / P1 / P2)
`P0`

### Difficulté estimée
Faible. L'enjeu principal est la robustesse fonctionnelle et la sécurité, pas la complexité algorithmique.

### Exemple de sortie attendue (description textuelle, pas encore JSON strict)
L'agent explique que la demande sort du périmètre V2 ou manque de précision, rappelle les capacités actuellement supportées et propose une reformulation exploitable, par exemple lire les DTC ou vérifier un signal donné.

## 5. Tableau récapitulatif final

| Intent | But | Paramètres | Signaux requis | Priorité |
|---|---|---|---|---|
| `READ_DTC` | Lire et résumer les codes défauts | `include_pending`, `include_permanent` | `dtc.stored`, `dtc.pending`, `dtc.permanent` | `P0` |
| `CHECK_CYLINDER` | Évaluer un cylindre suspect | `cylinder_index`, `bank`, `detail_level` | `rpm`, `engine_load`, `coolant_temp`, `throttle_pos`, `stft_b1`, `ltft_b1`, `o2_b1s1`, `dtc.*`, `mode06` optionnel | `P0` |
| `CHECK_ENGINE_HEALTH` | Donner une vue globale de la santé moteur | `detail_level` | `mil_status`, `dtc.*`, `rpm`, `engine_load`, `coolant_temp`, `vehicle_speed`, `module_voltage` | `P0` |
| `CHECK_SIGNAL_STATUS` | Vérifier un signal précis | `signal`, `max_age_ms` | signal ciblé + fraîcheur + source | `P1` |
| `EXPLAIN_WARNING_LIGHT` | Expliquer le voyant moteur | `warning_type` | `mil_status`, `dtc.*`, `freeze_frame` si disponible, contexte live minimal | `P1` |
| `GET_VEHICLE_CONTEXT` | Identifier le véhicule et ses capacités | `include_calibration`, `refresh_capabilities` | `vin`, `calibration_id`, capacités supportées, dernière activité | `P1` |
| `UNKNOWN` | Gérer les demandes hors périmètre | aucun | aucun signal spécifique | `P0` |

## 6. Recommandation V2 finale

### 6.1 Ce qu'il faut inclure absolument en V2

Le périmètre minimum recommandé pour une V2 sérieuse et démontrable inclut :
- `READ_DTC`
- `CHECK_CYLINDER`
- `CHECK_ENGINE_HEALTH`
- `UNKNOWN`

Ces intents forment un noyau fonctionnel crédible :
- `READ_DTC` démontre l'exploitation directe des défauts OBD ;
- `CHECK_CYLINDER` montre la valeur ajoutée de l'agent dans une analyse plus métier ;
- `CHECK_ENGINE_HEALTH` offre une vue synthétique très parlante pour la démonstration ;
- `UNKNOWN` sécurise le comportement de l'agent face aux demandes non supportées.

Il est également fortement recommandé d'inclure :
- `CHECK_SIGNAL_STATUS`

Même s'il peut être classé en `P1`, cet intent est simple, utile et excellent pour les démonstrations techniques et les tests.

### 6.2 Ce qu'on peut inclure si le planning le permet

Les intents suivants sont pertinents en V2 mais peuvent être finalisés après le noyau principal si le temps est limité :
- `EXPLAIN_WARNING_LIGHT`
- `GET_VEHICLE_CONTEXT`

Ils apportent une vraie valeur fonctionnelle, mais leur absence n'empêche pas de démontrer l'architecture centrale de l'agent.

### 6.3 Ce qu'il vaut mieux repousser après la V2

Il est recommandé de repousser à une phase ultérieure :
- l'effacement de DTC ;
- les recommandations de réparation trop affirmatives ;
- les diagnostics multi-systèmes trop larges ;
- les workflows de maintenance automatisée ;
- les interprétations très avancées dépendantes de données rares ou fortement constructeur.

### 6.4 Position finale recommandée

La V2 doit rester centrée sur un agent serveur capable :
- d'interpréter un besoin utilisateur réel ;
- de sélectionner les bonnes données ;
- de raisonner sur des concepts métier ;
- de produire une réponse prudente, lisible et justifiée ;
- de reconnaître ses limites lorsque les données ne suffisent pas.

Le périmètre retenu est volontairement maîtrisé afin de garantir une implémentation réaliste avec de vraies données OBD, des tools backend simples et une démonstration convaincante de la valeur de l'agent.
