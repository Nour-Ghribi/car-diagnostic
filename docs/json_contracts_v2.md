# Contrats JSON V2 de l'Agent IA

## 1. Objectif du document

Ce document définit les contrats JSON de référence pour la V2 du système de diagnostic automobile intelligent basé sur un agent IA côté serveur. Il a pour but de fournir une base commune pour :
- l'analyse des requêtes utilisateur ;
- l'orchestration des tools backend ;
- la structuration des réponses de l'agent ;
- la validation future via des modèles Pydantic.

Les contrats ci-dessous respectent les contraintes fonctionnelles suivantes :
- intents autorisés uniquement : `READ_DTC`, `CHECK_CYLINDER`, `CHECK_ENGINE_HEALTH`, `CHECK_SIGNAL_STATUS`, `EXPLAIN_WARNING_LIGHT`, `GET_VEHICLE_CONTEXT`, `UNKNOWN` ;
- signaux autorisés uniquement : `rpm`, `engine_load`, `coolant_temp`, `throttle_pos`, `stft_b1`, `ltft_b1`, `o2_b1s1`, `vehicle_speed`, `module_voltage` ;
- paramètres normalisés : `detail`, `cylinder_index`, `include_pending` ;
- `mode06` reste optionnel uniquement ;
- `missing_data` est obligatoire dans la réponse agent ;
- `raw_data_used` doit lister explicitement les signaux réellement utilisés ;
- `confidence` doit être comprise entre `0` et `1`.

## 2. Contrat `intent`

### Structure JSON

```json
{
  "name": "CHECK_CYLINDER",
  "confidence": 0.9,
  "parameters": {
    "cylinder_index": 2,
    "detail": "medium"
  }
}
```

### Description des champs

- `name`
  - type : `string`
  - obligatoire : oui
  - valeurs autorisées :
    - `READ_DTC`
    - `CHECK_CYLINDER`
    - `CHECK_ENGINE_HEALTH`
    - `CHECK_SIGNAL_STATUS`
    - `EXPLAIN_WARNING_LIGHT`
    - `GET_VEHICLE_CONTEXT`
    - `UNKNOWN`
  - rôle : nom de l'intent détecté par l'agent.

- `parameters`
  - type : `object`
  - obligatoire : oui
  - rôle : paramètres normalisés associés à l'intent.
  - remarque : le contenu varie selon l'intent, mais les clés doivent rester métier et ne jamais contenir de commande OBD brute.

- `confidence`
  - type : `number`
  - obligatoire : oui
  - contrainte : valeur comprise entre `0` et `1`
  - rôle : niveau de confiance de détection de l'intent par l'agent.

### Paramètres normalisés autorisés par intent

- `READ_DTC`
  - `include_pending` : `boolean`, optionnel, défaut `true`
  - `include_permanent` : `boolean`, optionnel, défaut `false`

- `CHECK_CYLINDER`
  - `cylinder_index` : `integer`, obligatoire, minimum `1`
  - `bank` : `integer`, optionnel
  - `detail` : `string`, optionnel, valeurs `low | medium | high`, défaut `medium`

- `CHECK_ENGINE_HEALTH`
  - `detail` : `string`, optionnel, valeurs `low | medium | high`, défaut `medium`

- `CHECK_SIGNAL_STATUS`
  - `signal` : `string`, obligatoire, parmi :
    - `rpm`
    - `engine_load`
    - `coolant_temp`
    - `throttle_pos`
    - `stft_b1`
    - `ltft_b1`
    - `o2_b1s1`
    - `vehicle_speed`
    - `module_voltage`
  - `max_age_ms` : `integer`, optionnel, minimum `0`

- `EXPLAIN_WARNING_LIGHT`
  - `warning_type` : `string`, optionnel, seule valeur supportée en V2 : `check_engine`

- `GET_VEHICLE_CONTEXT`
  - `include_calibration` : `boolean`, optionnel, défaut `true`
  - `refresh_capabilities` : `boolean`, optionnel, défaut `false`

- `UNKNOWN`
  - aucun paramètre métier requis

## 3. Contrat `agent_request`

### Structure JSON

```json
{
  "request_id": "req_20260326_0001",
  "ts": "2026-03-26T15:10:00Z",
  "vehicle_id": "veh_001",
  "session_id": "sess_abc123",
  "user_prompt": "Check cylinder 2",
  "locale": "fr",
  "constraints": {
    "allow_on_demand": true,
    "max_latency_ms": 6000,
    "max_tool_calls": 8
  },
  "prefetch": {
    "signals": {
      "rpm": {
        "value": 820,
        "unit": "RPM",
        "observed_ts": "2026-03-26T15:09:58Z",
        "source": "push_cache"
      }
    },
    "dtc": {
      "stored": ["P0302"],
      "pending": [],
      "permanent": []
    }
  }
}
```

### Description des champs

- `request_id`
  - type : `string`
  - obligatoire : oui
  - rôle : identifiant unique de la requête agent.

- `ts`
  - type : `string` au format `date-time`
  - obligatoire : oui
  - rôle : horodatage de création de la requête.

- `vehicle_id`
  - type : `string`
  - obligatoire : oui
  - rôle : identifiant fonctionnel du véhicule ou du boîtier côté backend.

- `session_id`
  - type : `string | null`
  - obligatoire : non
  - rôle : identifiant de session applicative.

- `user_prompt`
  - type : `string`
  - obligatoire : oui
  - rôle : requête utilisateur en langage naturel.

- `locale`
  - type : `string`
  - obligatoire : non
  - défaut recommandé : `fr`
  - rôle : langue préférée de réponse.

- `constraints`
  - type : `object | null`
  - obligatoire : non
  - rôle : contraintes d'exécution agent.

- `constraints.allow_on_demand`
  - type : `boolean`
  - rôle : autorise ou non les lectures on-demand côté backend.

- `constraints.max_latency_ms`
  - type : `integer`
  - rôle : budget de latence total de la requête.

- `constraints.max_tool_calls`
  - type : `integer`
  - rôle : nombre maximum d'appels tools autorisés.

- `prefetch`
  - type : `object | null`
  - obligatoire : non
  - rôle : données injectées directement dans la requête pour tests offline ou optimisation.

- `prefetch.signals`
  - type : `object`
  - rôle : instantané de signaux déjà disponibles.

- `prefetch.signals.<signal>`
  - type : `object`
  - rôle : mesure d'un signal autorisé.

- `prefetch.signals.<signal>.value`
  - type : `number | string | boolean | null`
  - rôle : valeur observée.

- `prefetch.signals.<signal>.unit`
  - type : `string | null`
  - rôle : unité de la valeur.

- `prefetch.signals.<signal>.observed_ts`
  - type : `string` au format `date-time`
  - rôle : horodatage de la mesure.

- `prefetch.signals.<signal>.source`
  - type : `string`
  - valeurs recommandées : `push_cache`, `db`, `on_demand`
  - rôle : provenance de la donnée.

- `prefetch.dtc`
  - type : `object | null`
  - rôle : DTC déjà connus au moment de la requête.

- `prefetch.dtc.stored`
  - type : `array[string]`
  - rôle : codes défaut confirmés.

- `prefetch.dtc.pending`
  - type : `array[string]`
  - rôle : codes défaut pending.

- `prefetch.dtc.permanent`
  - type : `array[string]`
  - rôle : codes défaut permanents.

## 4. Contrat `tool_request`

### Structure JSON

```json
{
  "request_id": "req_20260326_0001",
  "tool_name": "get_latest_signals",
  "vehicle_id": "veh_001",
  "signal_keys": ["rpm", "engine_load", "coolant_temp"],
  "options": {
    "max_age_ms": 2000,
    "mode06_optional": true,
    "include_pending": true
  }
}
```

### Description des champs

- `request_id`
  - type : `string`
  - obligatoire : oui
  - rôle : identifiant logique permettant de relier l'appel tool à la requête agent.

- `tool_name`
  - type : `string`
  - obligatoire : oui
  - valeurs recommandées :
    - `get_dtcs`
    - `get_latest_signals`
    - `request_fresh_signals`
    - `get_vehicle_context`
    - `request_mode06`
  - rôle : nom du tool backend appelé.

- `vehicle_id`
  - type : `string`
  - obligatoire : oui
  - rôle : cible de l'appel backend.

- `signal_keys`
  - type : `array[string]`
  - obligatoire : non
  - rôle : liste des signaux métier demandés.
  - contrainte : uniquement les signaux autorisés de la V2.

- `options`
  - type : `object`
  - obligatoire : non
  - rôle : paramètres techniques ou fonctionnels associés au tool.

- `options.max_age_ms`
  - type : `integer`
  - rôle : fraîcheur maximale acceptable.

- `options.mode06_optional`
  - type : `boolean`
  - rôle : indique explicitement que `mode06` est optionnel.

- `options.include_pending`
  - type : `boolean`
  - rôle : indique si les DTC pending doivent être inclus.

- `options.detail`
  - type : `string`
  - valeurs : `low | medium | high`
  - rôle : niveau de détail demandé.

- `options.cylinder_index`
  - type : `integer`
  - minimum : `1`
  - rôle : cylindre ciblé pour `CHECK_CYLINDER`.

## 5. Contrat `tool_response`

### Structure JSON

```json
{
  "request_id": "req_20260326_0001",
  "tool_name": "get_latest_signals",
  "status": "ok",
  "data": {
    "signals": {
      "rpm": {
        "value": 820,
        "unit": "RPM",
        "observed_ts": "2026-03-26T15:09:58Z",
        "source": "push_cache"
      }
    },
    "dtc": {
      "stored": ["P0302"],
      "pending": [],
      "permanent": []
    },
    "mode06": null
  },
  "missing_data": [
    {
      "key": "o2_b1s1",
      "reason": "unsupported"
    }
  ],
  "error_message": null
}
```

### Description des champs

- `request_id`
  - type : `string`
  - obligatoire : oui
  - rôle : identifiant de corrélation avec `tool_request`.

- `tool_name`
  - type : `string`
  - obligatoire : oui
  - rôle : nom du tool ayant produit la réponse.

- `status`
  - type : `string`
  - obligatoire : oui
  - valeurs autorisées :
    - `ok`
    - `partial`
    - `error`
  - rôle : état d'exécution du tool.

- `data`
  - type : `object | null`
  - obligatoire : oui
  - rôle : charge utile renvoyée par le tool.
  - structure attendue :
    - `signals` : objet de signaux normalisés ;
    - `dtc` : objet de DTC structurés ;
    - `mode06` : objet ou `null`, toujours optionnel en V2.

- `data.signals`
  - type : `object`
  - rôle : valeurs de signaux métier renvoyées par le tool.

- `data.dtc`
  - type : `object | null`
  - rôle : ensemble structuré des DTC retournés par le tool.

- `data.mode06`
  - type : `object | null`
  - rôle : données `mode06` si disponibles ; ce champ reste optionnel d'un point de vue métier et peut valoir `null`.

- `missing_data`
  - type : `array[object]`
  - obligatoire : oui
  - rôle : liste normalisée des données manquantes ou indisponibles.

- `missing_data[].key`
  - type : `string`
  - rôle : donnée absente ou non exploitable.

- `missing_data[].reason`
  - type : `string`
  - valeurs recommandées :
    - `unsupported`
    - `not_collected`
    - `stale`
    - `timeout`
    - `no_data`
  - rôle : raison de l'absence.

- `error_message`
  - type : `string | null`
  - obligatoire : oui
  - rôle : message d'erreur lisible si `status = error`.

## 6. Contrat `agent_response`

### Structure JSON

```json
{
  "request_id": "req_20260326_0001",
  "ts": "2026-03-26T15:10:02Z",
  "vehicle_id": "veh_001",
  "intent": {
    "name": "CHECK_CYLINDER",
    "confidence": 0.9,
    "parameters": {
      "cylinder_index": 2,
      "detail": "medium"
    }
  },
  "diagnosis": "Les éléments disponibles suggèrent une suspicion modérée sur le cylindre 2, principalement à cause d'un DTC P0302 et d'un mélange corrigé positivement.",
  "confidence": 0.72,
  "evidence": [
    {
      "key": "rpm",
      "label": "Régime moteur",
      "value": 820,
      "unit": "RPM",
      "observed_ts": "2026-03-26T15:09:58Z",
      "source": "push_cache"
    },
    {
      "key": "dtc.stored",
      "label": "Codes défaut confirmés",
      "value": ["P0302"],
      "unit": null,
      "observed_ts": "2026-03-26T15:09:57Z",
      "source": "db"
    }
  ],
  "signals_used": ["rpm", "engine_load", "coolant_temp", "stft_b1", "o2_b1s1"],
  "actions_taken": ["get_dtcs", "get_latest_signals", "request_mode06"],
  "missing_data": [
    {
      "key": "mode06",
      "reason": "unsupported",
      "impact": "confidence_reduced"
    }
  ],
  "recommendations": [
    "Vérifier l'état de la bougie ou de l'injecteur du cylindre 2.",
    "Compléter avec une lecture fraîche si le véhicule est disponible."
  ]
}
```

### Description des champs

- `request_id`
  - type : `string`
  - obligatoire : oui
  - rôle : identifiant unique de la réponse agent.

- `ts`
  - type : `string` au format `date-time`
  - obligatoire : oui
  - rôle : horodatage de production de la réponse.

- `vehicle_id`
  - type : `string`
  - obligatoire : oui
  - rôle : véhicule concerné par le diagnostic.

- `intent`
  - type : `object`
  - obligatoire : oui
  - rôle : intent retenu par l'agent avec ses paramètres normalisés.

- `diagnosis`
  - type : `string`
  - obligatoire : oui
  - rôle : diagnostic textuel synthétique lisible par l'utilisateur.

- `confidence`
  - type : `number`
  - obligatoire : oui
  - contrainte : valeur comprise entre `0` et `1`
  - rôle : estimation globale de confiance du diagnostic.

- `evidence`
  - type : `array[object]`
  - obligatoire : oui
  - rôle : éléments de preuve structurés utilisés pour produire la conclusion.

- `evidence[].key`
  - type : `string`
  - rôle : identifiant de l'élément de preuve, par exemple `rpm` ou `dtc.stored`.

- `evidence[].label`
  - type : `string`
  - rôle : libellé lisible de la preuve.

- `evidence[].value`
  - type : `number | string | boolean | array | object | null`
  - rôle : valeur brute ou interprétée.

- `evidence[].unit`
  - type : `string | null`
  - rôle : unité si applicable.

- `evidence[].observed_ts`
  - type : `string` au format `date-time`
  - rôle : date d'observation de la preuve.

- `evidence[].source`
  - type : `string`
  - rôle : provenance de la preuve.

- `signals_used`
  - type : `array[string]`
  - obligatoire : oui
  - rôle : liste explicite des signaux réellement utilisés par l'agent.
  - contrainte : uniquement les signaux autorisés en V2.

- `actions_taken`
  - type : `array[string]`
  - obligatoire : oui
  - rôle : liste ordonnée des actions ou tools effectivement exécutés par l'agent pour produire le diagnostic.

- `missing_data`
  - type : `array[object]`
  - obligatoire : oui
  - rôle : données manquantes, non supportées, trop anciennes ou non disponibles.

- `missing_data[].impact`
  - type : `string`
  - obligatoire : oui
  - rôle : effet explicite de l'absence de la donnée sur la réponse finale, par exemple `confidence_reduced`.

- `recommendations`
  - type : `array[string]`
  - obligatoire : non
  - rôle : pistes d'action ou vérifications complémentaires.

## 7. Remarques de conception

- Le contrat `intent` reste volontairement générique avec un objet `parameters` pour simplifier l'évolution future.
- Les tools backend manipulent exclusivement des concepts métier et non des commandes OBD brutes.
- `UNKNOWN` est un intent officiel et non un cas d'erreur implicite.
- `mode06` n'est jamais requis pour conclure ; il peut uniquement renforcer ou affaiblir la confiance si disponible.
- `signals_used`, `actions_taken` et `missing_data` sont obligatoires pour améliorer la traçabilité, l'auditabilité et l'explicabilité côté agent.
