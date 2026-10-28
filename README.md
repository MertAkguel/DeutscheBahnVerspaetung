# DB-Verspätungsprognose (db-delay-mlops)

Ein End-to-End-MLOps-Projekt, das vorhersagt, ob ein Zug der Deutschen Bahn an einem Bahnhof **mindestens 6 Minuten zu spät ankommt**. Es entsteht als Abschlussprojekt des [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) von DataTalks.Club.

## Projektidee

- **Frage:** Kommt ICE 123 heute um 17:42 in Köln Hbf pünktlich an?
- **ML-Aufgabe:** Binäre Klassifikation mit Wahrscheinlichkeit. Eine Zeile entspricht einem Halt eines Zuges an einem Bahnhof.
- **Zielvariable:** Ankunftsverspätung ≥ 6 Minuten. Die Schwelle übernimmt die offizielle Pünktlichkeitsdefinition der DB: Ein Halt gilt als pünktlich, wenn er weniger als 6 Minuten verspätet ist.
- **Prognosezeitpunkt:** am Vortag. Das Modell nutzt nur Informationen, die zu diesem Zeitpunkt schon bekannt sind, zum Beispiel die Wettervorhersage statt des tatsächlichen Wetters.
- **Warum das Thema zu MLOps passt:** Jeden Monat kommen neue Daten hinzu, und die tatsächliche Verspätung ist danach bekannt. Streiks, Winter und Baustellen sorgen für echten Drift. Daraus ergeben sich Deployment, Monitoring und Retraining.

## Aktueller Stand

Stand 28.10.2026: Das Projekt ist angelegt, die Umsetzung beginnt.

| Baustein | Werkzeug (geplant) | Status |
| --- | --- | --- |
| Problembeschreibung | dieses README | 🔄 in Arbeit |
| Datenanalyse und Baseline | Jupyter, pandas | ⬜ offen |
| Experiment Tracking und Model Registry | MLflow | ⬜ offen |
| Workflow-Orchestrierung | Orchestrator aus Modul 3 | ⬜ offen |
| Modell-Deployment | FastAPI, Docker | ⬜ offen |
| Monitoring mit Retraining bei Drift | Evidently, Grafana | ⬜ offen |
| Cloud und Infrastructure as Code | Terraform | ⬜ offen |
| Best Practices | pytest, Linter, pre-commit, Makefile, CI/CD | ⬜ offen |

## Voraussetzungen

**Software**

| Werkzeug | Version | Wofür |
| --- | --- | --- |
| Linux, macOS oder Windows 11 mit WSL2 (Ubuntu 24.04) | – | Entwicklungsumgebung |
| Python | 3.12 | Projektsprache |
| [uv](https://docs.astral.sh/uv/) | ≥ 0.12 | Pakete und virtuelle Umgebung |
| Git | – | Versionskontrolle |
| Docker mit Docker Compose | – | Container für API, MLflow und Monitoring |

**Konten und Zugänge**

- Für die Hauptdaten ist **kein Konto nötig**. Datensatz, Wetter und Feiertage sind frei abrufbar.
- Optional: ein kostenloser API-Key vom [DB API Marketplace](https://developers.deutschebahn.com/) für die Timetables- und StaDa-API. Er wird nur für Plan B gebraucht (siehe unten).
- Ein Cloud-Konto kommt erst in der Cloud-Phase hinzu.

**Speicherplatz:** Eine Monatsdatei ist bis Oktober 2025 etwa 60 MB groß, ab November 2025 etwa 340 MB. Für mehrere Monate sollten ein paar GB frei sein.

## Daten

| Quelle | Inhalt | Zugang | Lizenz |
| --- | --- | --- | --- |
| [piebro/deutsche-bahn-data](https://huggingface.co/datasets/piebro/deutsche-bahn-data) (Hugging Face) | Plan- und Ist-Zeiten aller Halte, eine Parquet-Datei pro Monat (`monthly_processed_data/data-YYYY-MM.parquet`), ab Juli 2024 | frei, ohne Anmeldung | Daten CC BY 4.0 (Quelle: Deutsche Bahn), Code MIT |
| [Open-Meteo Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api) | frühere Wettervorhersagen je Bahnhof, ab Januar 2024 | frei, ohne API-Key, nicht-kommerziell | CC BY 4.0 |
| [`holidays`](https://pypi.org/project/holidays/) (Python-Paket) | gesetzliche Feiertage je Bundesland | offline | siehe Paket |
| [OpenHolidays API](https://www.openholidaysapi.org/en/) | Schulferien je Bundesland | frei | siehe Website |
| [db-stations](https://github.com/derhuerst/db-stations) | Bahnhöfe mit Koordinaten und EVA-Nummer | frei | siehe Repository |

**Wie der Bahn-Datensatz entsteht:** Laut [GitHub-README](https://github.com/piebro/deutsche-bahn-data) fragt ein GitHub-Actions-Job viermal täglich die Timetables-API der DB ab (Endpunkte `plan` und `fchg`) und verarbeitet die Rohantworten einmal im Monat zu Tabellen.

**Abdeckung:** Bis zum 02.11.2025 wurden nur die rund 100 größten Bahnhöfe gesammelt, danach alle. Dieses Projekt beschränkt sich auf diese rund 100 Bahnhöfe. So ist die Historie ab Juli 2024 einheitlich.

**Wetter ohne Leakage:** Für das Training nutzen wir die Vorhersage, die am Vortag gemacht wurde (Open-Meteo `_previous_day1`), nicht das gemessene Wetter. So sieht das Modell im Training dieselben Informationen wie später im Betrieb.

**Plan B:** Wird der Datensatz eingestellt, sammelt das Projekt die Daten selbst über die [DB Timetables-API](https://developers.deutschebahn.com/) mit einem kostenlosen Key. Scraping von bahn.de ist nicht vorgesehen.

## Quellenangabe

Bahndaten: Deutsche Bahn, aufbereitet von [piebro/deutsche-bahn-data](https://github.com/piebro/deutsche-bahn-data), CC BY 4.0. Wetterdaten: [Open-Meteo.com](https://open-meteo.com/), CC BY 4.0.
