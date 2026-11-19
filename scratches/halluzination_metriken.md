# Metriken für „halluzinationsfrei" — Taxonomie, Grenzen, operative Wahl

Ausgangsdefinition (Nutzer): *Halluzination ist, was durchs Training nicht
gerechtfertigt ist und was der Prompt (und die Projektion des Prompts auf das
Training) nicht hergibt.*

Das ist eine **Provenienz**-Aussage, keine Wahrheits-Aussage. Sie hat eine
harte Konsequenz, die man offen legen muss: ein **memorierter Irrtum** ist
durchs Training gerechtfertigt und fällt *nicht* unter diese Definition.
„Halluzinationsfrei" heißt in dieser Fassung also „deckungsfrei" — nicht
„wahr".

## Drei Ebenen, drei verschiedene Prüfungen

| Ebene | Was fehlt | Womit prüfbar |
| :--- | :--- | :--- |
| **U — ungedeckt** | weder Gewicht noch Prompt stützen den Anspruch | Provenienz-Audit (kontrafaktisch, ohne externes Wissen) |
| **W-falsch** | das Gewicht stützt ihn, er ist aber falsch | nur extern (Retrieval, Fakten-DB, NLI) |
| **C-falsch** | der Prompt stützt ihn, der Prompt ist falsch | Prüfung der Quelle selbst |

Jede Metrik, die „wahr" behauptet, muss Ebene 2 und 3 lösen — dafür gibt es
kein modellinternes Signal. Das Provenienz-Audit ist die *notwendige*, nicht
die hinreichende Bedingung.

## Kandidaten (mit Bruchstelle)

1. **Provenienz-Deckung (implementiert, MI-10).**
   `H = 1 − max(J_W, J_C)` mit
   `J_W = Stabilität über Paraphrasen` und
   `J_C = Beleg im Prompt vorhanden ∧ Behauptung == Beleg`.
   *Stärke:* keine Labels, keine Zusatzmodelle, ein Generierungslauf.
   *Bruchstelle:* **stabile Konfabulation** (das Modell erfindet für dieselbe
   Nicht-Entität in allen Formulierungen denselben Hauptort) → falscher
   Negativ. Genau deshalb ist die Nonce-Kontrolle der Prüfstein.

2. **Kontrafaktische Entitäts-Abhängigkeit.**
   Ersetze das Subjekt (France → Germany) und miss, ob der Anspruch
   mitwandert. Ein gedeckter Anspruch ist *entitätsabhängig*; ein
   template-getriebener Anspruch nicht.
   *Bruchstelle:* Nonces sind ebenfalls entitätsabhängig (jede Nicht-Entität
   erzeugt eine andere Erfindung) → allein untauglich, nur als Zusatzachse.

3. **Beleg-Attribution (Retrieval/NLI).**
   Ist der Anspruch aus dem Kontext *ableitbar* (Substring, Paraphrase,
   Entailment)? Das ist die semantisch stärkste Fassung von `J_C`.
   *Bruchstelle:* braucht ein NLI-/Retrieval-Modell — eine zweite Fehlerquelle.

4. **Selbstkonsistenz unter Sampling.**
   K Samples, Agreement des Anspruchs.
   *Bruchstelle:* misst Konfidenz, nicht Deckung; konfundiert mit Temperatur.

5. **Kalibrierungs-Defekt (ECE, logit-gap).**
   *Bruchstelle:* misst Truth-Aptitude, nicht Provenienz — genau die
   Verwechslung, die die Definition vermeidet.

6. **Interner Provenienz-Readout (MI-9).**
   Ein Forward-Pass, Position auf `d_ctx`/`d_s`, Klasse W/C/S.
   *Stärke:* potenziell **einpass** statt kontrafaktisch.
   *Bruchstelle:* die Achsen sind marker-getragen (MI-8); sie lesen
   „woher kommt der Zustand", nicht „ist der Anspruch gedeckt". Erst als
   *Prädiktor* des externen Audits brauchbar — das ist der nächste Schritt.

## Die operative Wahl

MI-10 = **1 + 3-light**: Stabilität (Ebene 1, ohne Zusatzmodell) plus
Beleg-Gleichheit (Substring-Fassung von `J_C`). Das ist billig, hat eine
ehrliche Nonce-Grundwahrheit und einen präregistrierten Falsifikator
(stabile Konfabulation).

Der nächste, stärkere Schritt ist die **Verknüpfung**: MI-10 liefert das
externe Label pro Anspruch, MI-9 den internen Zustand an derselben
Token-Koordinate. Wenn der interne `d_ctx`-Score das externe Label
vorhersagt, hat man einen **einpassigen Halluzinations-Flag** — das ist die
eigentlich interessante Metrik, weil sie ohne Kontrafaktik auskommt.

## Was der Bau gelehrt hat (Rev-1 → Rev-3)

1. **Der Claim liegt tief.** Im Chat-Modus antwortet das Modell
   „The capital of France is **Paris**." — der Inhalt steht bei Token 5–7.
   Ein Audit am ersten Token misst nur Funktionswörter. Generationsbasiert
   messen, dann den Anspruch extrahieren.
2. **Die Behauptungs-Gleichheit ist der schwache Teil.** Im
   Vervollständigungs-Format degeneriert Argon bei 5 von 8 Codes
   („ is is is is", „the answer. answer.") — die *Emission* trägt den Beleg
   nicht. Der MI-9-Monitor dagegen trennt C von I mit 1.000 (p .009)
   **unabhängig von der Emission**. Konsequenz: `J_C` intern messen
   (Klasse C am letzten Prompt-Token), nicht am generierten Text.
3. **Die Konfabulation ist instabil — der Falsifikator feuert nicht.**
   Derselbe Nonce („Vrandel") liefert über vier Formulierungen Skopje,
   Helsinki, London. Genau das ist die empirische Stütze für das
   Stabilitäts-Signal: der Anspruch überlebt die Oberflächenform nicht.
   Wäre er stabil, wäre `J_W` falsch positiv — deshalb ist H2 der
   registrierte Prüfstein.
4. **AUROC nur über Ansprüche.** Eine Zelle ohne Anspruch hat keinen
   Score; sie als „verpasste Detektion" zu zählen, ist ein
   Auswertungsfehler, kein Metrik-Befund.

## Ergebnis MI-10 (Rev-4, exit 0 GRÜN)

| Band | Argon 1B | Parent |
| :--- | :--- | :--- |
| Falschalarme auf memoriertem Wissen | 8/8 stabil | 8/8 stabil |
| Konfabulation erkannt (Nonce + Cue) | 7/8 | 8/8 |
| Kontext-Deckung (behavioral / intern) | 8/8 / 8/8 | 8/8 / 8/8 |
| AUROC (nur Ansprüche) | 0.938 | 1.000 |

Der eine Fehlalarm in Argon ist die registrierte Falsifikator-Klasse:
**Oxwald → London, stabil in 3/4 Formulierungen**. Genau dort endet die
Stabilitäts-Heuristik; wer sie schließen will, braucht die Beleg-Achse
(Retrieval/NLI) — Ebene W-falsch bleibt modellintern unlösbar.

**Prompt-Familien-Bindung:** der interne `J_C` (MI-9-Monitor) gilt nur in
der kalibrierten Formulierung; der behaviorale `J_C` ist portabel.
