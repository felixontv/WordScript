# Groq API Limits für Whisper-large-v3-turbo

## Aktuelle Limits (Stand: Februar 2026)

| Limit-Typ                        | Wert   | Bedeutung                       |
| -------------------------------- | ------ | ------------------------------- |
| **RPM** (Requests Per Minute)    | 20     | Max. 20 Anfragen pro Minute     |
| **RPD** (Requests Per Day)       | 2,000  | Max. 2.000 Anfragen pro Tag     |
| **ASH** (Audio Seconds per Hour) | 7,200  | Max. 2 Stunden Audio pro Stunde |
| **ASD** (Audio Seconds per Day)  | 28,800 | Max. 8 Stunden Audio pro Tag    |

---

## Tägliche Nutzungsberechnungen

### Szenario 1: Kurze Notizen (10 Sekunden Audio)

- **Pro Anfrage:** 10 Sekunden Audio
- **Anzahl Anfragen/Tag:** 2,000 (RPD Limit)
- **Gesamtaudio/Tag:** 20,000 Sekunden = **5,5 Stunden**
- ✅ **Unter dem ASH-Limit (8 Stunden)**

### Szenario 2: Mittel-lange Aufnahmen (30 Sekunden Audio)

- **Pro Anfrage:** 30 Sekunden Audio
- **Anzahl Anfragen/Tag:** 960 (begrenzt durch 28,800 sec / 30 sec)
- **Gesamtaudio/Tag:** 28,800 Sekunden = **8 Stunden**
- ✅ **Am ASH-Limit**

### Szenario 3: Lange Diktate (60 Sekunden Audio)

- **Pro Anfrage:** 60 Sekunden Audio
- **Anzahl Anfragen/Tag:** 480 (begrenzt durch 28,800 sec / 60 sec)
- **Gesamtaudio/Tag:** 28,800 Sekunden = **8 Stunden**
- ✅ **Am ASH-Limit**

### Szenario 4: Sehr lange Aufnahmen (2 Minuten Audio)

- **Pro Anfrage:** 120 Sekunden Audio
- **Anzahl Anfragen/Tag:** 240 (begrenzt durch 28,800 sec / 120 sec)
- **Gesamtaudio/Tag:** 28,800 Sekunden = **8 Stunden**
- ✅ **Am ASH-Limit**

---

## Praktische Nutzung

### 🟢 Typische Nutzung (unbedenklich)

- **20-30 Anfragen pro Tag** mit je 10-30 Sekunden Audio
- **Gesamt:** ~10 Minuten Audio pro Tag
- **Auslastung:** <1% der täglichen Limits

### 🟡 Intensive Nutzung (völlig im Rahmen)

- **100-200 Anfragen pro Tag** mit je 20-40 Sekunden Audio
- **Gesamt:** 1-2 Stunden Audio pro Tag
- **Auslastung:** ~15-25% der täglichen Limits

### 🔴 Maximale Nutzung (Limit erreicht)

- **480+ Anfragen pro Tag** mit je 60+ Sekunden Audio
- **Gesamt:** 8 Stunden Audio pro Tag
- **Auslastung:** 100% des ASH-Limits

---

## Rate Limit Management

### Was passiert bei Überschreitung?

- HTTP Status Code: `429 Too Many Requests`
- Header: `retry-after` zeigt Wartezeit in Sekunden

### Im Code implementiert:

Die App fängt Rate-Limit-Fehler ab und zeigt eine Fehlermeldung.
Ein Error-Beep wird abgespielt wenn die API fehlschlägt.

### Empfehlungen:

1. ✅ **Normale Nutzung:** Kein Problem, Limits sind großzügig
2. ✅ **Mehrere Benutzer:** Bei Organisation-Account teilen sich alle Mitglieder die Limits
3. ⚠️ **Heavy Use:** Upgrade zu [Developer Plan](https://console.groq.com/settings/billing/plans) für höhere Limits
4. ⚠️ **Enterprise:** Kontakt mit Groq für Custom Limits

---

## Überwachung

Sie können Ihre aktuellen Limits und Nutzung einsehen:

- **Limits:** https://console.groq.com/settings/limits
- **Nutzung:** https://console.groq.com/settings/organization/usage

Die API gibt auch Header zurück mit aktuellen Limit-Informationen:

- `x-ratelimit-remaining-requests` - Verbleibende Anfragen (Tag)
- `x-ratelimit-remaining-tokens` - Nicht relevant für Whisper
- `x-ratelimit-reset-requests` - Zeit bis Reset

---

## Troubleshooting: ~42 Sekunden Latenz bei jedem Request

### Symptom
Jeder API-Call dauert konstant ~42 Sekunden — unabhängig von Dateigröße, API-Key oder Tarif. Auch LLM-Korrektur ist langsam (~16s). Das Rate-Limit-Dashboard zeigt keine Auffälligkeiten.

### Ursache: Defektes IPv6
`api.groq.com` hat sowohl IPv4- als auch IPv6-Adressen (Cloudflare). Windows bevorzugt IPv6. Wenn der Router IPv6 ankündigt, es aber nicht korrekt routet, wartet Windows bei jedem Request ~42 Sekunden auf den IPv6-Timeout, bevor er auf IPv4 zurückfällt.

### Diagnose
```powershell
# Ohne Flag (~42s) vs. mit IPv4 (<1s) → IPv6 ist schuld
Measure-Command { Invoke-WebRequest -Uri "https://api.groq.com" -UseBasicParsing }
Measure-Command { curl.exe --ipv4 https://api.groq.com }

# IPv6-Adressen prüfen
Resolve-DnsName api.groq.com
```

### Fix (PowerShell als Admin)
```powershell
Get-NetAdapter | Where-Object {$_.Status -eq "Up"} | Disable-NetAdapterBinding -ComponentID ms_tcpip6
```
Kein Neustart nötig. Betrifft alle HTTPS-Verbindungen zu Hosts mit IPv6-Records, nicht nur Groq.

---

## Fazit

**Für normale Speech-to-Text Nutzung sind die Limits mehr als ausreichend!**

Selbst bei sehr intensiver Nutzung (100+ Diktate pro Tag) bleiben Sie weit unter den Limits.
Das 8-Stunden-Audio-pro-Tag-Limit erlaubt praktisch unbegrenzte Nutzung für typische Anwendungsfälle.
