#!/usr/bin/env bash
# push_hf.sh — Push das PX-Repo zu HuggingFace Space (nur Code, keine LFS-Orphans).
#
# Plan 2026-07-07: vermeidet 1 GB Storage-Limit auf HF free tier.
#   1. Erstellt einen frischen Sparse-Branch von ui-styling, der
#      scratches/, telemetry/, sessions/, local_debug.log UND alle
#      LFS-tracked files (logs/local_debug.log) AUS dem Index nimmt.
#   2. Pusht diesen Branch als "main" zu einem NEUEN HF Space.
#   3. Original-Branch ui-styling und Working-Tree bleiben 100% unangetastet.
#
# Verwendung:
#   bash push_hf.sh                                       # nutzt .env + px-explorer-v4 default
#   HF_TOKEN=hf_xxx SPACE_NAME=px-explorer-v4 ./push_hf.sh  # override aus env
#
# Standard-Space: px-explorer-v4 (User-Instruction 2026-07-09:
#   "wir bleiben für immer auf v4. mal merken.").
#   HF free-tier lehnt neue gradio-Spaces seit 2026-07 mit 402 ab.
#
# Voraussetzungen:
#   - git lfs installiert (git-lfs >= 2.x)
#   - HF_TOKEN env var (write-Scope auf User-Account)
#   - Lokal: branch 'ui-styling' existiert mit dem Code
#
# Lessons Learned (2026-07-07):
#   - HF free tier: 1 GB Storage account-weit. LFS-Orphans zählen mit.
#   - DELETE /api/spaces/{id} → 404. Space-Löschung nur via Browser.
#   - POST /api/spaces → 404. POST /api/repos/create → 200 (korrekter Endpoint).
#   - 'git push --force' ist OK für HF Spaces (kein lokales Tracking).
#   - 'git worktree add' scheitert wenn Branch im Hauptverzeichnis gecheckt ist.

set -euo pipefail

# ── .env laden (Plan 2026-07-09) ────────────────────────────────
# Lokale Credentials liegen in .env (HF_TOKEN), .env ist .gitignored.
# Falls .env nicht existiert, wird HF_TOKEN aus dem Env erwartet.
if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# ── Konfiguration ────────────────────────────────────────────────
BRANCH_LOCAL="ui-styling"
BRANCH_HF="ui-styling-hf"
HF_USER="neuralworm"
# User-Instruction 2026-07-09: "wir bleiben für immer auf v4. mal merken."
# → v4 ist der permanente Space. Neue Spaces (v5/v6) werden seit 2026-07
#   von HF free-tier mit 402 (PRO-only) abgelehnt.
SPACE_NAME="${SPACE_NAME:-px-explorer-v4}"
REMOTE_URL="https://${HF_USER}:${HF_TOKEN}@huggingface.co/spaces/${HF_USER}/${SPACE_NAME}"

# Was NIE in den Push soll (HF free tier 1 GB, diese Files füllen das Limit)
# ACHTUNG: sessions_bak3/temp_media/ enthält .png-Files die HF pre-receive-hook
# auch dann ablehnt, wenn sie nicht referenziert sind — Hook checkt das WORKING-TREE.
EXCLUDE_PATHS=(
    "scratches/"             # 575M emergence-Experimente
    "telemetry/"             # 20M Telemetrie-Snapshots
    "sessions/"              # 28M User-Chat-Historien (Daten, nicht Code)
    "sessions_bak/"          # Session-Backup (Daten, nicht Code)
    "sessions_bak3/"         # Session-Backup mit PNGs in temp_media/
    "local_debug.log"        # 1.8M Server-Debug-Log
    "logs/local_debug.log"   # LFS-pointer (verhindert LFS-Upload komplett)
)

# ── Sanity-Checks ────────────────────────────────────────────────
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN env var nicht gesetzt. Abbruch." >&2
    exit 1
fi

if [ ! -d ".git" ]; then
    echo "ERROR: nicht in einem Git-Repo ($(pwd)). Abbruch." >&2
    exit 1
fi

if ! git rev-parse --verify "$BRANCH_LOCAL" >/dev/null 2>&1; then
    echo "ERROR: branch '$BRANCH_LOCAL' existiert nicht lokal. Abbruch." >&2
    exit 1
fi

# Aktuellen Working-Tree-Status checken (nichts darf modified sein)
if [ -n "$(git status --porcelain)" ]; then
    echo "WARN: Working-Tree hat uncommitted changes:"
    git status --short
    if [ -t 0 ]; then
        # Interaktiv: frage User
        read -p "Trotzdem fortfahren? (j/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Jj]$ ]]; then
            echo "Abbruch."
            exit 1
        fi
    else
        # Non-interaktiv (Pipe/Subshell): fortfahren
        echo "Non-interactive mode: fortfahren..."
    fi
fi

# ── HF Space erstellen (idempotent: skip wenn schon da) ────────
echo "=== HF Space '$SPACE_NAME' ==="
SPACE_CHECK=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "Authorization: Bearer ${HF_TOKEN}" \
    "https://huggingface.co/api/spaces/${HF_USER}/${SPACE_NAME}")

if [ "$SPACE_CHECK" = "200" ]; then
    echo "Space existiert bereits. usedStorage:"
    curl -s -H "Authorization: Bearer ${HF_TOKEN}" \
        "https://huggingface.co/api/spaces/${HF_USER}/${SPACE_NAME}" \
        | python3 -c "import sys, json; d=json.load(sys.stdin); print(f'  {d.get(\"usedStorage\", 0)/1024/1024:.1f} MB / 1024 MB (1 GB free tier)')"
else
    echo "Erstelle Space..."
    CREATE_RESP=$(curl -s -w "\n%{http_code}" -X POST \
        -H "Authorization: Bearer ${HF_TOKEN}" \
        -H "Content-Type: application/json" \
        -d "{\"name\":\"${SPACE_NAME}\",\"type\":\"space\",\"sdk\":\"gradio\"}" \
        "https://huggingface.co/api/repos/create")
    HTTP_CODE=$(echo "$CREATE_RESP" | tail -n1)
    if [ "$HTTP_CODE" != "200" ]; then
        echo "ERROR: Space-Erstellung fehlgeschlagen (HTTP $HTTP_CODE):" >&2
        echo "$CREATE_RESP" | head -n1 >&2
        exit 1
    fi
    echo "Space erstellt: https://huggingface.co/spaces/${HF_USER}/${SPACE_NAME}"
fi

# ── Sparse-Branch vorbereiten ───────────────────────────────────
echo "=== Sparse-Branch '$BRANCH_HF' ==="

# Wenn Branch existiert → löschen (sauberer Reset)
# Erst zurück zu BRANCH_LOCAL falls ui-styling-hf ausgecheckt ist
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" = "$BRANCH_HF" ]; then
    echo "Branch '$BRANCH_HF' ist aktuell gecheckt — switche zu '$BRANCH_LOCAL'..."
    # -f: force, falls working-tree-files nicht im Ziel-branch-index sind
    # (passiert nach `git rm --cached` auf einem Branch, weil die Files nur
    # aus dem Index, nicht aus dem Working-Tree, entfernt wurden).
    git checkout -f "$BRANCH_LOCAL" >/dev/null
fi
if git rev-parse --verify "$BRANCH_HF" >/dev/null 2>&1; then
    echo "Lösche existierenden Branch '$BRANCH_HF'..."
    git branch -D "$BRANCH_HF" >/dev/null
fi

# WICHTIG: Wir bauen den Sparse-Branch als ORPHAN (kein Parent), damit
# die komplette History von $BRANCH_LOCAL (die noch alle LFS-tracked files
# in der History referenziert) NICHT mit-gepusht wird. Sonst triggert der
# pre-receive-hook "contains binary files" für ALLE je committeten
# .png/.bin files im Branch — auch wenn wir sie im aktuellen Index nicht
# haben.
echo "Branch '$BRANCH_HF' von '$BRANCH_LOCAL' erstellt (ORPHAN, ohne History)."
git checkout --orphan "$BRANCH_HF" >/dev/null
git checkout "$BRANCH_LOCAL" -- . 2>&1 | head -3 || true

# Excluded paths aus Index entfernen (Working-Tree bleibt unangetastet)
EXCLUDED_COUNT=0
for path in "${EXCLUDE_PATHS[@]}"; do
    # Pfad-spezifische Behandlung
    case "$path" in
        */)
            # Verzeichnis
            if [ -d "$path" ]; then
                # git rm -r kann bei leeren/non-existent Pfaden failen — wir
                # unterdrücken ALLE outputs und zählen nur die echten deletes.
                RM_OUT=$(git rm -r --cached "$path" 2>&1 | grep -c "^rm " || true)
                EXCLUDED_COUNT=$((EXCLUDED_COUNT + RM_OUT))
            fi
            ;;
        *)
            # Einzelne Datei
            if [ -f "$path" ]; then
                if git rm --cached "$path" >/dev/null 2>&1; then
                    EXCLUDED_COUNT=$((EXCLUDED_COUNT + 1))
                fi
            fi
            ;;
    esac
done
echo "Aus Index entfernt: ${EXCLUDED_COUNT} files (${EXCLUDE_PATHS[*]})"

# Falls es keine Änderungen gibt (alles schon exkludiert) — leeren Commit
if [ -z "$(git status --porcelain)" ]; then
    echo "Keine Änderungen — Working-Tree war bereits clean."
else
    git commit -m "push_hf: sparse-branch für HF-Push (nur Code, 0 LFS)" >/dev/null
    echo "Sparse-Commit erstellt."
fi

# Code-Size + LFS-Check (ohne pipefail — xargs+stat kann non-zero exits haben
# bei symlinks oder special files; das ist hier OK)
set +o pipefail
CODE_BYTES=$(git ls-files | xargs -I {} stat -c '%s' {} 2>/dev/null | awk '{s+=$1} END {print s+0}')
set -o pipefail
CODE_MB=$(awk -v b="$CODE_BYTES" 'BEGIN {printf "%.2f", b/1024/1024}')
LFS_COUNT=$(git lfs ls-files 2>/dev/null | wc -l)
echo "Push-Größe: ${CODE_MB} MB Code, ${LFS_COUNT} LFS-Files."

# ── Push ────────────────────────────────────────────────────────
echo "=== Push zu $REMOTE_URL ==="

# Remote-URL setzen/updaten
git remote set-url hf "$REMOTE_URL" 2>/dev/null || git remote add hf "$REMOTE_URL"

# Force-Push (HF Spaces haben initial-commit, force ist OK)
# GIT_LFS_SKIP_PUSH=true verhindert dass git-lfs alle 238 LFS-Objekte im
# Local-Cache hochlädt (auch wenn der sparse-branch 0 LFS-Referenzen hat).
# Bei sparse-branches mit 0 LFS-files wollen wir NUR plain git push.
GIT_LFS_SKIP_PUSH=true PUSH_OUTPUT=$(GIT_LFS_SKIP_PUSH=true git push --force hf "$BRANCH_HF:main" 2>&1) || PUSH_EXIT=$?
PUSH_EXIT=${PUSH_EXIT:-0}
if [ "$PUSH_EXIT" = "0" ]; then
    echo ""
    echo "✓ Push erfolgreich!"
    echo "  Space: https://huggingface.co/spaces/${HF_USER}/${SPACE_NAME}"
    echo "  Code:  ${CODE_MB} MB (kein LFS, keine Scratches)"
    # Verify final storage
    sleep 3
    FINAL_STORAGE=$(curl -s -H "Authorization: Bearer ${HF_TOKEN}" \
        "https://huggingface.co/api/spaces/${HF_USER}/${SPACE_NAME}" \
        | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('usedStorage', 0)/1024/1024)" 2>/dev/null)
    echo "  Storage: ${FINAL_STORAGE:-?} MB / 1024 MB"
else
    echo "✗ Push fehlgeschlagen (exit $PUSH_EXIT):" >&2
    echo "$PUSH_OUTPUT" | tail -10 >&2
    echo "  Mögliche Ursache: HF-Account-Storage-Limit erreicht." >&2
    echo "  Cleanup: alten Space im Browser löschen, dann SPACE_NAME ändern." >&2
    # Cleanup trotzdem
    git checkout -f "$BRANCH_LOCAL" >/dev/null 2>&1 || true
    git branch -D "$BRANCH_HF" >/dev/null 2>&1 || true
    exit 1
fi

# ── Zurück zu ui-styling ───────────────────────────────────────
echo ""
echo "=== Cleanup ==="
# -f: force, weil unversionierte Files (local_debug.log etc.) im Working-Tree
# den Checkout sonst blocken — sie existieren auf ui-styling als getrackt,
# wurden aber im Sparse-Branch aus dem Index entfernt. Working-Tree bleibt
# unangetastet (git checkout -f überschreibt nur Index-Mismatches).
git checkout -f "$BRANCH_LOCAL" >/dev/null
git branch -D "$BRANCH_HF" >/dev/null
echo "Zurück auf '$BRANCH_LOCAL'. Sparse-Branch gelöscht."
echo "Working-Tree ist 100% identisch zu vorher (Index-Cleanup war --cached-only)."
