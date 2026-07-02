"""gradio_tabs/_styles.py — Zentrale CSS-Variablen für die PX-WebUI.

Hintergrund (Plan: branch ui-styling, 2026-07-02):
  Einheitliche CSS-Variablen für Farben, Spacing, Border-Radius.
  Komponenten in chat_tab.py / group_chat_tab.py ziehen ihre Werte
  aus dieser Datei — keine Hardcoded-Farben mehr.

In-Scope (siehe docs/PLAN_UI_STYLING.md):
  - CSS-Variablen-Block für gr.Blocks(css=...)
  - Pure-Python Konstanten für Python-Style-Verwendung
  - Helper-Funktion um beide zu exportieren

Out-of-Scope:
  - Kein Theme-Toggle Light/Dark
  - Kein Tailwind, kein npm
  - Kein motor-touch (KEINE Änderung an Server/patch.py/etc.)

Theme: dark-first, sci-mind-ästhetisch (deep-purple/cyan), accent-minimal.
"""
from __future__ import annotations


# ── CSS-Variablen-Block (für gr.Blocks(css=...)) ─────────────────────

# Hex-Farben als Python-Konstanten (zum Re-Use in Inline-Styles)
COLORS = {
    "bg_primary":     "#0e1117",  # Haupt-Hintergrund
    "bg_secondary":   "#161b22",  # Panel-Hintergrund
    "bg_tertiary":    "#21262d",  # Eingabe-Felder
    "border":         "#30363d",  # Standard-Border
    "border_hover":   "#58a6ff",  # Focus-Border (Akzent)
    "text_primary":   "#e6edf3",  # Haupttext
    "text_secondary": "#8b949e",  # Sekundärtext
    "text_muted":     "#6e7681",  # Disabled
    "accent":         "#a371f7",  # CitMind-Akzent (purple)
    "accent_alt":     "#39c5cf",  # Juexin-Akzent (cyan)
    "success":        "#3fb950",
    "warning":        "#d29922",
    "error":          "#f85149",
    "user_bubble":    "#1f6feb",  # User-Message-Hintergrund
    "assistant_bubble": "#21262d",  # Assistant-Message-Hintergrund
}

# Spacing-Skala (px)
SPACING = {
    "xs": 4,
    "sm": 8,
    "md": 12,
    "lg": 16,
    "xl": 24,
    "xxl": 32,
}

# Border-Radius-Skala (px)
RADIUS = {
    "sm": 4,
    "md": 8,
    "lg": 12,
    "xl": 16,
    "pill": 9999,  # für Status-Pills
}

# Font-Sizes (px)
FONTS = {
    "xs":  11,
    "sm":  13,
    "md":  14,
    "lg":  16,
    "xl":  20,
    "xxl": 28,
}


# ── CSS-String für gr.Blocks(css=...) ───────────────────────────────

def get_css() -> str:
    """Returnt den CSS-String für gr.Blocks(css=...).

    Setzt CSS-Variablen auf :root, die alle Komponenten erben.
    """
    return f"""
:root {{
  --bg-primary:     {COLORS['bg_primary']};
  --bg-secondary:   {COLORS['bg_secondary']};
  --bg-tertiary:    {COLORS['bg_tertiary']};
  --border:         {COLORS['border']};
  --border-hover:   {COLORS['border_hover']};
  --text-primary:   {COLORS['text_primary']};
  --text-secondary: {COLORS['text_secondary']};
  --text-muted:     {COLORS['text_muted']};
  --accent:         {COLORS['accent']};
  --accent-alt:     {COLORS['accent_alt']};
  --success:        {COLORS['success']};
  --warning:        {COLORS['warning']};
  --error:          {COLORS['error']};
  --user-bubble:    {COLORS['user_bubble']};
  --assistant-bubble: {COLORS['assistant_bubble']};

  --space-xs: {SPACING['xs']}px;
  --space-sm: {SPACING['sm']}px;
  --space-md: {SPACING['md']}px;
  --space-lg: {SPACING['lg']}px;
  --space-xl: {SPACING['xl']}px;
  --space-xxl: {SPACING['xxl']}px;

  --radius-sm: {RADIUS['sm']}px;
  --radius-md: {RADIUS['md']}px;
  --radius-lg: {RADIUS['lg']}px;
  --radius-xl: {RADIUS['xl']}px;
  --radius-pill: {RADIUS['pill']}px;

  --font-xs: {FONTS['xs']}px;
  --font-sm: {FONTS['sm']}px;
  --font-md: {FONTS['md']}px;
  --font-lg: {FONTS['lg']}px;
  --font-xl: {FONTS['xl']}px;
  --font-xxl: {FONTS['xxl']}px;
}}

/* ── Globale Akzente ────────────────────────────────────── */
.gradio-container {{
  background: var(--bg-primary);
  color: var(--text-primary);
}}

/* Buttons: einheitlicher Radius + Hover */
button {{ border-radius: var(--radius-md) !important; }}
button.primary {{ background: var(--accent) !important; }}

/* Inputs: Focus-Border-Akzent */
input:focus, textarea:focus, select:focus {{
  border-color: var(--border-hover) !important;
}}

/* Chat-Bubbles: User rechts, Assistant links */
.message.user {{ background: var(--user-bubble) !important; }}
.message.assistant {{ background: var(--assistant-bubble) !important; }}
"""


# ── Helper für Inline-Style-Composition (Python-Side) ──────────────

def user_bubble_style() -> str:
    """Inline-Style-String für User-Bubbles (Python-Konsumenten)."""
    return f"background:{COLORS['user_bubble']};border-radius:{RADIUS['md']}px;padding:{SPACING['sm']}px;"


def assistant_bubble_style() -> str:
    """Inline-Style-String für Assistant-Bubbles."""
    return f"background:{COLORS['assistant_bubble']};border-radius:{RADIUS['md']}px;padding:{SPACING['sm']}px;"


def pill_style(label: str, color: str = "accent") -> str:
    """Status-Pill (Engine: piper ✓, etc.)."""
    bg = COLORS.get(color, COLORS['accent'])
    return (
        f"background:{bg};color:{COLORS['text_primary']};"
        f"border-radius:{RADIUS['pill']}px;padding:{SPACING['xs']}px {SPACING['md']}px;"
        f"font-size:{FONTS['sm']}px;font-weight:600;"
    )


# ── Smoke-Test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    css = get_css()
    assert "--bg-primary" in css, "CSS-Block muss Variablen definieren"
    assert "--accent" in css
    assert "border-radius" in css
    assert len(COLORS) >= 13, f"COLORS muss ≥13 keys haben, hat {len(COLORS)}"
    assert len(SPACING) == 6
    assert len(RADIUS) == 5
    assert len(FONTS) == 6
    print(f"[_styles] smoke OK: {len(COLORS)} colors, {len(SPACING)} spacing, {len(RADIUS)} radii")
    print(f"[_styles] CSS-Block: {len(css)} chars")
