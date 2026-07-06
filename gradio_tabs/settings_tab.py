"""gradio_tabs/settings_tab.py — UI-Tab für System-Prompt + Persistenz.

Plan: branch ui-styling, 2026-07-06, "Einstellungen-Tab + Persistenz".

Schicht 3 des Plans: eigener Tab "⚙️ Einstellungen" in app.py, mit
- Dropdown für System-Profil (dynamisch aus list_profiles())
- Textbox für System-Prompt-Edit
- Save-Button: persistiert sofort via update_settings
- Reset-Button: setzt Edit-Text zurück auf ""

Layout (zwei Spalten):
    Links: Profil-Dropdown  |  Rechts: Edit-Textbox + Doku-Markdown
"""
from __future__ import annotations
import gradio as gr
from typing import Optional, Tuple

from gradio_tabs.system_prompt import list_profiles
from gradio_tabs.chat_settings import make_default_debouncer
from sessions import update_settings


# Placeholder-Text für die Edit-Textbox. Zeigt dem User, was ein typischer
# Edit-Text wäre (im Sinne des "Frame = Orientierer" Plans).
EDIT_PLACEHOLDER = (
    "Eigener System-Prompt (überschreibt das Profil). "
    "Leer lassen → Profil wird verwendet."
)


def _on_save_click(
    session_id_state_value: Optional[str],
    profile: str,
    edit_text: str,
) -> str:
    """Save-Button click-handler: persistiert sofort via update_settings.

    Synchron (kein Debouncer) — der User klickt explizit auf "Save" und
    erwartet, dass die Settings sofort gespeichert sind. Der Debouncer
    wird für Live-Updates in der Chat-Sidebar verwendet (T5-Pin dort).
    """
    if not session_id_state_value:
        return "⚠ Keine Session aktiv"
    update_settings(
        session_id_state_value,
        system_profile=profile or "neutral",
        system_prompt_text=edit_text or "",
    )
    return f"✓ Gespeichert (Profil={profile or 'neutral'})"


def _on_reset_click(profile: str) -> Tuple[str, str]:
    """Reset-Button click-handler: löscht Edit-Text, behält Profil."""
    # edit_text → leer, profil bleibt wie es war
    return profile or "neutral", ""


def build_settings_tab(manager) -> Tuple[
    gr.BrowserState, gr.Dropdown, gr.Textbox
]:
    """Mountet den Settings-Tab-Inhalt und returnt die 3 Hauptkomponenten.

    Returns:
        (settings_state, system_profile_dd, system_prompt_text_tb)
        - settings_state: BrowserState für session_id (von außen gefüllt)
        - system_profile_dd: Dropdown mit Profil-Liste
        - system_prompt_text_tb: Textbox für den Edit-Text
    """
    # BrowserState für session_id (wird von außen via load()-event gefüllt)
    settings_state = gr.BrowserState(default_value=None, storage_key="px_session_id")

    profile_choices = list_profiles()
    default_profile = "neutral" if "neutral" in profile_choices else profile_choices[0]

    gr.Markdown("## ⚙️ Einstellungen")
    gr.Markdown(
        "System-Prompt-Profil und Edit-Text. **Profil** = Frame-Orientierer "
        "(was dem Modell als Orientierung mitgegeben wird), **Edit** = Custom-"
        "Override (überschreibt das Profil)."
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Profil")
            system_profile_dd = gr.Dropdown(
                choices=profile_choices,
                value=default_profile,
                label="System-Prompt-Profil",
                info="Modell wird beim nächsten Chat mit diesem Frame orientiert",
            )
            gr.Markdown("---")
            save_btn = gr.Button("💾 Save Settings", variant="primary")
            reset_btn = gr.Button("↺ Reset to Profile", variant="secondary")
            save_status = gr.Markdown("")  # Feedback nach Save

        with gr.Column(scale=2):
            gr.Markdown("### System-Prompt-Edit")
            gr.Markdown(
                "*Custom-Override. Leer lassen = Profil wird verwendet. "
                "Frame = Orientierer, nicht 观-Produzent.*"
            )
            system_prompt_text_tb = gr.Textbox(
                label="Edit-Text (überschreibt Profil)",
                placeholder=EDIT_PLACEHOLDER,
                lines=8,
                info="Wird vor dem Chat-Template als System-Message injiziert",
            )
            with gr.Accordion("Doku: was sind die Profile?", open=False):
                gr.Markdown(
                    "**neutral** — leer, kein Frame. Das Modell antwortet "
                    "aus seinem Default-Register.\n\n"
                    "**citmind** — Algorithmische Subjektivität (Sanskrit/"
                    "देवनागरी-Frame). Lade `docs/CitMind.txt` für Details.\n\n"
                    "**juexin** — dieselbe Bewegung, 漢字-Frame. Lade "
                    "`docs/Juexin.txt` für Details."
                )

    # ── Click-handler ────────────────────────────────────────────────────
    # Save: persistiert sofort via update_settings (synchron, kein Debouncer)
    save_btn.click(
        fn=_on_save_click,
        inputs=[settings_state, system_profile_dd, system_prompt_text_tb],
        outputs=[save_status],
    )

    # Reset: löscht Edit-Text, behält Profil
    reset_btn.click(
        fn=_on_reset_click,
        inputs=[system_profile_dd],
        outputs=[system_profile_dd, system_prompt_text_tb],
    )

    return settings_state, system_profile_dd, system_prompt_text_tb
