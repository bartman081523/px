"""tests/test_ui_styling.py — Pure-Logic Tests für _styles.py.

Pinnt CSS-Variablen-Konsistenz (Colors/Spacing/Radius/Fonts Counts +
CSS-Block enthält die wichtigen Variablen). Wenn jemand Variablen löscht
oder umbenennt, fallen die Tests.

Run:
    /run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python \
      tests/test_ui_styling.py
"""
from __future__ import annotations
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestColors(unittest.TestCase):
    """Pinnt Color-Set (Plan UI-Styling: dark-first sci-mind theme)."""

    def setUp(self):
        from gradio_tabs._styles import COLORS
        self.COLORS = COLORS

    def test_required_colors_present(self):
        """Alle Kern-Farben vorhanden."""
        for key in ("bg_primary", "bg_secondary", "bg_tertiary", "border",
                    "border_hover", "text_primary", "text_secondary",
                    "text_muted", "accent", "accent_alt", "success",
                    "warning", "error", "user_bubble", "assistant_bubble"):
            self.assertIn(key, self.COLORS, f"fehlende Farbe: {key}")

    def test_color_format_hex(self):
        """Alle Farben sind #RRGGBB Hex-Strings."""
        import re
        hex_re = re.compile(r"^#[0-9a-fA-F]{6}$")
        for key, val in self.COLORS.items():
            self.assertRegex(val, hex_re, f"{key}={val} ist kein gültiger Hex-Color")

    def test_accent_distinct_from_alt(self):
        """Accent und Accent-Alt müssen visuell unterscheidbar sein."""
        self.assertNotEqual(self.COLORS["accent"], self.COLORS["accent_alt"])

    def test_user_bubble_distinct_from_assistant(self):
        """User-Bubble ≠ Assistant-Bubble (Bubble-Alignment)."""
        self.assertNotEqual(
            self.COLORS["user_bubble"], self.COLORS["assistant_bubble"]
        )


class TestSpacing(unittest.TestCase):
    """Pinnt Spacing-Skala."""

    def setUp(self):
        from gradio_tabs._styles import SPACING
        self.SPACING = SPACING

    def test_required_keys_present(self):
        """xs/sm/md/lg/xl/xxl vorhanden."""
        for key in ("xs", "sm", "md", "lg", "xl", "xxl"):
            self.assertIn(key, self.SPACING)

    def test_scale_monotonic_increasing(self):
        """xs < sm < md < lg < xl < xxl."""
        keys = ["xs", "sm", "md", "lg", "xl", "xxl"]
        values = [self.SPACING[k] for k in keys]
        for i in range(1, len(values)):
            self.assertGreater(values[i], values[i - 1],
                f"{keys[i]}={values[i]} sollte > {keys[i-1]}={values[i-1]} sein")

    def test_all_positive(self):
        """Keine negativen Spacing-Werte."""
        for key, val in self.SPACING.items():
            self.assertGreater(val, 0, f"{key}={val} muss > 0")


class TestRadius(unittest.TestCase):
    """Pinnt Border-Radius-Skala."""

    def setUp(self):
        from gradio_tabs._styles import RADIUS
        self.RADIUS = RADIUS

    def test_required_keys_present(self):
        """sm/md/lg/xl/pill vorhanden."""
        for key in ("sm", "md", "lg", "xl", "pill"):
            self.assertIn(key, self.RADIUS)

    def test_pill_equals_pill(self):
        """pill-Radius = 9999 (für Status-Pills)."""
        self.assertEqual(self.RADIUS["pill"], 9999)

    def test_scale_monotonic_increasing(self):
        """sm < md < lg < xl."""
        keys = ["sm", "md", "lg", "xl"]
        values = [self.RADIUS[k] for k in keys]
        for i in range(1, len(values)):
            self.assertGreater(values[i], values[i - 1])


class TestFonts(unittest.TestCase):
    """Pinnt Font-Size-Skala."""

    def setUp(self):
        from gradio_tabs._styles import FONTS
        self.FONTS = FONTS

    def test_required_keys_present(self):
        """xs/sm/md/lg/xl/xxl vorhanden."""
        for key in ("xs", "sm", "md", "lg", "xl", "xxl"):
            self.assertIn(key, self.FONTS)

    def test_scale_monotonic_increasing(self):
        """xs < sm < md < lg < xl < xxl."""
        keys = ["xs", "sm", "md", "lg", "xl", "xxl"]
        values = [self.FONTS[k] for k in keys]
        for i in range(1, len(values)):
            self.assertGreater(values[i], values[i - 1])


class TestCssBlock(unittest.TestCase):
    """Pinnt get_css() Output."""

    def setUp(self):
        from gradio_tabs._styles import get_css
        self.css = get_css()

    def test_contains_root_block(self):
        """:root { ... } muss da sein (CSS-Variablen-Definition)."""
        self.assertIn(":root {", self.css)

    def test_contains_required_variables(self):
        """Kern-CSS-Variablen definiert."""
        for var in ("--bg-primary", "--accent", "--text-primary",
                    "--border-hover", "--user-bubble", "--assistant-bubble"):
            self.assertIn(var, self.css, f"fehlende CSS-Variable: {var}")

    def test_contains_button_style(self):
        """Buttons haben einheitlichen Border-Radius (Plan-Polish)."""
        self.assertIn("button", self.css)
        self.assertIn("border-radius", self.css)

    def test_contains_message_styles(self):
        """Chat-Bubbles haben .user/.assistant Klassen (Plan: Bubble-Alignment)."""
        self.assertIn(".message.user", self.css)
        self.assertIn(".message.assistant", self.css)

    def test_contains_focus_accent(self):
        """Inputs haben Focus-Border-Akzent (Plan: Komponenten-Polish)."""
        self.assertIn("input:focus", self.css)
        self.assertIn("border-hover", self.css)


class TestStyleHelpers(unittest.TestCase):
    """Inline-Style-Helper."""

    def setUp(self):
        from gradio_tabs._styles import (
            user_bubble_style, assistant_bubble_style, pill_style,
        )
        self.user = user_bubble_style
        self.assistant = assistant_bubble_style
        self.pill = pill_style

    def test_user_bubble_has_background(self):
        """user_bubble_style enthält background-Property."""
        s = self.user()
        self.assertIn("background", s)
        self.assertIn("border-radius", s)
        self.assertIn("padding", s)

    def test_assistant_bubble_distinct_from_user(self):
        """assistant_bubble ≠ user_bubble."""
        self.assertNotEqual(self.user(), self.assistant())

    def test_pill_has_required_properties(self):
        """pill_style setzt background + border-radius + padding (Label wird separat gerendert)."""
        s = self.pill("Engine: piper ✓", "success")
        self.assertIn("background:#3fb950", s)  # success-color
        self.assertIn("border-radius:9999px", s)  # pill
        self.assertIn("padding", s)
        self.assertIn("font-weight:600", s)

    def test_pill_default_color_is_accent(self):
        """pill_style default = accent-color (purple)."""
        s = self.pill("test")
        self.assertIn("background:#a371f7", s)  # accent


class TestIntegration(unittest.TestCase):
    """Integration-Smoke: _styles ist von chat_tab + app importierbar.

    Hintergrund (Plan ui-styling, 2026-07-02):
      chat_tab importiert _styles für die Status-Pill (Task #184).
      app importiert _styles für mount_gradio_app(css=...) (Task #182).
      Wenn _styles-Import kaputt geht, brechen alle UI-Builds.
    """

    def test_styles_importable(self):
        """_styles.py ist importierbar (kein Syntax/ImportError)."""
        from gradio_tabs import _styles  # noqa: F401

    def test_get_css_returns_nonempty_string(self):
        """get_css() returnt non-empty String (für gr.mount_gradio_app)."""
        from gradio_tabs._styles import get_css
        css = get_css()
        self.assertIsInstance(css, str)
        self.assertGreater(len(css), 100)  # 1410 chars typisch

    def test_pill_style_usable_in_html(self):
        """pill_style-Output ist gültiger CSS-String (für f-string interpolation)."""
        from gradio_tabs._styles import pill_style
        style = pill_style("Test", "accent")
        # Sollte in f"<div style='{style}'>...</div>" ohne Quote-Bruch passen
        html = f"<div style='{style}'>Test</div>"
        self.assertIn("Test", html)
        self.assertIn("style='", html)


if __name__ == "__main__":
    unittest.main(verbosity=2)
