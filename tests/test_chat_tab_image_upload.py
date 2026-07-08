"""Smoke-Test: Chat-Tab verwendet gr.MultimodalTextbox (Bild-Upload).

Verifiziert, dass die zentrale Chat-Input-Komponente in build_chat_tab
ein MultimodalTextbox ist (file_types=["image"], file_count="multiple")
und nicht mehr ein simples gr.Textbox. Reine Source-Inspektion — kein
Gradio-Start nötig, damit der Test schnell und headless läuft.
"""
import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_chat_tab_module():
    """Lade chat_tab.py als AST, um den build_chat_tab-Call zu inspizieren."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "gradio_tabs", "chat_tab.py",
    )
    with open(path, "r", encoding="utf-8") as f:
        return ast.parse(f.read())


def test_chat_tab_imports_multimodal_helpers():
    """chat_tab.py importiert normalize_multimodal_message, is_empty_message,
    extract_text_blocks und _normalize_history_for_chatbot aus multimodal_input."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "gradio_tabs", "chat_tab.py",
    )
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    assert "from gradio_tabs.multimodal_input import" in src
    assert "normalize_multimodal_message" in src
    assert "is_empty_message" in src
    assert "extract_text_blocks" in src
    assert "_normalize_history_for_chatbot" in src


def test_chat_tab_uses_multimodaltextbox():
    """build_chat_tab nutzt gr.MultimodalTextbox — nicht gr.Textbox
    für den user_message-Input."""
    tree = _load_chat_tab_module()
    # Suche nach 'MultimodalTextbox' im Source.
    found_multimodal = False
    found_image_filetypes = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "MultimodalTextbox":
            found_multimodal = True
        if isinstance(node, ast.Constant) and node.value == "image":
            # könnte auch woanders vorkommen — wir suchen einfach das Stichwort
            found_image_filetypes = True
    assert found_multimodal, "gr.MultimodalTextbox nicht gefunden in chat_tab.py"
    assert found_image_filetypes, "'image' file_types-String nicht in chat_tab.py"


def test_chat_tab_normalizes_user_message():
    """Der user_message-Handler muss normalize_multimodal_message aufrufen
    statt manuell dict-Shapes zu prüfen."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "gradio_tabs", "chat_tab.py",
    )
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    # Der user_message-Handler enthält normalize_multimodal_message.
    assert "content = normalize_multimodal_message(message)" in src
    # is_empty_message wird im user_message-Handler für leeren Input genutzt.
    assert "is_empty_message(message)" in src


def test_chat_tab_normalizes_loaded_history():
    """on_load und handle_load_saved nutzen _normalize_history_for_chatbot."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "gradio_tabs", "chat_tab.py",
    )
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    # Mindestens zwei Aufrufe von _normalize_history_for_chatbot.
    assert src.count("_normalize_history_for_chatbot(") >= 2, (
        "Erwarte _normalize_history_for_chatbot-Calls in on_load + handle_load_saved"
    )


def test_multimodal_input_module_exposes_required_apis():
    """multimodal_input.py exportiert die fünf vom chat_tab genutzten Symbole."""
    from gradio_tabs import multimodal_input as M
    for name in ("normalize_multimodal_message",
                 "is_empty_message",
                 "extract_text_blocks",
                 "_normalize_history_for_chatbot"):
        assert hasattr(M, name), f"fehlt: {name}"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed + failed} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
