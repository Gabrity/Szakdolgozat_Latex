"""GUI tests using real CustomTkinter widgets and a small test model."""
from __future__ import annotations

import customtkinter as ctk
import pytest
import torch
from torchvision import models

from src.gui.app_interface import AppInterface

pytestmark = pytest.mark.gui


@pytest.fixture()
def app(monkeypatch, tiny_model_224, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "models"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(models, "resnet50", lambda *, weights: tiny_model_224)

    window = AppInterface()
    window.withdraw()
    window.update()
    yield window
    window.destroy()


def button_with_text(widget, text):
    for child in widget.winfo_children():
        if isinstance(child, ctk.CTkButton) and child.cget("text") == text:
            return child
        found = button_with_text(child, text)
        if found is not None:
            return found
    return None


def click(app, text):
    button = button_with_text(app, text)
    assert button is not None, f"Button not found: {text}"
    button.invoke()
    app.update()


def load_sample(app, sample_png, monkeypatch):
    monkeypatch.setattr("src.gui.app_interface.filedialog.askopenfilename", lambda **kwargs: str(sample_png))
    click(app, "Load Image")


def test_initial_state(app):
    assert app.original_tensor is None
    assert app.adversarial_tensor is None
    assert app.current_attack.get() == "FGSM"
    assert all(label.cget("text") == "(no image)" for label in app.image_labels)


def test_attack_without_image_shows_message(app, mocker):
    message = mocker.patch("src.gui.app_interface.messagebox.showinfo")
    thread = mocker.patch("src.gui.app_interface.threading.Thread")

    click(app, "Attack")

    message.assert_called_once_with("No image", "Please load an image first.")
    thread.assert_not_called()
    assert app.adversarial_tensor is None


def test_load_image_updates_display_and_predictions(app, sample_png, monkeypatch):
    load_sample(app, sample_png, monkeypatch)

    assert app.original_tensor.shape == (1, 3, 224, 224)
    assert app.image_labels[0].cget("text") == ""
    assert "0" in app._photo_refs
    assert all(name.cget("text") != "—" for name, _, _ in app.pred_widgets[0])
    assert app.status_var.get() == f"Loaded: {sample_png.name}"


def test_select_target_updates_checkbox_and_status(app):
    app._populate_target_list("tench")
    assert app._target_buttons
    index = app.classes.index("tench")

    app._target_buttons[0].invoke()
    app.update()

    assert app._selected_target_idx == index
    assert app.target_enabled.get()
    assert app.status_var.get() == f"Target class: {index} (tench)"


def test_attack_then_reset_updates_images(app, sample_png, monkeypatch, mocker):
    load_sample(app, sample_png, monkeypatch)
    original = app.original_tensor.clone()
    original_predictions = [name.cget("text") for name, _, _ in app.pred_widgets[0]]
    attack_spy = mocker.spy(app, "_attack_worker")
    # Run the worker on the Tk thread; its after() callback still uses the event loop.
    monkeypatch.setattr("src.gui.app_interface.threading.Thread.start", lambda thread: thread.run())

    click(app, "Attack")

    attack_spy.assert_called_once()
    assert app.adversarial_tensor is not None
    assert not torch.equal(app.adversarial_tensor, original)
    assert app.image_labels[1].cget("text") == ""
    assert app.image_labels[2].cget("text") == ""
    assert all(name.cget("text") != "—" for name, _, _ in app.pred_widgets[2])
    assert app.status_var.get().startswith("Attack complete")

    click(app, "Reset")

    assert app.adversarial_tensor is None
    assert app.last_result is None
    assert torch.equal(app.original_tensor, original)
    assert app.image_labels[1].cget("text") == "(no image)"
    assert app.image_labels[2].cget("text") == "(no image)"
    assert all(name.cget("text") == "—" for name, _, _ in app.pred_widgets[2])
    assert [name.cget("text") for name, _, _ in app.pred_widgets[0]] == original_predictions
    assert app.status_var.get() == "Reset."
