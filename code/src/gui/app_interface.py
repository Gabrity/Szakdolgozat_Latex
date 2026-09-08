"""CustomTkinter GUI for the Adversarial Image Generator."""
from __future__ import annotations

import os
import threading
import traceback
from typing import List, Optional

import customtkinter as ctk
import torch
from PIL import Image
from tkinter import filedialog, messagebox

from src import attack_engine
from src.attack_engine import AttackContext, AttackResult
from src.defense_module import DefenseModule
from src.imagenet_classes import get_imagenet_classes
from src.model_manager import ModelManager
from src.utils import load_image_as_tensor, perturbation_to_pil, tensor_to_pil

# Optional drag-and-drop support.
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD  # type: ignore

    _DND_AVAILABLE = True
except Exception:  # pragma: no cover - optional dep
    _DND_AVAILABLE = False


IMAGE_DISPLAY_SIZE = (260, 260)


class AppInterface(ctk.CTk):
    """Main application window."""

    def __init__(self) -> None:
        # Optional DnD support: TkinterDnD must wrap the root.
        if _DND_AVAILABLE:
            # CustomTkinter's CTk inherits from tk.Tk; we patch DnD onto it.
            try:
                TkinterDnD._require(self)
            except Exception:
                pass

        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("Adversarial Image Generator")
        self.geometry("1280x820")
        self.minsize(1100, 720)

        # ---- State ---------------------------------------------------- #
        self.model_manager = ModelManager()
        self.defense = DefenseModule()
        self.classes: List[str] = get_imagenet_classes()

        self.current_model_name = ctk.StringVar(value=ModelManager.SUPPORTED[0])
        self.current_attack = ctk.StringVar(value="FGSM")
        self.epsilon_var = ctk.DoubleVar(value=0.03)
        self.iterations_var = ctk.IntVar(value=10)
        self.target_enabled = ctk.BooleanVar(value=False)

        self.original_tensor: Optional[torch.Tensor] = None  # (1,3,H,W) [0,1]
        self.original_pil: Optional[Image.Image] = None
        self.adversarial_tensor: Optional[torch.Tensor] = None
        self.last_result: Optional[AttackResult] = None
        # Cached sign-gradient (FGSM only) for live epsilon adjustment.
        self._fgsm_sign: Optional[torch.Tensor] = None
        self._fgsm_targeted: bool = False

        # Image references so PhotoImages aren't garbage collected.
        self._photo_refs: dict[str, ctk.CTkImage] = {}
        # Persistent transparent placeholder: CTkLabel.configure(image=None) doesn't
        # clear the underlying Tk image reference, so clearing to None and letting the
        # old CTkImage get garbage-collected leaves a stale Tcl image name behind and
        # crashes on the next configure() call. Reusing one never-collected blank image
        # avoids that entirely.
        blank = Image.new("RGBA", IMAGE_DISPLAY_SIZE, (0, 0, 0, 0))
        self._blank_image = ctk.CTkImage(light_image=blank, dark_image=blank,
                                         size=IMAGE_DISPLAY_SIZE)

        self._build_layout()
        self._bind_events()
        self._refresh_status(f"Device: {self.model_manager.device}")

    # ====================================================================
    # Layout
    # ====================================================================
    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_main_area()
        self._build_statusbar()

    def _build_sidebar(self) -> None:
        sidebar = ctk.CTkFrame(self, width=280, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsw")
        sidebar.grid_propagate(False)

        row = 0

        ctk.CTkLabel(sidebar, text="Adversarial Generator",
                     font=ctk.CTkFont(size=18, weight="bold")
                     ).grid(row=row, column=0, padx=16, pady=(16, 8), sticky="w")
        row += 1

        ctk.CTkButton(sidebar, text="Load Image", command=self._on_load_image
                      ).grid(row=row, column=0, padx=16, pady=6, sticky="ew")
        row += 1

        # ---- Model & attack selection -------------------------------- #
        ctk.CTkLabel(sidebar, text="Model").grid(
            row=row, column=0, padx=16, pady=(16, 0), sticky="w"
        )
        row += 1
        ctk.CTkOptionMenu(
            sidebar, values=list(ModelManager.SUPPORTED),
            variable=self.current_model_name, command=lambda _: self._on_model_changed(),
        ).grid(row=row, column=0, padx=16, pady=4, sticky="ew")
        row += 1

        ctk.CTkLabel(sidebar, text="Attack").grid(
            row=row, column=0, padx=16, pady=(8, 0), sticky="w"
        )
        row += 1
        ctk.CTkOptionMenu(
            sidebar, values=["FGSM", "PGD"], variable=self.current_attack,
            command=lambda _: self._on_attack_changed(),
        ).grid(row=row, column=0, padx=16, pady=4, sticky="ew")
        row += 1

        # ---- Epsilon slider ------------------------------------------ #
        self.eps_label = ctk.CTkLabel(sidebar, text="ε = 0.030")
        self.eps_label.grid(row=row, column=0, padx=16, pady=(12, 0), sticky="w")
        row += 1
        self.eps_slider = ctk.CTkSlider(
            sidebar, from_=0.0, to=0.3, number_of_steps=30,
            variable=self.epsilon_var, command=self._on_epsilon_changed,
        )
        self.eps_slider.grid(row=row, column=0, padx=16, pady=4, sticky="ew")
        row += 1

        # ---- Iterations slider --------------------------------------- #
        self.iters_label = ctk.CTkLabel(sidebar, text="Iterations: 10")
        self.iters_label.grid(row=row, column=0, padx=16, pady=(12, 0), sticky="w")
        row += 1
        self.iters_slider = ctk.CTkSlider(
            sidebar, from_=1, to=100, number_of_steps=99,
            variable=self.iterations_var, command=self._on_iters_changed,
        )
        self.iters_slider.grid(row=row, column=0, padx=16, pady=4, sticky="ew")
        row += 1

        # ---- Target selector ----------------------------------------- #
        ctk.CTkCheckBox(
            sidebar, text="Targeted attack", variable=self.target_enabled,
            command=self._on_target_toggle,
        ).grid(row=row, column=0, padx=16, pady=(12, 4), sticky="w")
        row += 1

        self.target_search = ctk.CTkEntry(sidebar, placeholder_text="search class...")
        self.target_search.grid(row=row, column=0, padx=16, pady=4, sticky="ew")
        row += 1

        self.target_listbox = ctk.CTkScrollableFrame(sidebar, height=160)
        self.target_listbox.grid(row=row, column=0, padx=16, pady=4, sticky="ew")
        row += 1
        self._target_buttons: list[ctk.CTkButton] = []
        self._selected_target_idx: Optional[int] = None
        self._populate_target_list("")

        # ---- Action buttons ------------------------------------------ #
        ctk.CTkButton(
            sidebar, text="Attack", fg_color="#b13b3b", hover_color="#8a2929",
            command=self._on_attack,
        ).grid(row=row, column=0, padx=16, pady=(16, 4), sticky="ew")
        row += 1

        defense_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        defense_frame.grid(row=row, column=0, padx=16, pady=4, sticky="ew")
        defense_frame.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkButton(
            defense_frame, text="Defend (Blur)", command=lambda: self._on_defend("gaussian")
        ).grid(row=0, column=0, padx=(0, 4), sticky="ew")
        ctk.CTkButton(
            defense_frame, text="Defend (JPEG)", command=lambda: self._on_defend("jpeg")
        ).grid(row=0, column=1, padx=(4, 0), sticky="ew")
        row += 1

        ctk.CTkButton(
            sidebar, text="Reset", fg_color="gray35", hover_color="gray25",
            command=self._on_reset,
        ).grid(row=row, column=0, padx=16, pady=(8, 16), sticky="ew")

    def _build_main_area(self) -> None:
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.grid(row=0, column=1, sticky="nsew", padx=12, pady=12)
        main.grid_columnconfigure((0, 1, 2), weight=1)
        main.grid_rowconfigure(1, weight=0)
        main.grid_rowconfigure(3, weight=1)

        # Image labels
        titles = ("Original", "Perturbation (×10)", "Adversarial")
        self.image_labels: list[ctk.CTkLabel] = []
        for col, t in enumerate(titles):
            ctk.CTkLabel(main, text=t, font=ctk.CTkFont(size=14, weight="bold")
                         ).grid(row=0, column=col, pady=(0, 4))
            lbl = ctk.CTkLabel(
                main, text="(no image)", width=IMAGE_DISPLAY_SIZE[0],
                height=IMAGE_DISPLAY_SIZE[1], fg_color="gray20", corner_radius=8,
            )
            lbl.grid(row=1, column=col, padx=8, sticky="n")
            self.image_labels.append(lbl)

        # Prediction panels
        ctk.CTkLabel(main, text="Predictions",
                     font=ctk.CTkFont(size=14, weight="bold")
                     ).grid(row=2, column=0, columnspan=3, pady=(16, 4), sticky="w", padx=8)

        self.pred_frames: list[ctk.CTkFrame] = []
        self.pred_widgets: list[list[tuple[ctk.CTkLabel, ctk.CTkProgressBar, ctk.CTkLabel]]] = []
        for col, header in enumerate(("Original", "Perturbation N/A", "Adversarial")):
            frame = ctk.CTkFrame(main)
            frame.grid(row=3, column=col, padx=8, pady=4, sticky="nsew")
            frame.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(frame, text=header, font=ctk.CTkFont(size=12, weight="bold")
                         ).grid(row=0, column=0, padx=8, pady=(8, 4), sticky="w")
            widgets = []
            for i in range(3):
                name_lbl = ctk.CTkLabel(frame, text=f"{i+1}. —", anchor="w")
                name_lbl.grid(row=1 + i * 2, column=0, padx=8, sticky="ew")
                bar = ctk.CTkProgressBar(frame)
                bar.set(0)
                bar.grid(row=2 + i * 2, column=0, padx=8, pady=(0, 4), sticky="ew")
                pct_lbl = ctk.CTkLabel(frame, text="0.00%", anchor="e")
                pct_lbl.grid(row=2 + i * 2, column=0, padx=8, pady=(0, 4), sticky="e")
                widgets.append((name_lbl, bar, pct_lbl))
            self.pred_frames.append(frame)
            self.pred_widgets.append(widgets)

        # Hide the middle prediction frame: noise has no meaningful classification.
        self.pred_frames[1].grid_remove()

    def _build_statusbar(self) -> None:
        self.status_var = ctk.StringVar(value="Ready.")
        status = ctk.CTkLabel(
            self, textvariable=self.status_var, anchor="w",
            font=ctk.CTkFont(size=11),
        )
        status.grid(row=1, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 6))

    # ====================================================================
    # Event wiring
    # ====================================================================
    def _bind_events(self) -> None:
        self.target_search.bind("<KeyRelease>", lambda _e: self._populate_target_list(
            self.target_search.get()
        ))

        # Drag & drop onto the original-image label.
        if _DND_AVAILABLE:
            try:
                self.image_labels[0].drop_target_register(DND_FILES)  # type: ignore[attr-defined]
                self.image_labels[0].dnd_bind("<<Drop>>", self._on_drop)  # type: ignore[attr-defined]
            except Exception:
                pass

    # ====================================================================
    # Status / display helpers
    # ====================================================================
    def _refresh_status(self, msg: str) -> None:
        self.status_var.set(msg)

    def _set_image(self, slot: int, pil_img: Optional[Image.Image]) -> None:
        if pil_img is None:
            self.image_labels[slot].configure(image=self._blank_image, text="(no image)")
            self._photo_refs.pop(str(slot), None)
            return
        display = pil_img.resize(IMAGE_DISPLAY_SIZE, Image.LANCZOS)
        ctk_img = ctk.CTkImage(light_image=display, dark_image=display,
                               size=IMAGE_DISPLAY_SIZE)
        self._photo_refs[str(slot)] = ctk_img
        self.image_labels[slot].configure(image=ctk_img, text="")

    def _update_predictions(self, slot: int, x_tensor: torch.Tensor) -> None:
        bundle = self.model_manager.get(self.current_model_name.get())
        ctx = AttackContext(bundle.model, bundle.mean, bundle.std)
        probs, idxs = attack_engine.predict(ctx, x_tensor, topk=3)
        probs = probs[0].tolist()
        idxs = idxs[0].tolist()

        widgets = self.pred_widgets[slot]
        for i, (name_lbl, bar, pct_lbl) in enumerate(widgets):
            cls_idx = idxs[i]
            p = probs[i]
            name_lbl.configure(text=f"{i+1}. {self.classes[cls_idx]}")
            bar.set(float(p))
            pct_lbl.configure(text=f"{p * 100:.2f}%")

    # ====================================================================
    # Target list
    # ====================================================================
    def _populate_target_list(self, query: str) -> None:
        for btn in self._target_buttons:
            btn.destroy()
        self._target_buttons.clear()

        q = query.strip().lower()
        matches = [
            (i, name) for i, name in enumerate(self.classes)
            if q == "" or q in name.lower()
        ][:200]

        for idx, name in matches:
            b = ctk.CTkButton(
                self.target_listbox, text=f"{idx}: {name}",
                anchor="w", height=22, fg_color="transparent",
                hover_color="gray25",
                command=lambda i=idx: self._on_target_pick(i),
            )
            b.pack(fill="x", padx=2, pady=1)
            self._target_buttons.append(b)

    def _on_target_pick(self, idx: int) -> None:
        self._selected_target_idx = idx
        self.target_enabled.set(True)
        self._refresh_status(f"Target class: {idx} ({self.classes[idx]})")

    def _on_target_toggle(self) -> None:
        if not self.target_enabled.get():
            self._selected_target_idx = None
            self._refresh_status("Untargeted mode.")

    # ====================================================================
    # Slider callbacks
    # ====================================================================
    def _on_epsilon_changed(self, val: float) -> None:
        self.eps_label.configure(text=f"ε = {float(val):.3f}")
        # Live FGSM preview if we have a cached sign-gradient.
        if (self._fgsm_sign is not None
                and self.original_tensor is not None
                and self.current_attack.get() == "FGSM"):
            self._apply_cached_fgsm()

    def _on_iters_changed(self, val: float) -> None:
        self.iters_label.configure(text=f"Iterations: {int(float(val))}")

    def _on_model_changed(self) -> None:
        # Cached gradient is model-specific; invalidate.
        self._fgsm_sign = None
        if self.original_tensor is not None:
            self._update_predictions(0, self.original_tensor)
        if self.adversarial_tensor is not None:
            self._update_predictions(2, self.adversarial_tensor)
        self._refresh_status(f"Model: {self.current_model_name.get()}")

    def _on_attack_changed(self) -> None:
        self._refresh_status(f"Attack: {self.current_attack.get()}")

    # ====================================================================
    # Image loading
    # ====================================================================
    def _on_load_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._load_path(path)

    def _on_drop(self, event) -> None:  # pragma: no cover - GUI
        raw = event.data.strip()
        if raw.startswith("{") and raw.endswith("}"):
            raw = raw[1:-1]
        path = raw.split("} {")[0] if "} {" in raw else raw
        if os.path.isfile(path):
            self._load_path(path)

    def _load_path(self, path: str) -> None:
        try:
            tensor, pil = load_image_as_tensor(path, self.model_manager.device)
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))
            return
        self.original_tensor = tensor
        self.original_pil = pil
        self.adversarial_tensor = None
        self.last_result = None
        self._fgsm_sign = None

        self._set_image(0, pil)
        self._set_image(1, None)
        self._set_image(2, None)
        self._update_predictions(0, tensor)
        for widgets in self.pred_widgets[2]:
            name_lbl, bar, pct_lbl = widgets
            name_lbl.configure(text="—")
            bar.set(0)
            pct_lbl.configure(text="0.00%")
        self._refresh_status(f"Loaded: {os.path.basename(path)}")

    # ====================================================================
    # Attack
    # ====================================================================
    def _current_label(self) -> int:
        assert self.original_tensor is not None
        bundle = self.model_manager.get(self.current_model_name.get())
        ctx = AttackContext(bundle.model, bundle.mean, bundle.std)
        _, idxs = attack_engine.predict(ctx, self.original_tensor, topk=1)
        return int(idxs[0, 0].item())

    def _on_attack(self) -> None:
        if self.original_tensor is None:
            messagebox.showinfo("No image", "Please load an image first.")
            return

        method = self.current_attack.get()
        eps = float(self.epsilon_var.get())
        iters = int(self.iterations_var.get())
        target = self._selected_target_idx if self.target_enabled.get() else None

        self._refresh_status(f"Running {method} (ε={eps:.3f}, iters={iters})...")
        self.update_idletasks()

        # Run on a worker thread to keep the UI responsive.
        threading.Thread(
            target=self._attack_worker,
            args=(method, eps, iters, target),
            daemon=True,
        ).start()

    def _attack_worker(
        self,
        method: str,
        epsilon: float,
        iterations: int,
        target_label: Optional[int],
    ) -> None:
        try:
            bundle = self.model_manager.get(self.current_model_name.get())
            ctx = AttackContext(bundle.model, bundle.mean, bundle.std)
            label = self._current_label()
            result = attack_engine.attack(
                ctx,
                self.original_tensor,  # type: ignore[arg-type]
                label=label,
                method=method,
                target_label=target_label,
                epsilon=epsilon,
                iterations=iterations,
            )
        except Exception:
            err = traceback.format_exc()
            self.after(0, lambda: messagebox.showerror("Attack failed", err))
            self.after(0, lambda: self._refresh_status("Attack failed."))
            return

        # Cache sign-gradient for FGSM live-preview.
        if method.upper() == "FGSM":
            sign = result.perturbation.sign()
            self._fgsm_sign = sign
            self._fgsm_targeted = target_label is not None
        else:
            self._fgsm_sign = None

        self.after(0, lambda: self._apply_attack_result(result))

    def _apply_attack_result(self, result: AttackResult) -> None:
        self.last_result = result
        self.adversarial_tensor = result.adversarial
        self._set_image(1, perturbation_to_pil(result.perturbation, magnify=10.0))
        self._set_image(2, tensor_to_pil(result.adversarial))
        self._update_predictions(2, result.adversarial)
        self._refresh_status(
            f"Attack complete ({self.current_attack.get()}, "
            f"ε={self.epsilon_var.get():.3f})."
        )

    def _apply_cached_fgsm(self) -> None:
        """Recompute adversarial from cached sign-gradient when ε changes."""
        if self._fgsm_sign is None or self.original_tensor is None:
            return
        eps = float(self.epsilon_var.get())
        if self._fgsm_targeted:
            adv = self.original_tensor - eps * self._fgsm_sign
        else:
            adv = self.original_tensor + eps * self._fgsm_sign
        adv = adv.clamp(0.0, 1.0)
        delta = adv - self.original_tensor
        self.adversarial_tensor = adv
        self._set_image(1, perturbation_to_pil(delta, magnify=10.0))
        self._set_image(2, tensor_to_pil(adv))
        self._update_predictions(2, adv)

    # ====================================================================
    # Defense
    # ====================================================================
    def _on_defend(self, method: str) -> None:
        if self.adversarial_tensor is None:
            messagebox.showinfo("No adversarial", "Run an attack first.")
            return
        try:
            cleaned = self.defense.apply(self.adversarial_tensor, method)
        except Exception as exc:
            messagebox.showerror("Defense failed", str(exc))
            return
        self._set_image(2, tensor_to_pil(cleaned))
        self._update_predictions(2, cleaned)
        self._refresh_status(f"Applied defense: {method}.")

    # ====================================================================
    # Reset
    # ====================================================================
    def _on_reset(self) -> None:
        self.adversarial_tensor = None
        self.last_result = None
        self._fgsm_sign = None
        self._set_image(1, None)
        if self.original_pil is not None:
            self._set_image(2, None)
        for widgets in self.pred_widgets[2]:
            name_lbl, bar, pct_lbl = widgets
            name_lbl.configure(text="—")
            bar.set(0)
            pct_lbl.configure(text="0.00%")
        self._refresh_status("Reset.")
