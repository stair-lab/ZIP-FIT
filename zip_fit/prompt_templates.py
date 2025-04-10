def default_prompt_fn_name(nl_stmt: str) -> str:
    """
    This is the default prompt template that is used to translate a natural language statement to a formal Lean statement version.
    """
    return (
        f"natural language statement:\n{nl_stmt}\n"
        "formal Lean language statement:"
    )

def prompt_temp_fn_lean4_pn_val_v0(nl_stmt: str) -> str:
    """
    This prompt template is used to translate a natural language statement to a formal Lean statement version from:
        https://huggingface.co/datasets/UDACA/proofnet-v3-lean4/
    if testing on pn avoid using this prompt template to avoid accidental model cheating.
    """
    return (
        "Your task is translate the natural language version of the mathematical statement "
        "to a formal Lean statement version, using the following format:\n"
        "natural language statement:\nSuppose that $f$ is holomorphic in an open set $\Omega$. Prove that if $|f|$ is constant, then $f$ is constant.\n"
        "formal Lean language statement:##\ntheorem exercise_1_13c {f : ℂ → ℂ} (Ω : Set ℂ) (a b : Ω) (h : IsOpen Ω) (hf : DifferentiableOn ℂ f Ω) (hc : ∃ (c : ℝ), ∀ z ∈ Ω, abs (f z) = c) : f a = f b:= sorry\n##"
        "natural language statement:\nProve that the power series $\sum zn/n^2$ converges at every point of the unit circle.\n"
        "formal Lean language statement:##\ntheorem exercise_1_19b (z : ℂ) (hz : abs z = 1) (s : ℕ → ℂ) (h : s = (λ n => ∑ i in (range n), i * z / i ^ 2)) : ∃ y, Tendsto s atTop (𝓝 y):= sorry\n##"
        "natural language statement:\nSuppose $f$ is continuous in a region $\Omega$. Prove that any two primitives of $f$ (if they exist) differ by a constant.\n"
        "formal Lean language statement:##\ntheorem exercise_1_26 (f F₁ F₂ : ℂ → ℂ) (Ω : Set ℂ) (h1 : IsOpen Ω) (h2 : IsConnected Ω) (hF₁ : DifferentiableOn ℂ F₁ Ω) (hF₂ : DifferentiableOn ℂ F₂ Ω) (hdF₁ : ∀ x ∈ Ω, deriv F₁ x = f x) (hdF₂ : ∀ x ∈ Ω, deriv F₂ x = f x) : ∃ c : ℂ, ∀ x, F₁ x = F₂ x + c:= sorry\n##"
        f"natural language statement:\n{nl_stmt}\n"
        "formal Lean language statement:"
    )

def prompt_temp_fn_lean4_minif2f_val_v0(nl_stmt: str) -> str:
    """
    This prompt template is used to translate a natural language statement to a formal Lean statement version from:
        https://huggingface.co/datasets/UDACA/minif2f-lean4/
    if testing on minif2f avoid using this prompt template to avoid accidental model cheating.
    """
    return (
        "Your task is translate the natural language version of the mathematical statement "
        "to a formal Lean statement version, using the following format:\n"
        "natural language statement:\nLet $z=\frac{1+i}{\sqrt{2}}.$What is $\left(z^{1^2}+z^{2^2}+z^{3^2}+\dots+z^{{12}^2}\right) \cdot \left(\frac{1}{z^{1^2}}+\frac{1}{z^{2^2}}+\frac{1}{z^{3^2}}+\dots+\frac{1}{z^{{12}^2}}\right)?$ $\textbf{(A) } 18 \qquad \textbf{(B) } 72-36\sqrt2 \qquad \textbf{(C) } 36 \qquad \textbf{(D) } 72 \qquad \textbf{(E) } 72+36\sqrt2$ Show that it is \textbf{(C) }36.\n"
        "formal Lean language statement:##\ntheorem amc12a_2019_p21 (z : ℂ) (h₀ : z = (1 + Complex.I) / Real.sqrt 2) : (∑ k in Finset.Icc 1 12, (z^(k^2))) * (∑ k in Finset.Icc 1 12, (1 / z^(k^2))) = 36 := sorry\n##"
        "natural language statement:\nIntegers $x$ and $y$ with $x>y>0$ satisfy $x+y+xy=80$. What is $x$? $ \textbf{(A)}\ 8 \qquad\textbf{(B)}\ 10 \qquad\textbf{(C)}\ 15 \qquad\textbf{(D)}\ 18 \qquad\textbf{(E)}\ 26$ Show that it is \textbf{(E)}\ 26.\n"
        "formal Lean language statement:##\ntheorem amc12a_2015_p10 (x y : ℤ) (h₀ : 0 < y) (h₁ : y < x) (h₂ : x + y + (x * y) = 80) : x = 26 := sorry\n##"
        "natural language statement:\nWhat is the [[volume]] of a [[cube]] whose [[surface area]] is twice that of a cube with volume 1? $\mathrm{(A)}\ \sqrt{2}\qquad\mathrm{(B)}\ 2\qquad\mathrm{(C)}\ 2\sqrt{2}\qquad\mathrm{(D)}\ 4\qquad\mathrm{(E)}\ 8$ Show that it is \mathrm{(C)}.\n"
        "formal Lean language statement:##\ntheorem amc12a_2008_p8 (x y : ℝ) (h₀ : 0 < x ∧ 0 < y) (h₁ : y^3 = 1) (h₂ : 6 * x^2 = 2 * (6 * y^2)) : x^3 = 2 * Real.sqrt 2 := sorry\n##"
        f"natural language statement:\n{nl_stmt}\n"
        "formal Lean language statement:"
    )