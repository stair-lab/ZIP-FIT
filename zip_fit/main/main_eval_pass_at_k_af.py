import wandb 
import os
import time
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import math
from typing import List

from huggingface_hub import create_repo, upload_file, whoami, login

from pantograph import Server

from utils import seed_everything
from metrics.lean4_comp_pass_at_k import run_lean4_comp_pass_k_unbiased_eval

def main_eval_lean4_model_performance_pass_at_k(config: dict = {}) -> None:
    """
    Evaluates language model performance on Lean4 theorem statement translation using pass@k methodology.

    To run (without indent so I can copy-paste):
conda activate zip_fit; export CUDA_VISIBLE_DEVICES=1; python /lfs/skampere1/0/brando9/ZIP-FIT/zip_fit/lean_pass_k_unbiased.py 
    
    Parameters:
        config (dict): Configuration dictionary that can override default parameters
        
    Returns:
        None: Results are printed to console and logged with wandb if configured
    """
    seed_everything(config.get('seed', 42))

    # 0) PyPantograph Lean4 Server
    server = Server(imports=["Mathlib", "Init"], project_path=os.path.expanduser("~/mathlib4"))
    # 1) Manual snippet test
    # test_manual_snippets(server)

    # 2) Log In
    key_file_path = "~/keys/master_hf_token.txt"
    key_file_path = os.path.abspath(os.path.expanduser(key_file_path))
    with open(key_file_path, "r", encoding="utf-8") as f:
        token = f.read().strip()
    login(token=token)
    os.environ['HUGGINGFACE_TOKEN'] = token
    user_info = whoami()
    print(f"Currently logged in as: {user_info['name']}\n")

    # 3) Model pass@k test (toy)
    model_name = 'gpt2'
    model_name = 'UDACA/math-gpt2-zipfit'
    model_name = 'UDACA/math-gpt2-dsir'
    model_name = 'UDACA/math-gpt2-less'
    # model_name = 'internlm/internlm2-math-plus-1_8b'
    model_name = 'google/gemma-2-2b'
    model_name = 'UDACA/math-gemma-2-2b-zipfit'
    model_name = 'UDACA/math-gemma-2-2b-less'
    model_name = 'UDACA/math-gemma-2-2b-dsir'
    # model_name = 'mistralai/Mistral-7B-v0.1'
    # model_name = 'meta-llama/Meta-Llama-3-8B'
    # model_name = 'google/gemma-2-2b-it'
    print(f'f{model_name=}') 
    # Sanity Check Data
    # def my_prompt_format(nl_stmt: str) -> str:
    #     return (
    #         "Your task is translate the natural language version of the mathematical statement "
    #         "to a formal Lean statement version, using the following format:\n"
    #         "natural language statement:\nSuppose that $f$ is holomorphic in an open set $\Omega$. Prove that if $|f|$ is constant, then $f$ is constant.\n"
    #         "formal Lean language statement:##\ntheorem exercise_1_13c {f : ℂ → ℂ} (Ω : Set ℂ) (a b : Ω) (h : IsOpen Ω) (hf : DifferentiableOn ℂ f Ω) (hc : ∃ (c : ℝ), ∀ z ∈ Ω, abs (f z) = c) : f a = f b:= sorry\n##"
    #         "natural language statement:\nProve that the power series $\sum zn/n^2$ converges at every point of the unit circle.\n"
    #         "formal Lean language statement:##\ntheorem exercise_1_19b (z : ℂ) (hz : abs z = 1) (s : ℕ → ℂ) (h : s = (λ n => ∑ i in (range n), i * z / i ^ 2)) : ∃ y, Tendsto s atTop (𝓝 y):= sorry\n##"
    #         "natural language statement:\nSuppose $f$ is continuous in a region $\Omega$. Prove that any two primitives of $f$ (if they exist) differ by a constant.\n"
    #         "formal Lean language statement:##\ntheorem exercise_1_26 (f F₁ F₂ : ℂ → ℂ) (Ω : Set ℂ) (h1 : IsOpen Ω) (h2 : IsConnected Ω) (hF₁ : DifferentiableOn ℂ F₁ Ω) (hF₂ : DifferentiableOn ℂ F₂ Ω) (hdF₁ : ∀ x ∈ Ω, deriv F₁ x = f x) (hdF₂ : ∀ x ∈ Ω, deriv F₂ x = f x) : ∃ c : ℂ, ∀ x, F₁ x = F₂ x + c:= sorry\n##"
    #         f"natural language statement:\n{nl_stmt}\n"
    #         "formal Lean language statement:"
    #     )
    def my_prompt_format(nl_stmt: str) -> str:
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
    from datasets import load_dataset
    ds_test = load_dataset('UDACA/proofnet-v3-lean4', split='test')
    # ds_test = ds_test.select(list(range(10)))

    # Promptify & get Gold Truth Headers
    prompts = [my_prompt_format(row['nl_statement']) for row in ds_test]
    # model_name = None
    # prompts = [f"##\n{row['formal_statement']}\n##" for row in ds_test]
    # print(f'{prompts=}')
    gold_headers = [row['header_no_import'] for row in ds_test]
    # print(f'{gold_headers}=')
    print(f'Number prompts: {len(prompts)=}')
    print(f'Number of gold headers: {len(gold_headers)=}')

    # Start timer
    global_start_time = time.time()  # Start overall timer

    debug = True
    # debug = False
    eval_batch_size = 32 # for vllm how many prompts to batch for speed
    k = 5
    # num_samples = 20
    # num_samples = 40
    num_samples = 5000
    score = run_lean4_comp_pass_k_unbiased_eval(prompts, model_name, server, headers=gold_headers, k=k, num_samples=num_samples, eval_batch_size=eval_batch_size, debug=debug)
    print(f"\n==== For {model_name} Final Average Pass@{k=}N={num_samples} across {len(prompts)} tasks: {score:.3f} ====\n")

    # End overall timer
    global_end_time = time.time()
    total_seconds = global_end_time - global_start_time
    print(f"\nDone. Total run time for all models: {total_seconds:.2f} seconds, {total_seconds/60:.2f} minutes, {total_seconds/3600:.2f} hours.\a")


def main_experiment_pass_k_vs_N_config(config: dict = {}):
    """
    Conducts systematic pass@k experiments to analyze how model performance and evaluation time scale with sample size.
    
    This experiment:
    1) Evaluates model performance across different sample sizes (N values)
    2) Repeats each experiment multiple times to measure variance
    3) Computes confidence intervals for both performance and runtime
    4) Generates visualizations of the results
    5) Logs all metrics and plots to wandb
    
    Configuration parameters:
      - n_start: Starting value for sample size N
      - n_end: Ending value for sample size N
      - num_points: Number of evenly spaced N values to evaluate
      - num_reps: Number of repetitions per N to estimate variance
      - k: The k value for pass@k evaluation metric
      - plot_title: Title for generated plots
      - model_name: Model identifier to evaluate
      - seed: Base random seed for reproducibility
    
    Returns:
        None: Results are saved to disk and logged to wandb
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import math
    from typing import List

    # Log In
    from huggingface_hub import login, whoami
    key_file_path = "~/keys/master_hf_token.txt"
    key_file_path = os.path.abspath(os.path.expanduser(key_file_path))
    with open(key_file_path, "r", encoding="utf-8") as f:
        token = f.read().strip()
    login(token=token)
    os.environ['HUGGINGFACE_TOKEN'] = token
    user_info = whoami()
    print(f"Currently logged in as: {user_info['name']}\n")

    # ---------------------------------------------------------------------
    # Set the random seed from the configuration (default seed = 42)
    seed_everything(config.get('seed', 42))
    # conda activate zip_fit
    # conda activate zip_fit; export CUDA_VISIBLE_DEVICES=0; python ~/ZIP-FIT/zip_fit/lean_pass_k_unbiased.py --model_name "gpt2" 
    # conda activate zip_fit; export CUDA_VISIBLE_DEVICES=1; python ~/ZIP-FIT/zip_fit/lean_pass_k_unbiased.py --model_name "Qwen/Qwen2.5-0.5B" 
    # conda activate zip_fit; export CUDA_VISIBLE_DEVICES=2; python ~/ZIP-FIT/zip_fit/lean_pass_k_unbiased.py --model_name "meta-llama/Llama-3.2-1B" 
    # conda activate zip_fit; export CUDA_VISIBLE_DEVICES=3; python ~/ZIP-FIT/zip_fit/lean_pass_k_unbiased.py --model_name "google/gemma-2-2b'"
    # conda activate zip_fit; export CUDA_VISIBLE_DEVICES=3; python ~/ZIP-FIT/zip_fit/lean_pass_k_unbiased.py --model_name "UDACA/math-gemma-2-2b-zipfit"
    # conda activate zip_fit; export CUDA_VISIBLE_DEVICES=3; python ~/ZIP-FIT/zip_fit/lean_pass_k_unbiased.py --model_name "UDACA/math-gemma-2-2b-dsir"
    # conda activate zip_fit; export CUDA_VISIBLE_DEVICES=3; python ~/ZIP-FIT/zip_fit/lean_pass_k_unbiased.py --model_name "UDACA/math-gemma-2-2b-less"
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # ---------------------------------------------------------------------
    # Read configuration parameters with defaults
    n_start    = config.get('n_start', 50)          # Starting value for N
    n_end      = config.get('n_end', 500)           # Ending value for N
    num_points = config.get('num_points', 10)      # Number of N values between n_start and n_end
    num_reps   = config.get('num_reps', 5)           # Repetitions per N
    k          = config.get('k', 5)                # The k value for pass@k (e.g., pass@5)
    plot_title = config.get('plot_title', "Pass@5 vs. N and Evaluation Time")
    # model_name      = config.get('model_name', 'SmolLM-135M')
    model_name      = config.get('model_name', 'gpt2')
    model_name      = config.get('model_name', 'Qwen/Qwen2.5-0.5B')
    model_name      = config.get('model_name', 'meta-llama/Llama-3.2-1B')
    model_name      = config.get('model_name', 'google/gemma-2-2b')
    model_name      = config.get('model_name', 'UDACA/math-gemma-2-2b-zipfit')
    model_name      = config.get('model_name', 'UDACA/math-gemma-2-2b-less')
    model_name      = config.get('model_name', 'UDACA/math-gemma-2-2b-dsir')
    print(f'{model_name=}')
    
    # ---------------------------------------------------------------------
    # Load prompts and gold headers from a dataset.
    # We assume that our prompt formatting function is defined below.
    def my_prompt_format(nl_stmt: str) -> str:
        return (
            "Your task is translate the natural language version of the mathematical statement "
            "to a formal Lean statement version, using the following format:\n"
            "natural language statement:\nLet $z=\\frac{1+i}{\\sqrt{2}}.$What is $\\left(z^{1^2}+z^{2^2}+z^{3^2}+\\dots+z^{{12}^2}\\right) \\cdot "
            "\\left(\\frac{1}{z^{1^2}}+\\frac{1}{z^{2^2}}+\\frac{1}{z^{3^2}}+\\dots+\\frac{1}{z^{{12}^2}}\\right)?$ "
            "$\\textbf{(A)}\\ 18 \\qquad \\textbf{(B)}\\ 72-36\\sqrt2 \\qquad \\textbf{(C)}\\ 36 \\qquad \\textbf{(D)}\\ 72 \\qquad \\textbf{(E)}\\ 72+36\\sqrt2$ "
            "Show that it is \\textbf{(C)}\\ 36.\n"
            "formal Lean language statement:##\ntheorem amc12a_2019_p21 (z : ℂ) (h₀ : z = (1 + Complex.I) / Real.sqrt 2) : "
            "(∑ k in Finset.Icc 1 12, (z^(k^2))) * (∑ k in Finset.Icc 1 12, (1 / z^(k^2))) = 36 := sorry\n##"
            "natural language statement:\nIntegers $x$ and $y$ with $x>y>0$ satisfy $x+y+xy=80$. What is $x$? "
            "$\\textbf{(A)}\\ 8 \\qquad\\textbf{(B)}\\ 10 \\qquad\\textbf{(C)}\\ 15 \\qquad\\textbf{(D)}\\ 18 \\qquad\\textbf{(E)}\\ 26$ Show that it is \\textbf{(E)}\\ 26.\n"
            "formal Lean language statement:##\ntheorem amc12a_2015_p10 (x y : ℤ) (h₀ : 0 < y) (h₁ : y < x) "
            "(h₂ : x + y + (x * y) = 80) : x = 26 := sorry\n##"
            "natural language statement:\nWhat is the [[volume]] of a [[cube]] whose [[surface area]] is twice that of a cube with volume 1? "
            "$\\mathrm{(A)}\\ \\sqrt{2}\\qquad\\mathrm{(B)}\\ 2\\qquad\\mathrm{(C)}\\ 2\\sqrt{2}\\qquad\\mathrm{(D)}\\ 4\\qquad\\mathrm{(E)}\\ 8$ Show that it is \\mathrm{(C)}.\n"
            "formal Lean language statement:##\ntheorem amc12a_2008_p8 (x y : ℝ) (h₀ : 0 < x ∧ 0 < y) "
            "(h₁ : y^3 = 1) (h₂ : 6 * x^2 = 2 * (6 * y^2)) : x^3 = 2 * Real.sqrt 2 := sorry\n##"
            f"natural language statement:\n{nl_stmt}\n"
            "formal Lean language statement:"
        )
    from datasets import load_dataset
    ds_test = load_dataset('UDACA/proofnet-v3-lean4', split='test')
    # ds_test = ds_test.select(list(range(185)))  # optionally select a subset
    prompts = [my_prompt_format(row['nl_statement']) for row in ds_test]
    gold_headers = [row['header_no_import'] for row in ds_test]
    print(f'Number prompts: {len(prompts)=}')
    print(f'Number of gold headers: {len(gold_headers)=}')
    
    # ---------------------------------------------------------------------
    # Generate a list of N values.
    # Instead of a fixed step, we generate "num_points" evenly spaced values from n_start to n_end.
    arr_float = np.linspace(n_start, n_end, num_points)
    # Round the floats to the nearest integer.
    N_values = np.rint(arr_float).astype(int)
    N_values = [int(N) for N in N_values]
    
    # ---------------------------------------------------------------------
    # Initialize lists to store statistics across N values:
    mean_passk_per_N = []  # Mean pass@k for each N
    std_passk_per_N = []   # Std dev of pass@k for each N
    ci_passk_array = []    # 95% CI for pass@k for each N
    avg_times = []         # Average evaluation time (sec) for each N
    time_stds = []         # Std dev of evaluation times for each N
    ci_time_array = []     # 95% CI for evaluation time for each N
    
    # ---------------------------------------------------------------------
    # Initialize the Lean 4 server via PyPantograph.
    server = Server(imports=["Mathlib", "Init"], project_path=os.path.expanduser("~/mathlib4"))
    
    print("Starting experiments...")
    global_start_time = time.time()  # Overall experiment timer
    
    # Get the base seed from the config (default 42)
    base_seed = config.get('seed', 42)
    
    # Loop over each N value (each representing the number of completions generated per prompt)
    for i, N in enumerate(tqdm.tqdm(N_values, desc="Evaluating N values")):
        print(f'{model_name=}')
        rep_start_time = time.time()  # Timer for all repetitions at this N
        
        # Lists to record pass@k scores and evaluation times for each repetition at this N
        passk_runs = []
        rep_times = []
        
        # Run the experiment num_reps times for the current N to assess variance.
        for rep in range(num_reps):
            # Calculate a new seed for this repetition to ensure variability.
            new_seed = base_seed + (i * num_reps) + rep
            seed_everything(new_seed)  # Set all random seeds
            
            # Record the start time for this repetition.
            start_time = time.time()
            # Call run_pass_k_eval which:
            #   - Generates 'N' completions per prompt,
            #   - Checks each for correctness (e.g., compilation),
            #   - Computes and returns the overall pass@k score.
            score = run_lean4_comp_pass_k_unbiased_eval(
                prompts=prompts,
                model_name=model_name,
                server=server,
                headers=gold_headers,
                k=k,
                num_samples=N,
                eval_batch_size=32,
                seed=new_seed,
                debug=False
            )
            # Record the end time and compute the elapsed time for this repetition.
            end_time = time.time()
            rep_time = end_time - start_time
            
            # Append the results from this repetition.
            passk_runs.append(score)
            rep_times.append(rep_time)

        rep_end_time = time.time()
        elapsed_N = rep_end_time - rep_start_time
        
        # Compute statistics for pass@k scores for this N.
        avg_passk = np.mean(passk_runs)
        std_passk = np.std(passk_runs, ddof=1)
        ci_passk = 1.96 * (std_passk / math.sqrt(num_reps))
        
        # Compute statistics for evaluation times for this N.
        avg_time = np.mean(rep_times)
        std_time = np.std(rep_times, ddof=1)
        ci_time = 1.96 * (std_time / math.sqrt(num_reps))
        
        # Save the computed statistics.
        mean_passk_per_N.append(avg_passk)
        std_passk_per_N.append(std_passk)
        ci_passk_array.append(ci_passk)
        avg_times.append(avg_time)
        time_stds.append(std_time)
        ci_time_array.append(ci_time)

        wandb.log({
                "N": N,
                "avg_passk": avg_passk,
                "std_passk": std_passk,
                "ci_passk": ci_passk,
                "avg_time": avg_time,
                "std_time": std_time,
                "ci_time": ci_time,
            })
        
        # Print a summary for the current N.
        print(f"N={N}, Pass@{k} mean={avg_passk:.4f}, stdev={std_passk:.4f}, CI(95%)={ci_passk:.4f}, "
              f"avg time={avg_time:.2f} sec, time CI={ci_time:.2f} sec for {num_reps} reps")
    
    # End overall experiment timer.
    global_end_time = time.time()
    total_seconds = global_end_time - global_start_time
    print(f"\nDone. Total experiment time: {total_seconds:.2f} seconds, "
          f"{total_seconds/60:.2f} minutes, {total_seconds/3600:.2f} hours.")
    
    # Print detailed results for each N value.
    print("\nDetailed Results:")
    for N, mean_val, std_val, ci_val, t_avg, t_ci in zip(N_values, mean_passk_per_N, std_passk_per_N, ci_passk_array, avg_times, ci_time_array):
        print(f"N={N}, Pass@{k} mean={mean_val:.4f}, stdev={std_val:.4f}, CI(95%)={ci_val:.4f}, "
              f"avg time={t_avg:.2f} sec, time CI={t_ci:.2f} sec")
    
    # ---------------------------------------------------------------------
    # Plotting:
    # Create two subplots: one for pass@k and one for evaluation time.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot 1: N vs. Mean pass@k with 95% CI error bars.
    ax1.errorbar(N_values, mean_passk_per_N, yerr=ci_passk_array, fmt='o-', capsize=5, ecolor='red', color='blue')
    ax1.set_title(f"{plot_title} (Pass@{k})")
    ax1.set_xlabel("N (Number of completions generated)")
    ax1.set_ylabel(f"Mean Pass@{k} ± 95% CI")
    ax1.grid(True)
    
    # Plot 2: N vs. Average evaluation time with 95% CI error bars.
    ax2.errorbar(N_values, avg_times, yerr=ci_time_array, fmt='o-', capsize=5, ecolor='red', color='green')
    ax2.set_title("Evaluation Time vs. N")
    ax2.set_xlabel("N (Number of completions generated)")
    ax2.set_ylabel("Average Evaluation Time (sec) ± 95% CI")
    ax2.grid(True)
    
    # Save the figure locally.
    plot_file = "pass_at_k_and_time_plot.png"
    fig.savefig(plot_file)
    print(f"\nPlot saved to {plot_file}")
    
    # Log the plot image to wandb.
    wandb.log({"pass_at_k_and_time_plot": wandb.Image(plot_file)})
    
    # Display the plots.
    plt.show()


def _main(**kwargs):
    from datetime import datetime
    from socket import gethostname
    import wandb
    today = datetime.now().strftime('%Y_m%m_d%d_t%Hh_%Mm_%Ss') # eg '2024_m01_d22_t13h_00m_30s'
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(kwargs.get('CUDA_VISIBLE_DEVICES', '7'))
    tmux_sess_num = None
    kwargs = kwargs | {'today': today, 'tmux_sess_num': tmux_sess_num, 'hostname': gethostname()}
    run_name = f'{kwargs}' 
    # run = wandb.init(mode=kwargs.get('mode', 'online'), project="zip-fit-pass-at-k-af", name=run_name, save_code=True, config=kwargs)
    run = wandb.init(mode=kwargs.get('mode', 'dryrun'), project="zip-fit-pass-at-k-af", name=run_name, save_code=True, config=kwargs)
    wandb.save(__file__) # save current code now, don't wait to wandb.finish, also useful: wandb.save("*.py") # upload all .py files in current directory
    print(f'Kwargs to run:\n{kwargs}')
    main_eval_lean4_model_performance_pass_at_k(kwargs)
    # main_experiment_pass_k_vs_N_config(kwargs)
    run.alert(title="Run Completed", text=f"Run finished, run url: {run.get_url()}")
    print(f'{run.get_url()=}')
    wandb.finish()

if __name__ == "__main__":
    import fire
    import time
    start_time = time.time()
    fire.Fire(_main)
    print(f"\aTime taken: {time.time() - start_time:.2f} seconds, or {(time.time() - start_time) / 60:.2f} minutes, or {(time.time() - start_time) / 3600:.2f} hours.\a")
