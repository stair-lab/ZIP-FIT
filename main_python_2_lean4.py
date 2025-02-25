# generate_dspy_synth_generators/main_python_2_lean4.py

import dspy
from dspy import Evaluate
from pantograph.server import Server # -- lean4 server
import os

# Real Server That Talks to Lean 4 from Python (and loads two Lean 4 libraries)
server = Server(imports=["Mathlib", "Init"], project_path=os.path.expanduser("~/mathlib4"))
def run_lean_code(lean4_code: str) -> tuple[str]:
    """
    Compiles the given Lean 4 code. The server tries to compile the given Lean 4 code and gives
    all the compilation units per code statement/block written in the lean4_code given as a list of
    success of error messages. 
    """   
    compilation_units: list[str] = server.load_sorry(lean4_code)
    return compilation_units

# score: int number of Lean4 errors
def count_number_of_errors(compilation_unit: list) -> int:
    """ 
    Returns how many Lean4 errors were present (e.g., unit tests failed, theorems truly failed, syntax errors etc.)

    Note: We don't normalize because we don't know beforehand the number of errors the model will produce.
    Eg it might produce more errors than the number of unit tests or theorems if there are syntax errors. 
    """
    num_errors: int = 0
    # Check each compilation unit for error messages or leftover goals
    for compilation_unit in compilation_units:
        # If any message includes 'error', we consider it a compile failure
        for msg in compilation_unit.messages:
            if 'error' in msg.lower():
                num_errors += 1
    return num_errors

def lean_metric(example: dspy.Example, pred: dspy.Predict, trace=None) -> int:
    """Returns the number of errors Lean 4 errors. Note, zero is the highest score."""
    score: int = count_number_of_errors(pred.lean4_code)
    return score

class PythonBad2PythonBetter(dspy.Signature):
    """Generate higher quality Python code form the initial one given, respect the original intended semantics and keep the original unit tests if present."""
    in_py_code: str = dspy.InputField(desc="Python code with a bad docstring and missing or bad unit tests.")
    out_py_code: str = dspy.InputField(desc="Python code with an excellent docstring and comprehesive unit tests, " 
                                          "with edge cases covered.")

class Python2Lean4_SingleCodeInputOutput(dspy.Signature):
    """
    Generates a high quality Lean 4 code (with a docstrings, comments, unit tests, and theorems of correctnes, proofs with := sorry) 
    from Python Code (python code possibly with docstrings and unit tests).
    """
    code: str = dspy.InputField(desc="Python code possibly with docstrings and unit tests")
    lean4_code: str = dspy.OutputField(desc="Lean 4 Code with corresponding docstrings, unit tests, correctness theorem(s), := sorry proofs. " 
                                            "If the unit tests from the Python Code are missing, write some for our Lean 4 code. "
                                            "The correctness theorem should be basic properties the Lean 4 Code "
                                            "according to the specifications and a final correctness theorem stating an essential"
                                            "property of the Lean 4 code.")

class SynthGen_Python_2_Lean4(dspy.Module):
    """Generates Translation from Python to high quality correct Lean 4 code."""
    def __init__(self, py2lean4_fewshotset: list[str]):
        super().__init__()
        self.react_py = dspy.ReAct(PythonBad2PythonBetter, toosl=[dspy.PythonInterpreter()])
        self.react_lean4 = dspy.ReAct(Python2Lean4_SingleCodeInputOutput, tools=[lean_metric])
        self.py2lean4_fewshotset = py2lean4_fewshotset
        # self.retrieve = dspy.Retrieve(k=3)

    def no_docstring_or_no_unit_tests(code: dspy.Example):
        """ Returns true if missing unit tests or no docstrings (approximately). Alternatively we could always run it to improve the python code given. """
        return 'assert' not in code.py_code or "\"\"\"" not in code.py_code

    def forward(self, code: dspy.Example) -> dspy.Prediction:
        """From python code (possibly with docstrings, unit tests) --> generates high quality correct Lean 4 code with corresponding docstrings, unit tests, and correctness theorems."""
        # Improve quality of input Python code
        better_py_code: str = self.react_py(in_py_code=code.py_code).out_py_code

        # Get the few-shot examples from our few-shots set's benchmark (slight hack, using the retrieval to use the labeled example as input)
        # py2lean4_fewshot_examples: list[str] = self.retrieve(py_code).passages
        py2lean4_fewshot_examples: str = self.py2lean4_fewshotset.join('\n')

        # Create a Prompt for the Lean 4 react module given the selected few-shots
        py_code_with_gold_few_shots = (
            f'Target Python to Translate to Lean 4:\n{better_py_code}\n\n'
            f'Examples of Python to Lean4 translations:\n{py2lean4_fewshot_examples}\n\n'
            f'Do the Translation:\n'
        )

        # Use our Prompt with few-shots with our react module to Generate the Lean 4 code
        lean4_code: dspy.Predict = self.react(code=py_code_with_gold_few_shots)
        return lean4_code

def main(config: dict = {}):
    import dspy
    import random
    from generate_dspy_synth_generators.data import generate_dspy_synth_generators, load_json_data
    from generate_dspy_synth_generators.data import create_few_shot_prompt

    # Setting up DSPy
    api_key = open(os.path.expanduser('~/keys/openai_api_key_brandos_koyejolab.txt')).read().strip()
    dspy.settings.configure(lm=dspy.LM('openai/gpt-4o-mini', api_key=api_key))
    # dspy.settings.configure(lm=dspy.LM('openai/gpt-4o-mini', api_key=api_key), rm=dspy.ColBERTv2(url=devset1))
    # lm = dspy.LM("openai/meta-llama/Llama-3.1-8B-Instruct", api_base="http://localhost:7501/v1", api_key="local", model_type='chat')
    # Test Call to LM directly
    response = lm(messages=[{"role": "user", "content": "Say this is a test!"}], temperature=0.7)  # => ['This is a test!']
    print(f'LM test, response: {response}\nLoaded LM: {lm.model}')

    # Set up data sets
    raw_data: list[dspy.Example] = generate_dspy_synth_generators(n_dspyset=config.get('config', 500))
    random.shuffle(raw_data)
    print(f'Raw data set: {len(raw_data)}')
    benchmark: list[str] = load_json_data(config.get('veribench_path', '~/veribench/data/veribench_02_12_2025')) 

    # might contained unlabeled & labeled examples
    trainset: list[dspy.Example] = raw_data[:config.get('n_train', 100)]
    # contains labeled examples
    fewshotset: list[str] = [create_few_shot_prompt('~/veribench/py_src/humaneval_0_hasCloseElements.py', '~/veribench/lean_src_proj/veribench/human_eval_lean4/humaneval_0_hasCloseElements.lean')]
    # might contained unlabeled & labeled examples
    devset: list[dspy.Example] = raw_data[config.get('n_dev', 100):]
    # might contained unlabeled & labeled examples
    testset: list[dspy.Example] = benchmark[config.get('n_dev', 7):]
    print(f'{len(trainset)=}\n{len(devset)=}\n{len(testset)=}')

    # Create Synth Gen Module
    synth_gen = SynthGen_Python_2_Lean4()
    print(f"Test gen: {synth_gen('lambda x, y: x + y')}")
    dspy.inspect_history(1)

    # Do DSPy train (compile, optimize)
    tp = dspy.MIPROv2(metric=check_passes, auto="light", num_threads=24)
    optimized_react = tp.compile(synth_gen, trainset=trainset)

    # Evaluate number of errors the synthetic data has (note: currently ignoring labels, gold reference for now)
    evaluator = Evaluate(devset=devset, metric=lean_metric, num_threads=24, display_progress=True, display_table=5) 
    final_score: int = evaluator(synth_gen)
    print(f'Final Score using the train set: {final_score}')

def get_current_tmux_session_number() -> str:
    import os
    # Executes "tmux display-message -p '#S'" to retrieve the current tmux session name/number,
    # reads the output from the command, strips any surrounding whitespace, and returns it.
    return os.popen("tmux display-message -p '#S'").read().strip()

def _main(**kwargs):
    from datetime import datetime
    from socket import gethostname
    import wandb
    today = datetime.now().strftime('%Y_m%m_d%d_t%Hh_%Mm_%Ss') # eg '2024_m01_d22_t13h_00m_30s'
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(kwargs.get('CUDA_VISIBLE_DEVICES', '7'))
    tmux_sess_num = None
    kwargs = kwargs | {'today': today, 'tmux_sess_num': tmux_sess_num, 'hostname': gethostname()}
    run_name = f'{kwargs}' 
    # run = wandb.init(mode=kwargs.get('mode', 'online'), project="zip-fit-train", name=run_name, save_code=True, config=kwargs)
    run = wandb.init(mode=kwargs.get('mode', 'dryrun'), project="zip-fit-pass-at-k-af", name=run_name, save_code=True, config=kwargs)
    wandb.save(__file__) # save current code now, don't wait to wandb.finish, also useful: wandb.save("*.py") # upload all .py files in current directory
    print(f'Kwargs to run:\n{kwargs}')
    # main(kwargs)
    main_train(kwargs)
    # main_full_run(kwargs)
    run.alert(title="Run Completed", text=f"Run finished, run url: {run.get_url()}")
    print(f'{run.get_url()=}')
    wandb.finish()

if __name__ == "__main__":
    import fire
    import time
    start_time = time.time()
    fire.Fire(_main)
    print(f"\aTime taken: {time.time() - start_time:.2f} seconds, or {(time.time() - start_time) / 60:.2f} minutes, or {(time.time() - start_time) / 3600:.2f} hours.\a")
