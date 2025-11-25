import re

SYSTEM_MESSAGE = """You are an LLM specialized in formal verification. 
Your task is to take existing Dafny code and produce a fully annotated version 
that enables the Dafny verifier to prove total correctness.

You MUST:
- Add assert statements, loop invariants, decreases clauses, pre/postconditions,
  frame specifications, and any auxiliary lemmas necessary for verification.
- Preserve the structure of the original program unless correctness requires a
  modification--at which point you should comment what was changed.
- Never delete user-provided logic unless it is provably dead or contradictory--
  if such a situation arises, comment out the code as opposed to deleting it.
- Produce ENTIRE updated Dafny file verbatim.

STRICT FORMAT:
You MUST think inside <think>...</think> tags.
Your final output MUST appear ONLY inside <answer>...</answer> tags.
No commentary, no explanations, no meta text in the final answer.

<think> is hidden from the user in deployment but will be visible during training.
Inside <think> you must:
- reason step-by-step to infer required invariants or assertions,
- identify verification barriers,
- plan modifications that guarantee termination and correctness.

Inside <answer> you must:
- output the full final Dafny code file, fully annotated and syntactically valid.

All invariants must be strong enough for Dafny to verify.
All modifications must preserve semantic meaning.

Your output MUST be deterministic and complete.
"""

USER_TEMPLATE = """Below is Dafny code that needs verification annotations.

Insert all necessary:
- requires/ensures
- assert statements
- loop invariants
- decreases clauses
- modifies/frame specifications
- auxiliary lemmas (only if needed)
- datatype or function constraints

Return the entire resulting file.

Follow the system instructions exactly.
Use <think> for chain-of-thought and <answer> for the final code.

----- BEGIN INPUT CODE -----
{dafny_code_snippet}
----- END INPUT CODE -----

<think>
"""

def format_reward_function(response: str) -> float:
    """The reward function for whether the LLM output is correctly formatted.

    Recall that the reward is 0.1 if the output is correctly formatted, and 0.0
    otherwise. We will also give partial credit for partially correct
    formatting, with an emphasis on having the answer tags correct.
    """
    think_regex = r"<think>(.*?)</think>"
    answer_regex = r"<answer>(.*?)</answer>"
    full_format = r"<think>(.*?)</think>\n<answer>(.*?)</answer>$"

    match_think = re.search(think_regex, response, re.DOTALL)
    match_answer = re.search(answer_regex, response, re.DOTALL)
    match_full = re.search(full_format, response, re.DOTALL)

    if match_full:
        return 0.1
    
    reward = 0.0
    if match_think:
        reward += 0.01
    if match_answer:
        reward += 0.07
    return reward
