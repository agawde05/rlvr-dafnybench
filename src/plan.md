# Plan of Attack

## Reward Function

Our reward function is divided into three parts:

1. Formatting Reward: Our model earns a reward of 0.1 when it correctly uses
   the <think> and <answer> tags in its response, and follows the rest of the
   formatting instructions, and 0 otherwise.

2. Verification Reward: Our model earns a reward of 1 if the final Dafny code
   inside the <answer> tags verifies successfully using the Dafny verifier,
   and 0 otherwise.

3. Assumption Reward: Our model incurs a -1 penalty if the final Dafny code
   introduces any 'assume' statements that were not present in the original.
   Otherwise, it gets a 0.5 reward.

4. 