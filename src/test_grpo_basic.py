# test_grpo_basic.py
import torch
import model as md

def test_initialization():
    """Test that GRPO trainer initializes correctly"""
    print("Testing initialization...")
    
    # Use a small model for testing
    model, tokenizer = md.load_model_and_tokenizer("gpt2", dtype=torch.float32, device=torch.device("cpu"))
     
    # Mock Dafny verifier
    class MockDafnyVerifier:
        def verify(self, dafny_file, timeout_seconds=30):
            return True
    
    from grpo import DafnyGRPOTrainer
    
    config = {
        'num_answers_per_question': 2,
        'max_gen_len': 10,
        'micro_batch_size': 1
    }
    
    trainer = DafnyGRPOTrainer(model, tokenizer, MockDafnyVerifier(), config)
    assert trainer is not None
    assert trainer.model is model
    assert trainer.tokenizer is tokenizer
    
    print("✅ Initialization test passed!")

def test_simple_generation():
    """Test that we can generate text"""
    print("Testing generation...")
    
    model, tokenizer = md.load_model_and_tokenizer("gpt2", dtype=torch.float32, device=torch.device("cpu"))
    
    class MockDafnyVerifier:
        def verify(self, dafny_file, timeout_seconds=30):
            return True
    
    from grpo import DafnyGRPOTrainer
    from data_types import Minibatch
    
    config = {
        'num_answers_per_question': 1,
        'max_gen_len': 5,  # Keep it short for testing
        'micro_batch_size': 1
    }
    
    trainer = DafnyGRPOTrainer(model, tokenizer, MockDafnyVerifier(), config)
    
    # Create a simple minibatch
    prompt = "method Test() { "
    prompt_tokens = tokenizer.tokenize(prompt)
    prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    
    minibatch = Minibatch(
        prompts=[prompt],
        prompt_tokens=[prompt_tokens],
        prompt_token_ids=[prompt_token_ids]
    )
    
    # Test generation
    responses = trainer.generate_responses(minibatch)
    assert len(responses) == 1
    assert responses[0].prompt == prompt
    assert len(responses[0].generated_token_ids) > 0
    
    print("✅ Generation test passed!")

if __name__ == "__main__":
    test_initialization()
    test_simple_generation()
    print("🎉 All basic tests passed!")