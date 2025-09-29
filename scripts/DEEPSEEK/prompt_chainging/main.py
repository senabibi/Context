import json
import os
import requests
import re
import time
from typing import Dict, List, Any
import random
from datetime import datetime
import sys

# === DEEPSEEK API CONFIGURATION ===
DEEPSEEK_API_KEY = "key"  

# === DEEPSEEK V3.1 Model Configuration ===
DEEPSEEK_CONFIG = {
    "model": "deepseek-chat",  # DeepSeek-V3.1 model
    "base_url": "https://api.deepseek.com/v1/chat/completions",
    "max_context_tokens": 128000,  # DeepSeek-V3.1 context length
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 512
}

# === CONSERVATIVE RATE LIMITING FOR BUDGET PROTECTION ===
RATE_LIMIT_CONFIG = {
    "requests_per_minute": 10,  # DeepSeek typically allows higher rates
    "max_retries": 3,
    "base_delay": 3,
    "max_delay": 20,
    "cost_per_million_input": 0.14,  # DeepSeek-V3.1 pricing (approximate)
    "cost_per_million_output": 0.28,
    "max_budget": 5.0,  # Increased budget due to lower costs
    "safety_margin": 0.5
}

# === FILE PATHS ===
PROMPT_FILES = {
    "Prompts\\data\\prompt_chainig_text_based.json",
    "Prompts\\data\\prompt_chainig_numeric_based.json"
}
OUTPUT_DIR = "DEEPSEEK\\prompt_chainging"

class TerminalLogger:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(output_dir, f"deepseek_terminal_results_{timestamp}.txt")
        
        # Redirect stdout to capture all terminal output
        self.original_stdout = sys.stdout
        self.log_buffer = []
        
    def write(self, text):
        # Write to original stdout (terminal)
        self.original_stdout.write(text)
        self.original_stdout.flush()
        
        # Store in buffer for later saving
        self.log_buffer.append(text)
    
    def flush(self):
        self.original_stdout.flush()
    
    def save_logs(self):
        """Save all captured terminal output to file"""
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write(''.join(self.log_buffer))
            print(f"Terminal logs saved to: {self.log_file}")
        except Exception as e:
            print(f"Error saving terminal logs: {e}")

class BudgetTracker:
    def __init__(self, max_budget: float):
        self.max_budget = max_budget
        self.current_cost = 0.0
        self.request_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
    
    def estimate_tokens(self, text: str) -> int:
        # DeepSeek uses similar tokenization to GPT models
        return int(len(text) / 3.5)
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        input_cost = (input_tokens / 1_000_000) * RATE_LIMIT_CONFIG["cost_per_million_input"]
        output_cost = (output_tokens / 1_000_000) * RATE_LIMIT_CONFIG["cost_per_million_output"]
        return input_cost + output_cost
    
    def can_afford_request(self, estimated_input_tokens: int, estimated_output_tokens: int = 512) -> bool:
        estimated_cost = self.calculate_cost(estimated_input_tokens, estimated_output_tokens)
        return (self.current_cost + estimated_cost) <= (self.max_budget - RATE_LIMIT_CONFIG["safety_margin"])
    
    def add_request_cost(self, input_tokens: int, output_tokens: int):
        cost = self.calculate_cost(input_tokens, output_tokens)
        self.current_cost += cost
        self.request_count += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        
        print(f"Request #{self.request_count}: ${cost:.6f} | Total: ${self.current_cost:.4f}/${self.max_budget:.2f}")
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "current_cost": self.current_cost,
            "max_budget": self.max_budget,
            "remaining_budget": self.max_budget - self.current_cost,
            "request_count": self.request_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens
        }

class RateLimiter:
    def __init__(self):
        self.request_times = []
        
    def wait_if_needed(self):
        current_time = time.time()
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        if len(self.request_times) >= RATE_LIMIT_CONFIG["requests_per_minute"]:
            wait_time = 60 - (current_time - self.request_times[0]) + 2
            if wait_time > 0:
                print(f"Rate limit reached. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
        
        self.request_times.append(current_time)

# Global instances
budget_tracker = BudgetTracker(RATE_LIMIT_CONFIG["max_budget"])
rate_limiter = RateLimiter()

def call_deepseek_api_with_retry(messages: List[Dict]) -> tuple[str, int, int]:
    """DeepSeek API call with budget and rate limiting"""
    
    total_input_text = " ".join([msg["content"] for msg in messages])
    estimated_input_tokens = budget_tracker.estimate_tokens(total_input_text)
    
    if not budget_tracker.can_afford_request(estimated_input_tokens):
        print(f"Budget limit reached! Current: ${budget_tracker.current_cost:.4f}")
        return "BUDGET_EXCEEDED", 0, 0
    
    for attempt in range(RATE_LIMIT_CONFIG["max_retries"]):
        try:
            rate_limiter.wait_if_needed()
            
            headers = {
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # Format messages for DeepSeek API (OpenAI-compatible format)
            formatted_messages = []
            
            for msg in messages:
                formatted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            payload = {
                "model": DEEPSEEK_CONFIG["model"],
                "messages": formatted_messages,
                "max_tokens": DEEPSEEK_CONFIG["max_tokens"],
                "temperature": DEEPSEEK_CONFIG["temperature"],
                "top_p": DEEPSEEK_CONFIG["top_p"],
                "stream": False
            }
            
            print(f"    DeepSeek API call (Attempt {attempt + 1})")
            print(f"    Estimated input tokens: {estimated_input_tokens}")
            
            response = requests.post(
                DEEPSEEK_CONFIG["base_url"], 
                headers=headers, 
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"].strip()
                
                # Extract token usage from response
                usage = result.get("usage", {})
                input_tokens = usage.get("prompt_tokens", estimated_input_tokens)
                output_tokens = usage.get("completion_tokens", budget_tracker.estimate_tokens(content))
                
                budget_tracker.add_request_cost(input_tokens, output_tokens)
                
                return content, input_tokens, output_tokens
            
            elif response.status_code == 429:
                print(f"    Rate limit (429) - Attempt {attempt + 1}")
                delay = RATE_LIMIT_CONFIG["base_delay"] * (2 ** attempt) + random.uniform(0, 2)
                delay = min(delay, RATE_LIMIT_CONFIG["max_delay"])
                
                print(f"    Waiting {delay:.1f} seconds...")
                time.sleep(delay)
                continue
            
            else:
                print(f"    API Error {response.status_code}: {response.text}")
                if attempt == RATE_LIMIT_CONFIG["max_retries"] - 1:
                    return f"API_ERROR: {response.status_code}", 0, 0
                time.sleep(5)
                continue
                
        except requests.exceptions.Timeout:
            print(f"    Timeout - Attempt {attempt + 1}")
            if attempt < RATE_LIMIT_CONFIG["max_retries"] - 1:
                time.sleep(10)
                continue
            return "API_ERROR: Timeout", 0, 0
            
        except requests.exceptions.RequestException as e:
            print(f"    Request Error - Attempt {attempt + 1}: {str(e)[:100]}")
            if attempt < RATE_LIMIT_CONFIG["max_retries"] - 1:
                time.sleep(5)
                continue
            return f"API_ERROR: {str(e)[:100]}", 0, 0
            
        except Exception as e:
            print(f"    Unexpected Error: {str(e)[:100]}")
            return f"ERROR: {str(e)[:100]}", 0, 0
    
    return "API_ERROR: Max retries exceeded", 0, 0

def score_text(response: str, expected: str) -> int:
    """Text scoring - case insensitive, allows substring & token presence"""
    if "ERROR" in response or "BUDGET_EXCEEDED" in response:
        return 0

    resp_clean = response.lower().strip()
    exp_clean = expected.lower().strip()

    if resp_clean == exp_clean:
        return 1

    if exp_clean in resp_clean:
        return 1

    resp_tokens = re.findall(r'\w+', resp_clean)
    exp_tokens = re.findall(r'\w+', exp_clean)
    
    if len(exp_tokens) > 0:
        matched_tokens = sum(1 for token in exp_tokens if token in resp_tokens)
        if matched_tokens / len(exp_tokens) >= 0.7:
            return 1

    return 0

def score_numeric(response: str, expected: str) -> int:
    """Numeric scoring - checks if expected numbers are in response with tolerance"""
    if "ERROR" in response or "BUDGET_EXCEEDED" in response:
        return 0

    try:
        resp_numbers = re.findall(r'-?\d+\.?\d*', response)
        exp_numbers = re.findall(r'-?\d+\.?\d*', expected)

        if not exp_numbers:
            return 0

        exp_nums = []
        for num_str in exp_numbers:
            try:
                exp_nums.append(float(num_str))
            except ValueError:
                continue
        
        resp_nums = []
        for num_str in resp_numbers:
            try:
                resp_nums.append(float(num_str))
            except ValueError:
                continue

        if not exp_nums:
            return 0

        found_count = 0
        for exp_num in exp_nums:
            for resp_num in resp_nums:
                if abs(resp_num - exp_num) < 0.01:
                    found_count += 1
                    break
        
        if found_count >= len(exp_nums):
            return 1

        if found_count >= len(exp_nums) * 0.8:
            return 1

        return 0

    except Exception as e:
        print(f"    Scoring error: {e}")
        return 0

def process_prompt_chaining_file(file_path: str) -> Dict[str, Any]:
    """Process prompt chaining files with budget awareness"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    file_name = os.path.basename(file_path)
    is_numeric = "numeric" in file_name
    scorer = score_numeric if is_numeric else score_text
    
    results = {
        "file": file_name,
        "model": DEEPSEEK_CONFIG["model"],
        "chains": [],
        "accuracy": 0,
        "total_chains": 0,
        "correct_answers": 0,
        "processing_time": 0,
        "budget_status": {}
    }
    
    print(f"\n{'='*60}")
    print(f"Processing PROMPT CHAINING: {file_name}")
    print(f"Scoring type: {'Numeric' if is_numeric else 'Text'}")
    print(f"Budget status: ${budget_tracker.current_cost:.4f}/${budget_tracker.max_budget:.2f}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Handle both list and single object structures
        if isinstance(data, list):
            chains = data
        else:
            chains = [data]
        
        print(f"Total chains to process: {len(chains)}")
        
        for chain_idx, chain in enumerate(chains, 1):
            if budget_tracker.current_cost >= (budget_tracker.max_budget - RATE_LIMIT_CONFIG["safety_margin"]):
                print(f"Budget limit reached! Stopping at chain {chain_idx}")
                break
            
            print(f"\nProcessing chain {chain_idx}/{len(chains)}")
            
            # Extract prompts from chain structure
            prompts = chain.get("prompts", [])
            if not prompts:
                print(f"No prompts found in chain {chain_idx}")
                continue
                
            print(f"   Total prompts: {len(prompts)}")
            
            # Find the needle and answer
            needle = ""
            expected_answer = ""
            
            for prompt in prompts:
                if prompt.get("type") == "needle_introduction":
                    needle = prompt.get("prompt", "")
                elif prompt.get("type") == "recall_test":
                    expected_answer = prompt.get("answer", "")
            
            print(f"   Expected answer: {expected_answer}")
            
            # Conversation history
            messages = []
            chain_data = {
                "chain_id": chain.get("chain_id", chain_idx),
                "title": chain.get("title", f"Chain {chain_idx}"),
                "needle": needle,
                "expected_answer": expected_answer,
                "prompts": [],
                "final_score": 0,
                "budget_exceeded": False
            }
            
            # Process each prompt
            for prompt_idx, prompt in enumerate(prompts, 1):
                prompt_text = prompt.get("prompt", "")
                if not prompt_text:
                    print(f"  Empty prompt in step {prompt_idx}")
                    continue
                    
                print(f"  Step {prompt_idx}/{len(prompts)} - {prompt.get('type', 'unknown')}")
                print(f"     Prompt: {prompt_text[:80]}...")
                
                # Add message and make API call
                messages.append({"role": "user", "content": prompt_text})
                response, input_tokens, output_tokens = call_deepseek_api_with_retry(messages)
                
                if "BUDGET_EXCEEDED" in response:
                    chain_data["budget_exceeded"] = True
                    print(f"Budget exceeded during chain {chain_idx}, prompt {prompt_idx}")
                    break
                
                messages.append({"role": "assistant", "content": response})
                
                print(f"     Response: {response[:100]}{'...' if len(response) > 100 else ''}")

                # Save prompt information
                prompt_data = {
                    "step": prompt.get("step", prompt_idx),
                    "type": prompt.get("type", "unknown"),
                    "prompt": prompt_text,
                    "response": response,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens
                }
                
                # Score if this is the recall test
                if prompt.get("type") == "recall_test" and expected_answer:
                    score = scorer(response, expected_answer)
                    prompt_data["expected"] = expected_answer
                    prompt_data["score"] = score
                    chain_data["final_score"] = score
                    
                    print(f"     Expected: {expected_answer}")
                    print(f"     Final Score: {score}")
                    
                    results["total_chains"] += 1
                    if score == 1:
                        results["correct_answers"] += 1
                
                chain_data["prompts"].append(prompt_data)
                
                # Short pause between prompts
                if prompt_idx < len(prompts):
                    time.sleep(1)
            
            results["chains"].append(chain_data)
            
            # Check budget before continuing
            remaining_budget = budget_tracker.max_budget - budget_tracker.current_cost
            print(f"   Remaining budget: ${remaining_budget:.4f}")
            
            if remaining_budget < RATE_LIMIT_CONFIG["safety_margin"]:
                print(f"Safety margin reached! Stopping chains.")
                break
            
            # Pause between chains
            if chain_idx < len(chains):
                print("   3 second pause...")
                time.sleep(3)

        # Calculate accuracy
        if results["total_chains"] > 0:
            results["accuracy"] = results["correct_answers"] / results["total_chains"]
        
        results["processing_time"] = time.time() - start_time
        results["budget_status"] = budget_tracker.get_status()
        
        print(f"\nRESULTS for {file_name}:")
        print(f"  Total chains: {results['total_chains']}")
        print(f"  Correct answers: {results['correct_answers']}")
        print(f"  Accuracy: {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)")
        print(f"  Processing time: {results['processing_time']:.1f} seconds")
        print(f"  Total cost: ${budget_tracker.current_cost:.4f}")

    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        results["error"] = str(e)
        results["budget_status"] = budget_tracker.get_status()
        import traceback
        traceback.print_exc()
    
    return results

def save_results(all_results: List[Dict], output_dir: str):
    """Save results to separate files based on type"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Separate results by type
    numeric_results = []
    text_results = []
    
    for result in all_results:
        if "numeric" in result.get("file", "").lower():
            numeric_results.append(result)
        else:
            text_results.append(result)
    
    # Save numeric results
    if numeric_results:
        numeric_file = os.path.join(output_dir, f"deepseek_numeric_based_results_{timestamp}.json")
        with open(numeric_file, "w", encoding='utf-8') as f:
            json.dump({
                "model": DEEPSEEK_CONFIG["model"],
                "timestamp": datetime.now().isoformat(),
                "rate_limit_config": RATE_LIMIT_CONFIG,
                "budget_tracker": budget_tracker.get_status(),
                "results": numeric_results
            }, f, indent=2, ensure_ascii=False)
        print(f"Numeric results saved to: {numeric_file}")
    
    # Save text results
    if text_results:
        text_file = os.path.join(output_dir, f"deepseek_text_based_results_{timestamp}.json")
        with open(text_file, "w", encoding='utf-8') as f:
            json.dump({
                "model": DEEPSEEK_CONFIG["model"],
                "timestamp": datetime.now().isoformat(),
                "rate_limit_config": RATE_LIMIT_CONFIG,
                "budget_tracker": budget_tracker.get_status(),
                "results": text_results
            }, f, indent=2, ensure_ascii=False)
        print(f"Text results saved to: {text_file}")

def main():
    # Setup terminal logging
    terminal_logger = TerminalLogger(OUTPUT_DIR)
    sys.stdout = terminal_logger
    
    try:
        print("DEEPSEEK V3.1 PROMPT CHAINING EVALUATOR")
        print(f"Model: {DEEPSEEK_CONFIG['model']}")
        print(f"Budget: ${RATE_LIMIT_CONFIG['max_budget']:.2f} (Safety margin: ${RATE_LIMIT_CONFIG['safety_margin']:.2f})")
        print(f"Rate limit: {RATE_LIMIT_CONFIG['requests_per_minute']}/minute")
        print(f"Target files: {len(PROMPT_FILES)}")
        
        if DEEPSEEK_API_KEY == "key":
            print("\nPlease set your DEEPSEEK_API_KEY in the script!")
            print("   You can get it from: https://platform.deepseek.com/")
            return
        
        # Verify files exist
        missing_files = [f for f in PROMPT_FILES if not os.path.exists(f)]
        if missing_files:
            print(f"\nMissing files:")
            for f in missing_files:
                print(f"   {f}")
            
            response = input("\nContinue with available files? (y/N): ").strip().lower()
            if response != 'y':
                return
        
        available_files = [f for f in PROMPT_FILES if os.path.exists(f)]
        
        all_results = []
        start_time = time.time()
        
        for file_idx, file_path in enumerate(available_files, 1):
            print(f"\nProcessing file {file_idx}/{len(available_files)}: {os.path.basename(file_path)}")
            
            # Budget check before each file
            if budget_tracker.current_cost >= (budget_tracker.max_budget - RATE_LIMIT_CONFIG["safety_margin"]):
                print(f"Budget limit reached! Remaining files skipped.")
                break
            
            try:
                result = process_prompt_chaining_file(file_path)
                all_results.append(result)
                
                if "error" not in result:
                    print(f"Successfully processed: {result['accuracy']:.3f} accuracy")
                else:
                    print(f"Processing failed with error")
                
                # Pause between files
                if file_idx < len(available_files) and budget_tracker.current_cost < (budget_tracker.max_budget - RATE_LIMIT_CONFIG["safety_margin"]):
                    print("5 second pause between files...")
                    time.sleep(5)
                
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")

        # Save results
        save_results(all_results, OUTPUT_DIR)

        # Final statistics
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS - DEEPSEEK V3.1 PROMPT CHAINING")
        print(f"{'='*60}")
        print(f"Total time: {duration/60:.1f} minutes ({duration:.1f} seconds)")
        print(f"Files processed: {len(all_results)}")
        print(f"Total cost: ${budget_tracker.current_cost:.4f}/${budget_tracker.max_budget:.2f}")
        print(f"Total requests: {budget_tracker.request_count}")
        print(f"Total tokens: {budget_tracker.total_input_tokens + budget_tracker.total_output_tokens:,}")
        
        if all_results:
            valid_results = [r for r in all_results if "error" not in r and r.get("total_chains", 0) > 0]
            if valid_results:
                total_chains = sum(r["total_chains"] for r in valid_results)
                total_correct = sum(r["correct_answers"] for r in valid_results)
                overall_accuracy = total_correct / total_chains if total_chains > 0 else 0
                
                print(f"\nOverall Statistics:")
                print(f"  Total chains tested: {total_chains}")
                print(f"  Total correct: {total_correct}")
                print(f"  Overall accuracy: {overall_accuracy:.3f} ({overall_accuracy*100:.1f}%)")
                
                print(f"\nPer-file breakdown:")
                for result in valid_results:
                    print(f"  {result['file']}: {result['accuracy']:.3f} ({result['correct_answers']}/{result['total_chains']}) - {result['processing_time']:.1f}s")
        
        remaining_budget = budget_tracker.max_budget - budget_tracker.current_cost
        print(f"Remaining budget: ${remaining_budget:.4f}")
        print("DeepSeek V3.1 prompt chaining evaluation completed!")

    finally:
        # Restore stdout and save terminal logs
        sys.stdout = terminal_logger.original_stdout
        terminal_logger.save_logs()

if __name__ == "__main__":
    main()