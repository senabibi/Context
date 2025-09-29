import json
import os
import requests
import re
import time
from typing import Dict, List, Any
import random
from datetime import datetime, timedelta
import sys
from io import StringIO

# === OPENAI GPT-4 TURBO CONFIGURATION ===
OPENAI_API_KEY = ""  
OPENAI_CONFIG = {
    "model": "gpt-4-turbo-2024-04-09",
    "base_url": "https://api.openai.com/v1/chat/completions",
    "max_context_tokens": 128000,
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 512
}

# === COST TRACKING (OpenAI GPT-4 Turbo pricing) ===
PRICING = {
    "input_cost_per_1m": 10.0,    # $10 per 1M input tokens
    "output_cost_per_1m": 30.0    # $30 per 1M output tokens
}

# === FILE PATHS ===
BASE_DIR = "baseline"
DATA_DIR = "data"
RESULTS_DIR = BASE_DIR

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# === TERMINAL OUTPUT CAPTURE ===
class TerminalCapture:
    def __init__(self):
        self.terminal_output = []
        self.original_stdout = sys.stdout
        
    def write(self, text):
        self.original_stdout.write(text)
        self.terminal_output.append(text)
        
    def flush(self):
        self.original_stdout.flush()
        
    def get_output(self):
        return ''.join(self.terminal_output)
    
    def save_terminal_output(self, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.get_output())

# Global terminal capturer
terminal_capture = TerminalCapture()
sys.stdout = terminal_capture

class CostTracker:
    def __init__(self, budget_limit: float = 10.0):
        self.budget_limit = budget_limit
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.request_count = 0
        
    def add_usage(self, input_tokens: int, output_tokens: int):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.request_count += 1
        
        input_cost = (input_tokens / 1_000_000) * PRICING["input_cost_per_1m"]
        output_cost = (output_tokens / 1_000_000) * PRICING["output_cost_per_1m"]
        request_cost = input_cost + output_cost
        
        self.total_cost += request_cost
        
        print(f"     Request cost: ${request_cost:.4f} (Input: {input_tokens}, Output: {output_tokens})")
        print(f"     Total cost: ${self.total_cost:.4f} / ${self.budget_limit:.2f}")
        
        return request_cost
    
    def can_afford_request(self, estimated_input_tokens: int = 1000) -> bool:
        estimated_output = 100
        estimated_cost = ((estimated_input_tokens / 1_000_000) * PRICING["input_cost_per_1m"] + 
                         (estimated_output / 1_000_000) * PRICING["output_cost_per_1m"])
        
        safe_remaining = self.get_remaining_budget() * 0.9
        return estimated_cost <= safe_remaining
    
    def get_remaining_budget(self) -> float:
        return self.budget_limit - self.total_cost
    
    def print_summary(self):
        print(f"\n COST SUMMARY:")
        print(f"   Total requests: {self.request_count}")
        print(f"   Input tokens: {self.total_input_tokens:,}")
        print(f"   Output tokens: {self.total_output_tokens:,}")
        print(f"   Total cost: ${self.total_cost:.4f}")
        print(f"   Remaining budget: ${self.get_remaining_budget():.4f}")
        print(f"   Budget utilization: {(self.total_cost/self.budget_limit)*100:.1f}%")

# Global cost tracker
cost_tracker = CostTracker(budget_limit=10.0)

# === PROMPT FILES ===
PROMPT_FILES = {
    "numeric": os.path.join(DATA_DIR, "baseline_numeric_base.json"),
    "text": os.path.join(DATA_DIR, "baseline_text_base.json")
}

class RateLimiter:
    def __init__(self):
        self.last_request_time = 0
        self.min_delay = 2.0  
        
    def wait_if_needed(self):
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.min_delay:
            wait_time = self.min_delay - elapsed
            print(f" Rate limiting: {wait_time:.1f}s waiting...")
            time.sleep(wait_time)
        
        self.last_request_time = time.time()

rate_limiter = RateLimiter()

def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 3.5)

def call_openai_api_with_retry(user_message: str, max_retries: int = 3) -> tuple:
    
    if not OPENAI_API_KEY or OPENAI_API_KEY == "key":
        return "ERROR: OpenAI API key not set", 0, 0
    
    estimated_input = estimate_tokens(user_message)
    
    if not cost_tracker.can_afford_request(estimated_input):
        print(f" Budget limit reached! Estimated cost would exceed remaining budget.")
        cost_tracker.print_summary()
        return "ERROR: Budget limit reached", 0, 0
    
    for attempt in range(max_retries):
        try:
            # Rate limiting
            rate_limiter.wait_if_needed()
            
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": OPENAI_CONFIG["model"],
                "messages": [
                    {
                        "role": "user",
                        "content": user_message
                    }
                ],
                "max_tokens": OPENAI_CONFIG["max_tokens"],
                "temperature": OPENAI_CONFIG["temperature"],
                "top_p": OPENAI_CONFIG["top_p"],
                "stream": False
            }
            
            print(f"     OpenAI GPT-4 Turbo API call (Attempt {attempt + 1})")
            
            response = requests.post(
                OPENAI_CONFIG["base_url"],
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"].strip()
                
                usage = result.get("usage", {})
                input_tokens = usage.get("prompt_tokens", estimated_input)
                output_tokens = usage.get("completion_tokens", estimate_tokens(content))
                
                cost_tracker.add_usage(input_tokens, output_tokens)
                
                return content, input_tokens, output_tokens
            
            elif response.status_code == 429:  # Rate limit
                print(f"     Rate limit (429) - Attempt {attempt + 1}")
                wait_time = (2 ** attempt) * 3 + random.uniform(0, 2)
                print(f"     {wait_time:.1f} saniye bekleniyor...")
                time.sleep(wait_time)
                continue
            
            elif response.status_code == 401:
                return "ERROR: Invalid API key", 0, 0
            
            elif response.status_code == 400:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", {}).get("message", "Bad Request")
                except:
                    error_msg = "Bad Request"
                return f"ERROR: {error_msg}", 0, 0
            
            else:
                error_msg = f"API Error {response.status_code}: {response.text[:200]}"
                print(f"     {error_msg}")
                
                if attempt == max_retries - 1:
                    return f"API_ERROR: {error_msg}", 0, 0
                
                time.sleep(10)
                continue
                
        except requests.exceptions.Timeout:
            print(f"     Timeout - Attempt {attempt + 1}")
            if attempt < max_retries - 1:
                time.sleep(15)
                continue
            return "API_ERROR: Timeout", 0, 0
            
        except Exception as e:
            error_msg = f"Request error: {str(e)}"
            print(f"     {error_msg}")
            if attempt < max_retries - 1:
                time.sleep(10)
                continue
            return f"API_ERROR: {error_msg}", 0, 0
    
    return "API_ERROR: Max retries exceeded", 0, 0

def save_checkpoint(results: Dict, checkpoint_file: str = None):
    if checkpoint_file is None:
        checkpoint_file = os.path.join(RESULTS_DIR, "openai_checkpoint.json")
    
    checkpoint_data = {
        "timestamp": time.time(),
        "model": OPENAI_CONFIG["model"],
        "cost_summary": {
            "total_cost": cost_tracker.total_cost,
            "total_input_tokens": cost_tracker.total_input_tokens,
            "total_output_tokens": cost_tracker.total_output_tokens,
            "request_count": cost_tracker.request_count
        },
        "results": results
    }
    
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
    
    print(f" Checkpoint saved: ${cost_tracker.total_cost:.4f} spent")

def load_checkpoint(checkpoint_file: str = None) -> Dict:
    if checkpoint_file is None:
        checkpoint_file = os.path.join(RESULTS_DIR, "openai_checkpoint.json")
    
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            
            results = checkpoint.get("results", {})
            cost_summary = checkpoint.get("cost_summary", {})
            
            cost_tracker.total_cost = cost_summary.get("total_cost", 0)
            cost_tracker.total_input_tokens = cost_summary.get("total_input_tokens", 0)
            cost_tracker.total_output_tokens = cost_summary.get("total_output_tokens", 0)
            cost_tracker.request_count = cost_summary.get("request_count", 0)
            
            print(f" Checkpoint loaded: ${cost_tracker.total_cost:.4f} already spent")
            return results
            
        except Exception as e:
            print(f" Checkpoint load error: {e}")
    
    return {}

def score_text(response: str, expected: str) -> int:
    """Text scoring - case insensitive"""
    if "ERROR" in response:
        return 0

    resp_clean = response.lower().strip()
    exp_clean = expected.lower().strip()

    # Exact match
    if resp_clean == exp_clean:
        return 1

    # Substring match
    if exp_clean in resp_clean:
        return 1

    # Token-based match
    resp_tokens = re.findall(r'\w+', resp_clean)
    exp_tokens = re.findall(r'\w+', exp_clean)
    if set(exp_tokens).issubset(set(resp_tokens)):
        return 1

    return 0

def score_numeric(response: str, expected: str) -> int:
    """Numeric scoring with tolerance"""
    if "ERROR" in response:
        return 0

    try:
        resp_numbers = re.findall(r'-?\d+\.?\d*', response)
        exp_numbers = re.findall(r'-?\d+\.?\d*', expected)

        if not exp_numbers:
            return 0

        exp_num = float(exp_numbers[0])

        for r in resp_numbers:
            try:
                resp_num = float(r)
                if abs(resp_num - exp_num) < 0.01:
                    return 1
            except ValueError:
                continue

        return 0

    except Exception:
        return 0

def process_file(file_path: str, file_type: str) -> Dict[str, Any]:
    if not os.path.exists(file_path):
        print(f" File not found: {file_path}")
        return {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    file_name = os.path.basename(file_path)
    is_numeric = file_type == "numeric"
    scorer = score_numeric if is_numeric else score_text
    
    results = {
        "file": file_name,
        "file_type": file_type,
        "model": OPENAI_CONFIG["model"],
        "steps": [],
        "accuracy": 0,
        "total_prompts": 0,
        "correct_answers": 0,
        "processing_time": 0,
        "cost_info": {
            "input_tokens": 0,
            "output_tokens": 0,
            "cost": 0.0
        }
    }
    
    print(f"\n{'='*60}")
    print(f" Processing: {file_name} ({file_type.upper()})")
    print(f" Scoring type: {'Numeric' if is_numeric else 'Text'}")
    print(f" Remaining budget: ${cost_tracker.get_remaining_budget():.4f}")
    print(f"{'='*60}")
    
    start_time = time.time()
    file_input_tokens = 0
    file_output_tokens = 0
    
    try:
        if isinstance(data, list):
            prompt_items = data
        else:
            print(f" Unexpected data format in {file_name}")
            return results

        print(f" Total prompts found: {len(prompt_items)}")
        
        max_affordable_requests = min(len(prompt_items), 
                                    int(cost_tracker.get_remaining_budget() * 0.8 / 0.015)) 
        
        if max_affordable_requests < len(prompt_items):
            print(f" Budget limitation: Processing only {max_affordable_requests}/{len(prompt_items)} prompts")
            prompt_items = prompt_items[:max_affordable_requests]
        
        for i, prompt_data in enumerate(prompt_items, 1):
            if not isinstance(prompt_data, dict):
                continue
            
            if i % 3 == 0 and not cost_tracker.can_afford_request():
                print(f" Budget limit reached at prompt {i}!")
                break
            
            needle = prompt_data.get('needle', '')
            haystack = prompt_data.get('haystack', '')
            question = prompt_data.get('question', '')
            
            full_prompt = f"{needle}\n\n{haystack}\n\n{question}".strip()
            
            print(f"   Prompt {i}/{len(prompt_items)}: {question[:50]}...")
            
            response, input_tokens, output_tokens = call_openai_api_with_retry(full_prompt)
            
            file_input_tokens += input_tokens
            file_output_tokens += output_tokens
            
            if "ERROR" in response:
                print(f"     API Error: {response}")
                if "Budget limit reached" in response:
                    break
                continue
            
            expected = prompt_data.get("answer", "")
            score = scorer(response, expected)
            
            print(f"     Expected: {expected}")
            print(f"     Got: {response[:100]}{'...' if len(response) > 100 else ''}")
            print(f"     Score: {score}")
            
            results["steps"].append({
                "prompt_id": prompt_data.get("id", i),
                "question": question,
                "needle": needle[:100] + "..." if len(needle) > 100 else needle,
                "response": response,
                "expected": expected,
                "score": score,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            })
            
            results["total_prompts"] += 1
            if score == 1:
                results["correct_answers"] += 1
            
            if i % 3 == 0:
                temp_results = {f"{file_type}_based_results": results}
                save_checkpoint(temp_results)
            
        if results["total_prompts"] > 0:
            results["accuracy"] = results["correct_answers"] / results["total_prompts"]
        
        results["processing_time"] = time.time() - start_time
        results["cost_info"] = {
            "input_tokens": file_input_tokens,
            "output_tokens": file_output_tokens,
            "cost": (file_input_tokens/1_000_000 * PRICING["input_cost_per_1m"] + 
                    file_output_tokens/1_000_000 * PRICING["output_cost_per_1m"])
        }
        
        print(f"\n RESULTS for {file_name}:")
        print(f"   Total prompts: {results['total_prompts']}")
        print(f"   Correct answers: {results['correct_answers']}")
        print(f"   Accuracy: {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)")
        print(f"   File cost: ${results['cost_info']['cost']:.4f}")
        print(f"   Processing time: {results['processing_time']:.1f} seconds")

    except Exception as e:
        print(f" Error processing {file_name}: {e}")
        results["error"] = str(e)
    
    return results

def main():
    print(" OPENAI GPT-4 TURBO BASELINE CONTEXT RETENTION TESTER")
    print(f" Model: {OPENAI_CONFIG['model']}")
    print(f" Budget: ${cost_tracker.budget_limit:.2f}")
    print(f" Pricing: ${PRICING['input_cost_per_1m']}/1M input, ${PRICING['output_cost_per_1m']}/1M output tokens")
    print(f" GPT-4 Turbo is premium pricing - budget carefully!")
    
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
        print("\n OpenAI API key not found!")
        print("   Please set your API key in the OPENAI_API_KEY variable")
        return
    
    all_results = load_checkpoint()
    
    print(f" Processing baseline needle-in-haystack tests...")
    
    start_time = time.time()
    
    # Process numeric and text files
    for file_type, file_path in PROMPT_FILES.items():
        result_key = f"{file_type}_based_results"
        
        if result_key in all_results and all_results[result_key]:
            print(f" {file_type.upper()} file already processed!")
            continue
        
        print(f"\n Processing {file_type.upper()} baseline tests: {file_path}")
        
        if not cost_tracker.can_afford_request():
            print(f" Budget limit reached before processing {file_type} tests")
            break
        
        try:
            result = process_file(file_path, file_type)
            if result:
                all_results[result_key] = result
                
                save_checkpoint(all_results)
                
                if "error" not in result:
                    print(f" Successfully processed {file_type}: {result['accuracy']:.3f} accuracy")
                
                if cost_tracker.get_remaining_budget() < 2.0:  
                    print(f" Low budget warning: ${cost_tracker.get_remaining_budget():.4f} remaining")
            
        except Exception as e:
            print(f" Failed to process {file_type}: {e}")
            save_checkpoint(all_results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_file = os.path.join(RESULTS_DIR, f"openai_baseline_results_{timestamp}.json")
    final_output = {
        "experiment_type": "baseline_needle_in_haystack",
        "model": OPENAI_CONFIG["model"],
        "timestamp": time.time(),
        "experiment_config": {
            "max_context_tokens": OPENAI_CONFIG["max_context_tokens"],
            "temperature": OPENAI_CONFIG["temperature"],
            "top_p": OPENAI_CONFIG["top_p"],
            "max_tokens": OPENAI_CONFIG["max_tokens"]
        },
        "cost_summary": {
            "total_cost": cost_tracker.total_cost,
            "total_input_tokens": cost_tracker.total_input_tokens,
            "total_output_tokens": cost_tracker.total_output_tokens,
            "request_count": cost_tracker.request_count,
            "budget_limit": cost_tracker.budget_limit,
            "budget_remaining": cost_tracker.get_remaining_budget()
        },
        "numeric_based_results": all_results.get("numeric_based_results", {}),
        "text_based_results": all_results.get("text_based_results", {})
    }
    
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    terminal_output_file = os.path.join(RESULTS_DIR, f"terminal_results_{timestamp}.txt")
    terminal_capture.save_terminal_output(terminal_output_file)

    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n{'='*60}")
    print(f" BASELINE EXPERIMENT COMPLETED")
    print(f"{'='*60}")
    print(f" Total time: {duration/60:.1f} minutes")
    print(f" Data types processed: {len([k for k in all_results.keys() if k.endswith('_results')])}")
    
    cost_tracker.print_summary()
    
    if all_results:
        print(f"\n Baseline Results Summary:")
        
        for result_type in ["numeric_based_results", "text_based_results"]:
            if result_type in all_results and all_results[result_type]:
                result = all_results[result_type]
                if "error" not in result and result.get("total_prompts", 0) > 0:
                    file_cost = result.get("cost_info", {}).get("cost", 0)
                    print(f"   {result_type.replace('_results', '').upper()}: {result['accuracy']:.3f} ({result['correct_answers']}/{result['total_prompts']}) - ${file_cost:.4f}")
        
        # Overall baseline performance
        valid_results = [all_results[k] for k in all_results.keys() if k.endswith('_results') and 
                        "error" not in all_results[k] and all_results[k].get("total_prompts", 0) > 0]
        
        if valid_results:
            total_prompts = sum(r["total_prompts"] for r in valid_results)
            total_correct = sum(r["correct_answers"] for r in valid_results)
            overall_accuracy = total_correct / total_prompts if total_prompts > 0 else 0
            
            print(f"\n Overall Baseline Performance:")
            print(f"   Total prompts tested: {total_prompts}")
            print(f"   Total correct: {total_correct}")
            print(f"   Overall accuracy: {overall_accuracy:.3f} ({overall_accuracy*100:.1f}%)")
    
    print(f"\n Results saved to: {output_file}")
    print(f" Terminal output saved to: {terminal_output_file}")
    
    checkpoint_file = os.path.join(RESULTS_DIR, "openai_checkpoint.json")
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(" Checkpoint file cleaned up")

if __name__ == "__main__":
    main()