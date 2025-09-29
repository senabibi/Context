import json
import os
import requests
import re
import time
from typing import Dict, List, Any
import random
from datetime import datetime, timedelta
import sys

# === MISTRAL CONFIGURATION ===
MISTRAL_API_KEY = "key"  
MISTRAL_CONFIG = {
    "model": "open-mixtral-8x7b",  # Mixtral 8x7B model
    "base_url": "https://api.mistral.ai/v1/chat/completions",
    "max_context_tokens": 32768,  # Mixtral context window
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 512
}

# === COST TRACKING (Mistral pricing) ===
PRICING = {
    "input_cost_per_1m": 0.7,    # $0.7 per 1M input tokens
    "output_cost_per_1m": 0.7    # $0.7 per 1M output tokens
}

# === FILE PATHS ===
DATA_DIR = "data"
OUTPUT_DIR = "summary"

PROMPT_FILES = {
    "numeric": os.path.join(DATA_DIR, "data\summary_numeric_base.json"),
    "text": os.path.join(DATA_DIR, "data\summary_text_base.json")
}


# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

class TerminalLogger:
    """Terminal output logger"""
    def __init__(self, log_file):
        self.log_file = log_file
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
    def __enter__(self):
        self.log_handle = open(self.log_file, 'w', encoding='utf-8')
        return self
        
    def __exit__(self, *args):
        self.log_handle.close()
        
    def write(self, text):
        # Write to both terminal and file
        self.original_stdout.write(text)
        self.original_stdout.flush()
        self.log_handle.write(text)
        self.log_handle.flush()
        
    def flush(self):
        self.original_stdout.flush()
        self.log_handle.flush()

class CostTracker:
    def __init__(self, budget_limit: float = 5.0):
        self.budget_limit = budget_limit
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.request_count = 0
        
    def add_usage(self, input_tokens: int, output_tokens: int):
        """Add token usage and calculate cost"""
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
        """Check if we can afford this request"""
        estimated_output = 100
        estimated_cost = ((estimated_input_tokens / 1_000_000) * PRICING["input_cost_per_1m"] + 
                         (estimated_output / 1_000_000) * PRICING["output_cost_per_1m"])
        
        safe_remaining = self.get_remaining_budget() * 0.9
        return estimated_cost <= safe_remaining
    
    def get_remaining_budget(self) -> float:
        """Return remaining budget"""
        return self.budget_limit - self.total_cost
    
    def print_summary(self):
        """Print cost summary"""
        print(f"\n COST SUMMARY:")
        print(f"   Total requests: {self.request_count}")
        print(f"   Input tokens: {self.total_input_tokens:,}")
        print(f"   Output tokens: {self.total_output_tokens:,}")
        print(f"   Total cost: ${self.total_cost:.4f}")
        print(f"   Remaining budget: ${self.get_remaining_budget():.4f}")
        print(f"   Budget utilization: {(self.total_cost/self.budget_limit)*100:.1f}%")

# Global cost tracker
cost_tracker = CostTracker(budget_limit=5.0)

class RateLimiter:
    """Rate limiter for Mistral API"""
    def __init__(self):
        self.last_request_time = 0
        self.min_delay = 1.5  # Moderate delay for Mistral
        
    def wait_if_needed(self):
        """Wait if needed to respect rate limits"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.min_delay:
            wait_time = self.min_delay - elapsed
            print(f" Rate limiting: waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
        
        self.last_request_time = time.time()

rate_limiter = RateLimiter()

def estimate_tokens(text: str) -> int:
    """Token estimation for Mistral (approximately 4 characters = 1 token)"""
    return max(1, int(len(text) / 4))

def call_mistral_api_with_retry(user_message: str, max_retries: int = 3) -> tuple:
    """Mistral API call with cost tracking"""
    
    if not MISTRAL_API_KEY or MISTRAL_API_KEY == "your_mistral_api_key_here":
        return "ERROR: Mistral API key not set", 0, 0
    
    # Token estimation and budget check
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
                "Authorization": f"Bearer {MISTRAL_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": MISTRAL_CONFIG["model"],
                "messages": [
                    {
                        "role": "user",
                        "content": user_message
                    }
                ],
                "max_tokens": MISTRAL_CONFIG["max_tokens"],
                "temperature": MISTRAL_CONFIG["temperature"],
                "top_p": MISTRAL_CONFIG["top_p"],
                "stream": False
            }
            
            print(f"     Mistral API call (Attempt {attempt + 1})")
            
            response = requests.post(
                MISTRAL_CONFIG["base_url"],
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"].strip()
                
                # Record token usage
                usage = result.get("usage", {})
                input_tokens = usage.get("prompt_tokens", estimated_input)
                output_tokens = usage.get("completion_tokens", estimate_tokens(content))
                
                cost_tracker.add_usage(input_tokens, output_tokens)
                
                return content, input_tokens, output_tokens
            
            elif response.status_code == 429:  # Rate limit
                print(f"     Rate limit (429) - Attempt {attempt + 1}")
                wait_time = (2 ** attempt) * 3 + random.uniform(0, 2)
                print(f"     Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                continue
            
            elif response.status_code == 401:
                return "ERROR: Invalid API key", 0, 0
            
            elif response.status_code == 400:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("message", "Bad Request")
                except:
                    error_msg = "Bad Request"
                return f"ERROR: {error_msg}", 0, 0
            
            else:
                error_msg = f"API Error {response.status_code}: {response.text}"
                print(f"     {error_msg}")
                
                if attempt == max_retries - 1:
                    return f"API_ERROR: {error_msg}", 0, 0
                
                time.sleep(5)
                continue
                
        except requests.exceptions.Timeout:
            print(f"     Timeout - Attempt {attempt + 1}")
            if attempt < max_retries - 1:
                time.sleep(10)
                continue
            return "API_ERROR: Timeout", 0, 0
            
        except Exception as e:
            error_msg = f"Request error: {str(e)}"
            print(f"     {error_msg}")
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            return f"API_ERROR: {error_msg}", 0, 0
    
    return "API_ERROR: Max retries exceeded", 0, 0

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

def process_summary_data(data: List[Dict], data_type: str) -> Dict[str, Any]:
    """Process summary-aided data"""
    
    scorer = score_numeric if data_type == "numeric" else score_text
    
    results = {
        "data_type": data_type,
        "model": MISTRAL_CONFIG["model"],
        "total_items": len(data),
        "completed_items": 0,
        "correct_answers": 0,
        "accuracy": 0.0,
        "cost_info": {
            "input_tokens": 0,
            "output_tokens": 0,
            "cost": 0.0
        },
        "items": []
    }
    
    print(f"\n{'='*60}")
    print(f" Processing {data_type.upper()} data")
    print(f" Scoring type: {data_type.capitalize()}")
    print(f" Total items: {len(data)}")
    print(f" Remaining budget: ${cost_tracker.get_remaining_budget():.4f}")
    print(f"{'='*60}")
    
    file_input_tokens = 0
    file_output_tokens = 0
    
    for i, item in enumerate(data, 1):
        if not cost_tracker.can_afford_request():
            print(f" Budget limit reached at item {i}!")
            break
        
        # Extract data from the summary structure
        stage_1 = item.get("stage_1", "")
        stage_2 = item.get("stage_2", {})
        
        summary = stage_2.get("summary", "")
        new_info = stage_2.get("new_info", "")
        question = stage_2.get("question", "")
        expected_answer = item.get("answer", "")
        
        # Construct the summary-aided prompt
        full_prompt = f"{stage_1}\n\n{summary} {new_info} {question}".strip()
        
        print(f"   Item {i}/{len(data)}: {item.get('title', f'Item {i}')}")
        print(f"    Question: {question[:80]}...")
        
        # API call
        response, input_tokens, output_tokens = call_mistral_api_with_retry(full_prompt)
        
        file_input_tokens += input_tokens
        file_output_tokens += output_tokens
        
        if "ERROR" in response:
            print(f"     API Error: {response}")
            if "Budget limit reached" in response:
                break
            continue
        
        score = scorer(response, expected_answer)
        
        print(f"     Expected: {expected_answer}")
        print(f"     Got: {response[:100]}{'...' if len(response) > 100 else ''}")
        print(f"     Score: {score}")
        
        item_result = {
            "id": item.get("id", i),
            "title": item.get("title", f"Item {i}"),
            "stage_1": stage_1,
            "stage_2": stage_2,
            "full_prompt": full_prompt,
            "response": response,
            "expected_answer": expected_answer,
            "score": score,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
        
        results["items"].append(item_result)
        results["completed_items"] += 1
        
        if score == 1:
            results["correct_answers"] += 1
    
    # Calculate final metrics
    if results["completed_items"] > 0:
        results["accuracy"] = results["correct_answers"] / results["completed_items"]
    
    results["cost_info"] = {
        "input_tokens": file_input_tokens,
        "output_tokens": file_output_tokens,
        "cost": (file_input_tokens/1_000_000 * PRICING["input_cost_per_1m"] + 
                file_output_tokens/1_000_000 * PRICING["output_cost_per_1m"])
    }
    
    print(f"\n RESULTS for {data_type.upper()} data:")
    print(f"   Completed items: {results['completed_items']}")
    print(f"   Correct answers: {results['correct_answers']}")
    print(f"   Accuracy: {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)")
    print(f"   Cost: ${results['cost_info']['cost']:.4f}")
    
    return results

def main():
    timestamp = int(time.time())
    terminal_log_file = os.path.join(OUTPUT_DIR, f"terminal_results_{timestamp}.txt")
    
    with TerminalLogger(terminal_log_file) as logger:
        sys.stdout = logger
        
        print(" MISTRAL MIXTRAL-8X7B SUMMARY-AIDED CONTEXT RETENTION TESTER")
        print(f" Model: {MISTRAL_CONFIG['model']}")
        print(f" Budget: ${cost_tracker.budget_limit:.2f}")
        print(f" Pricing: ${PRICING['input_cost_per_1m']}/1M input, ${PRICING['output_cost_per_1m']}/1M output tokens")
        print(f" Output directory: {OUTPUT_DIR}")
        
        if not MISTRAL_API_KEY or MISTRAL_API_KEY == "your_mistral_api_key_here":
            print("\n Mistral API key not found!")
            print("   Please set your API key in the MISTRAL_API_KEY variable")
            return
        
        start_time = time.time()
        all_results = {}
        
        # Process both numeric and text data using PROMPT_FILES dictionary
        for data_type, file_path in PROMPT_FILES.items():
            if os.path.exists(file_path):
                print(f"\n Loading {data_type} data from: {file_path}")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    results = process_summary_data(data, data_type)
                    all_results[f"{data_type}_based_results"] = results
                    
                except Exception as e:
                    print(f" Error processing {data_type} data: {e}")
            else:
                print(f" {data_type.capitalize()} file not found: {file_path}")
        
        # Save final results
        results_file = os.path.join(OUTPUT_DIR, f"mistral_summary_results_{timestamp}.json")
        
        final_output = {
            "experiment_type": "summary_aided_context_retention",
            "model": MISTRAL_CONFIG["model"],
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp).isoformat(),
            "cost_summary": {
                "total_cost": cost_tracker.total_cost,
                "total_input_tokens": cost_tracker.total_input_tokens,
                "total_output_tokens": cost_tracker.total_output_tokens,
                "request_count": cost_tracker.request_count,
                "budget_limit": cost_tracker.budget_limit,
                "budget_remaining": cost_tracker.get_remaining_budget()
            },
            "results": all_results
        }
        
        with open(results_file, "w", encoding='utf-8') as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        
        # Final summary
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n{'='*60}")
        print(f" FINAL RESULTS")
        print(f"{'='*60}")
        print(f" Total time: {duration/60:.1f} minutes")
        print(f" Data types processed: {len(all_results)}")
        
        # Cost summary
        cost_tracker.print_summary()
        
        # Overall statistics
        if all_results:
            print(f"\n Summary-Aided Results:")
            
            for result_type, result_data in all_results.items():
                if result_data.get("completed_items", 0) > 0:
                    accuracy = result_data.get("accuracy", 0)
                    completed = result_data.get("completed_items", 0)
                    correct = result_data.get("correct_answers", 0)
                    cost = result_data.get("cost_info", {}).get("cost", 0)
                    
                    print(f"   {result_type}: {accuracy:.3f} ({correct}/{completed}) - ${cost:.4f}")
        
        print(f"\n Results saved to: {results_file}")
        print(f" Terminal log saved to: {terminal_log_file}")
        
        sys.stdout = logger.original_stdout

if __name__ == "__main__":
    main()