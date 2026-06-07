"""DeepSeek API wrapper for generating pseudo labels"""
import os
import json
import time
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm


class DeepSeekLabeler:
    """Generate pseudo labels using DeepSeek LLM"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        model_name: str = "deepseek-chat",
        temperature: float = 0.0,
        max_tokens: int = 16000,  # Large value for reasoner (thinking chain + answer can be long)
        label_names: Optional[List[str]] = None
    ):
        """
        Initialize DeepSeek Labeler
        
        Args:
            api_key: DeepSeek API key. If None, reads from DEEPSEEK_API_KEY env variable
            base_url: DeepSeek API base URL
            model_name: DeepSeek model name (default: deepseek-chat)
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Maximum tokens in response
            label_names: List of pathology label names
        """
        # Get API key from parameter or environment
        if api_key is None:
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError(
                    "DeepSeek API key not found. Please set DEEPSEEK_API_KEY environment variable "
                    "or pass api_key parameter."
                )
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.label_names = label_names or []
    
    def create_prompt(self, report: str) -> str:
        """Create prompt for label extraction"""
        labels_str = ", ".join(self.label_names)
        
        prompt = f"""You are an expert radiologist. Based on the radiology report below, annotate each pathology with:
1.0 - The label was positively mentioned in the associated study, and is present in one or more of the corresponding images. e.g. "A large pleural effusion"

0.0 - The label was negatively mentioned in the associated study, and therefore should not be present in any of the corresponding images. e.g. "No pneumothorax."

-1.0 - The label was either: (1) mentioned with uncertainty in the report, and therefore may or may not be present to some degree in the corresponding image, or (2) mentioned with ambiguous language in the report and it is unclear if the pathology exists or not
    - Explicit uncertainty: "The cardiac size cannot be evaluated."
    - Ambiguous language: "The cardiac contours are stable." e.g. "The cardiac contours are stable."
-null - No mention of the label was made in the report

Pathology list: {labels_str}

Report:
{report}

Please output ONLY a valid JSON object with the pathology names as keys and their labels as values. Do not include any other text.
Example format: {{"Atelectasis": 1.0, "Cardiomegaly": 0.0, "Edema": -1.0, "Fracture": null}}
"""
        return prompt
    
    def extract_labels(self, report: str, retry: int = 3) -> Optional[Dict]:
        """Extract labels from a single report"""
        if not report or not report.strip():
            return None
        
        prompt = self.create_prompt(report)
        
        for attempt in range(retry):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                # For reasoning models (deepseek-reasoner):
                # According to docs: reasoning_content = thinking, content = final answer
                # BUT in practice: everything is in reasoning_content, content is empty
                message = response.choices[0].message
                
                # Try content first (official behavior)
                content = message.content.strip() if message.content else ""
                
                # If content is empty, use reasoning_content (actual behavior)
                if not content:
                    if hasattr(message, 'reasoning_content') and message.reasoning_content:
                        content = message.reasoning_content.strip()
                    else:
                        raise ValueError("Empty response from API (both content and reasoning_content are empty)")
                
                # Check if content is still empty
                if not content:
                    raise ValueError("Empty response from API")
                
                # For reasoning models, extract JSON from the content
                # The reasoner model puts thinking first, then the answer
                # We need to find the JSON at the END of the content
                json_content = None
                
                # 1. Try markdown code blocks (```json...```)
                if "```json" in content:
                    # Get the LAST occurrence
                    parts = content.split("```json")
                    if len(parts) > 1:
                        json_content = parts[-1].split("```")[0].strip()
                
                # 2. Try generic code blocks (```...```)
                if json_content is None and "```" in content and content.count("```") >= 2:
                    # Get content between last pair of ```
                    parts = content.split("```")
                    for i in range(len(parts) - 1, 0, -2):  # Work backwards
                        candidate = parts[i-1].strip()
                        if candidate.startswith("json"):
                            candidate = candidate[4:].strip()
                        if candidate.startswith("{") and "}" in candidate:
                            json_content = candidate
                            break
                
                # 3. Extract last JSON object from content (for reasoner output)
                if json_content is None and "{" in content and "}" in content:
                    # Find the LAST complete JSON object
                    last_open = content.rfind("{")
                    if last_open != -1:
                        # Find matching closing brace
                        brace_count = 0
                        start_idx = last_open
                        end_idx = -1
                        
                        for i in range(last_open, len(content)):
                            if content[i] == "{":
                                brace_count += 1
                            elif content[i] == "}":
                                brace_count -= 1
                                if brace_count == 0:
                                    end_idx = i + 1
                                    break
                        
                        if end_idx != -1:
                            json_content = content[start_idx:end_idx].strip()
                
                # Fallback: use entire content
                if json_content is None:
                    json_content = content
                
                labels = json.loads(json_content)
                
                # Validate keys
                for label_name in self.label_names:
                    if label_name not in labels:
                        labels[label_name] = None
                
                return labels
                
            except json.JSONDecodeError as e:
                # Print first error for debugging - show content (final answer) not reasoning
                if attempt == 0:
                    print(f"\n[DEBUG] JSON parse error:")
                    if 'message' in locals():
                        msg = locals()['message']
                        has_r = hasattr(msg, 'reasoning_content') and msg.reasoning_content
                        has_c = msg.content and msg.content.strip()
                        
                        if has_r:
                            print(f"  reasoning_content: EXISTS ({len(msg.reasoning_content)} chars)")
                        else:
                            print(f"  reasoning_content: EMPTY")
                        
                        if has_c:
                            print(f"  content (final answer): EXISTS ({len(msg.content)} chars)")
                            print(f"  Content:\n{msg.content}")
                        else:
                            print(f"  content (final answer): EMPTY - reasoning was truncated!")
                            
                    print(f"\nExtracted JSON attempt: {json_content[:500] if 'json_content' in locals() and json_content else 'N/A'}")
                if attempt == retry - 1:
                    return None
                time.sleep(0.5)  # Reduced delay for faster retries
            except Exception as e:
                # Print first error for debugging  
                if attempt == 0:
                    print(f"\n[DEBUG] API error: {e}")
                if attempt == retry - 1:
                    return None
                time.sleep(1)  # Reduced delay for faster retries
        
        return None
    
    def _extract_with_index(self, idx: int, report: str) -> tuple:
        """Helper function for parallel processing"""
        labels = self.extract_labels(report)
        return (idx, labels)
    
    def batch_extract(
        self,
        reports: List[str],
        batch_size: int = 10,
        save_path: Optional[str] = None,
        max_workers: int = 20
    ) -> Dict[str, Dict]:
        """Extract labels from multiple reports using parallel processing
        
        Args:
            reports: List of reports
            batch_size: Not used in parallel mode, kept for API compatibility
            save_path: Optional path to save results
            max_workers: Number of concurrent workers (default: 20)
        
        Returns:
            Dictionary mapping index to labels
        """
        results = {}
        
        # Use ThreadPoolExecutor for concurrent API calls
        failed_count = 0
        success_count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(self._extract_with_index, idx, report): idx
                for idx, report in enumerate(reports)
            }
            
            # Process completed tasks with progress bar
            for future in tqdm(as_completed(future_to_idx), total=len(reports), desc="Processing"):
                try:
                    idx, labels = future.result()
                    if labels:
                        results[str(idx)] = labels
                        success_count += 1
                        
                        # Show sample output every 20 successful labels
                        if success_count % 20 == 1 and success_count < 100:
                            print(f"\n[Sample #{success_count}] Successfully extracted labels:")
                            # Show first 3 labels as sample
                            sample_labels = dict(list(labels.items())[:3])
                            for k, v in sample_labels.items():
                                print(f"  {k}: {v}")
                    else:
                        failed_count += 1
                except Exception as e:
                    failed_count += 1
        
        # Print summary
        print(f"\nSuccessfully processed: {len(results)}/{len(reports)}")
        if failed_count > 0:
            print(f"Failed: {failed_count}")
        
        # Final save
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        return results

