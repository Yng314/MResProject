"""Qwen Preview API wrapper for generating pseudo labels with reasoning capability"""
import os
import json
import time
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm


class QwenPreviewLabeler:
    """Generate pseudo labels using Qwen Preview model (with thinking capability)
    
    Note: This model does NOT support batch API mode.
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str = "qwen3-max-preview",
        temperature: float = 0.0,
        max_tokens: int = 500,
        label_names: Optional[List[str]] = None,
        top_p: float = 0.8,
        thinking_budget: int = 500
    ):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.label_names = label_names or []
        self.top_p = top_p
        self.thinking_budget = thinking_budget
    
    def create_prompt(self, report: str) -> str:
        """Create prompt for label extraction"""
        labels_str = '", "'.join(self.label_names)
        labels_str = f'["{labels_str}"]'
        
        prompt = f"""You are an expert radiologist. Based on the radiology report below, annotate each pathology **using only the highest-priority available section of the report**, in this order:  
1. **Impression** section — if present, use ONLY this section.  
2. If no Impression is present, use the **Findings** section.  
3. If neither Impression nor Findings is present, use the other parts of the report.  

Apply the following labeling rules strictly:

- **1.0**: The pathology is explicitly and positively stated as present in the selected section (e.g., "large pleural effusion").
- **0.0**: The pathology is explicitly negated in the selected section (e.g., "no xxx").
- **-1.0**: The label was either: (1) mentioned with uncertainty in the report, and therefore may or may not be present to some degree in the corresponding image, or (2) mentioned with ambiguous language in the report and it is unclear if the pathology exists or not
	* Explicit uncertainty: "The cardiac size cannot be evaluated."
	* Ambiguous language: "The cardiac contours are stable."
- **null**: The pathology is not mentioned at all in the selected section.

**Special Rule for "No Finding"**:  
- Set to **1.0** *only if* the selected section explicitly states the study is normal, unremarkable, or has no acute/focal abnormalities.  
- Set to **null** in all other cases (including when any definite or uncertain pathology is described, or when support devices are present without pathology).  
- "No Finding" can **never** be 0.0 or -1.0.

Pathology list:  
{labels_str}

Report:
{report}

Output ONLY a valid JSON object with pathology names as keys and their labels (1.0, 0.0, -1.0, or null) as values. Do not include any other text.
"""
        return prompt
    
    def extract_labels(self, report: str, retry: int = 3) -> Optional[Dict]:
        """Extract labels from a single report using streaming API"""
        if not report or not report.strip():
            return None
        
        prompt = self.create_prompt(report)
        
        for attempt in range(retry):
            try:
                # Use streaming mode to handle reasoning + answer
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    stream=True,
                    extra_body={
                        "enable_thinking": True,
                        "thinking_budget": self.thinking_budget
                    }
                )
                
                # Collect only the answer content (ignore reasoning_content)
                answer_content = ""
                
                for chunk in completion:
                    if not chunk.choices:
                        continue
                    
                    delta = chunk.choices[0].delta
                    
                    # Only collect answer content, skip reasoning
                    if hasattr(delta, "content") and delta.content:
                        answer_content += delta.content
                
                content = answer_content.strip()
                
                # Try to parse JSON
                # Remove markdown code blocks if present
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                    content = content.strip()
                
                labels = json.loads(content)
                
                # Validate keys
                for label_name in self.label_names:
                    if label_name not in labels:
                        labels[label_name] = None
                
                return labels
                
            except json.JSONDecodeError as e:
                # Silently retry on JSON parse errors
                if attempt == retry - 1:
                    return None
                time.sleep(0.5)
            except Exception as e:
                # Silently retry on API errors
                if attempt == retry - 1:
                    return None
                time.sleep(1)
        
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
        
        Note: batch_size parameter is kept for API compatibility but not used.
        The Preview model does not support Batch API, but uses concurrent requests.
        
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
                except Exception:
                    # Silently skip errors
                    pass
        
        # Final save
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        return results

