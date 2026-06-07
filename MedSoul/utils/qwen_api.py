"""Qwen API wrapper for generating pseudo labels"""
import os
import json
import time
from typing import List, Dict, Optional
from openai import OpenAI
from tqdm import tqdm


class QwenLabeler:
    """Generate pseudo labels using Qwen LLM"""
    
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str = "qwen-max",
        temperature: float = 0.0,
        max_tokens: int = 500,
        label_names: Optional[List[str]] = None
    ):
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
                
                content = response.choices[0].message.content.strip()
                
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
                print(f"JSON parse error (attempt {attempt + 1}/{retry}): {e}")
                print(f"Response: {content[:200]}...")
                if attempt == retry - 1:
                    return None
                time.sleep(1)
            except Exception as e:
                print(f"API error (attempt {attempt + 1}/{retry}): {e}")
                if attempt == retry - 1:
                    return None
                time.sleep(2)
        
        return None
    
    def batch_extract(
        self,
        reports: List[str],
        batch_size: int = 10,
        save_path: Optional[str] = None
    ) -> Dict[str, Dict]:
        """Extract labels from multiple reports
        
        Returns:
            Dictionary mapping index to labels
        """
        results = {}
        
        print(f"Generating pseudo labels for {len(reports)} reports...")
        
        for i in tqdm(range(0, len(reports), batch_size)):
            batch_reports = reports[i:i + batch_size]
            
            for j, report in enumerate(batch_reports):
                idx = i + j
                labels = self.extract_labels(report)
                if labels:
                    results[str(idx)] = labels
                
                # Rate limiting
                time.sleep(0.5)
            
            # Save intermediate results
            if save_path and (i + batch_size) % 100 == 0:
                with open(save_path, 'w') as f:
                    json.dump(results, f, indent=2)
        
        # Final save
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Saved pseudo labels to {save_path}")
        
        return results
