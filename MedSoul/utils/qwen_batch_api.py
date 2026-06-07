"""Qwen Batch API wrapper for large-scale pseudo label generation"""
import os
import json
import time
from typing import List, Dict, Optional
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm


class QwenBatchLabeler:
    """Generate pseudo labels using Qwen Batch API (50% cost savings)"""
    
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str = "qwen-max",
        temperature: float = 0.0,
        max_tokens: int = 500,
        label_names: Optional[List[str]] = None,
        max_wait_hours: int = 24,
        poll_interval: int = 300
    ):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.label_names = label_names or []
        self.max_wait_hours = max_wait_hours
        self.poll_interval = poll_interval
    
    def create_prompt(self, report: str) -> str:
        """Create prompt for label extraction"""
        labels_str = ", ".join(self.label_names)
        
        prompt = f"""You are an expert radiologist. Based on the radiology report below, annotate each pathology with:
- 1.0: Explicitly mentioned as present in the report
- 0.0: Explicitly mentioned as absent (e.g., "no evidence of", "clear of")
- -1.0: Mentioned with uncertainty (e.g., "possible", "cannot exclude") or ambiguous language
- null: Not mentioned in the report at all

Pathology list: {labels_str}

Report:
{report}

Please output ONLY a valid JSON object with the pathology names as keys and their labels as values. Do not include any other text.
Example format: {{"Atelectasis": 1.0, "Cardiomegaly": 0.0, "Edema": -1.0, "Fracture": null}}
"""
        return prompt
    
    def create_jsonl_file(
        self,
        reports: List[str],
        output_path: str
    ) -> int:
        """
        Create JSONL file for batch API
        
        Returns:
            Number of requests created
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Creating JSONL file for {len(reports)} reports...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for idx, report in enumerate(tqdm(reports)):
                if not report or not report.strip():
                    continue
                
                prompt = self.create_prompt(report)
                
                request = {
                    "custom_id": str(idx),
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model_name,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens
                    }
                }
                
                f.write(json.dumps(request, ensure_ascii=False) + '\n')
        
        print(f"JSONL file created: {output_path}")
        return len(reports)
    
    def upload_file(self, file_path: str) -> str:
        """
        Upload JSONL file to Qwen
        
        Returns:
            file_id
        """
        print(f"Uploading file: {file_path}")
        
        with open(file_path, 'rb') as f:
            file_response = self.client.files.create(
                file=f,
                purpose='batch'
            )
        
        file_id = file_response.id
        print(f"File uploaded successfully. File ID: {file_id}")
        
        return file_id
    
    def create_batch_job(self, input_file_id: str) -> str:
        """
        Create batch inference job
        
        Returns:
            batch_id
        """
        print("Creating batch job...")
        
        # Format completion_window: must be between 24h and 14d
        hours = max(24, min(self.max_wait_hours, 336))  # 336h = 14d
        if hours >= 24 and hours % 24 == 0:
            completion_window = f"{hours // 24}d"
        else:
            completion_window = f"{hours}h"
        
        batch_response = self.client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window=completion_window
        )
        
        batch_id = batch_response.id
        print(f"Batch job created. Batch ID: {batch_id}")
        print(f"Status: {batch_response.status}")
        print(f"Completion window: {completion_window}")
        
        return batch_id
    
    def poll_batch_status(self, batch_id: str) -> Dict:
        """
        Poll batch job status until completion
        
        Returns:
            Final batch status info
        """
        print(f"\nPolling batch status (interval: {self.poll_interval}s)...")
        print("This may take a while depending on system load...")
        
        start_time = time.time()
        
        while True:
            batch_info = self.client.batches.retrieve(batch_id)
            status = batch_info.status
            
            elapsed_time = time.time() - start_time
            elapsed_hours = elapsed_time / 3600
            
            print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Status: {status}")
            print(f"Elapsed time: {elapsed_hours:.2f} hours")
            
            if hasattr(batch_info, 'request_counts'):
                counts = batch_info.request_counts
                print(f"Progress: Completed={counts.completed}, Failed={counts.failed}, Total={counts.total}")
            
            # Check if job is done
            if status == 'completed':
                print("\n✅ Batch job completed successfully!")
                return {
                    'status': status,
                    'output_file_id': batch_info.output_file_id,
                    'error_file_id': batch_info.error_file_id if hasattr(batch_info, 'error_file_id') else None,
                    'request_counts': batch_info.request_counts if hasattr(batch_info, 'request_counts') else None
                }
            
            elif status == 'failed':
                print("\n❌ Batch job failed!")
                error_info = batch_info.errors if hasattr(batch_info, 'errors') else "No error details available"
                print(f"Error: {error_info}")
                raise Exception(f"Batch job failed: {error_info}")
            
            elif status == 'expired':
                print(f"\n⏱️ Batch job expired after {self.max_wait_hours} hours")
                raise Exception(f"Batch job expired. Consider increasing max_wait_hours.")
            
            elif status == 'cancelled':
                print("\n🛑 Batch job was cancelled")
                raise Exception("Batch job was cancelled")
            
            # Wait before next poll
            time.sleep(self.poll_interval)
    
    def download_results(self, output_file_id: str, save_path: Optional[str] = None) -> Dict[str, Dict]:
        """
        Download and parse batch results
        
        Args:
            output_file_id: File ID of the batch output
            save_path: Optional path to save results. If None, results are not saved to disk.
        
        Returns:
            Dictionary mapping custom_id to labels
        """
        print(f"\nDownloading results from file: {output_file_id}")
        
        # Download file content
        file_content = self.client.files.content(output_file_id)
        
        # Parse JSONL results
        results = {}
        failed_count = 0
        
        for line in file_content.text.strip().split('\n'):
            try:
                result = json.loads(line)
                custom_id = result['custom_id']
                
                # Check if request succeeded
                if result.get('error'):
                    print(f"Warning: Request {custom_id} failed: {result['error']}")
                    failed_count += 1
                    continue
                
                # Extract response content
                response_content = result['response']['body']['choices'][0]['message']['content'].strip()
                
                # Parse JSON labels
                # Remove markdown code blocks if present
                if response_content.startswith("```"):
                    response_content = response_content.split("```")[1]
                    if response_content.startswith("json"):
                        response_content = response_content[4:]
                    response_content = response_content.strip()
                
                labels = json.loads(response_content)
                
                # Validate and fill missing labels
                for label_name in self.label_names:
                    if label_name not in labels:
                        labels[label_name] = None
                
                results[custom_id] = labels
                
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse result for custom_id {custom_id}: {e}")
                failed_count += 1
            except Exception as e:
                print(f"Warning: Error processing result for custom_id {custom_id}: {e}")
                failed_count += 1
        
        print(f"\n✅ Successfully parsed {len(results)} labels")
        if failed_count > 0:
            print(f"⚠️  {failed_count} requests failed or couldn't be parsed")
        
        # Save results if save_path is provided
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"Results saved to: {save_path}")
        
        return results
    
    def batch_extract(
        self,
        reports: List[str],
        save_path: Optional[str] = None,
        temp_dir: str = "temp_batch"
    ) -> Dict[str, Dict]:
        """
        Main workflow: Create JSONL → Upload → Submit → Poll → Download
        
        Args:
            reports: List of radiology reports
            save_path: Optional path to save final pseudo labels. If None, results are not saved to disk.
            temp_dir: Directory for temporary files (JSONL)
        
        Returns:
            Dictionary mapping index to labels
        """
        temp_dir = Path(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        jsonl_path = temp_dir / "batch_requests.jsonl"
        
        try:
            # Step 1: Create JSONL file
            print("\n" + "="*70)
            print("STEP 1: Creating JSONL file")
            print("="*70)
            num_requests = self.create_jsonl_file(reports, str(jsonl_path))
            
            # Check file size
            file_size_mb = jsonl_path.stat().st_size / (1024 * 1024)
            print(f"JSONL file size: {file_size_mb:.2f} MB")
            
            if file_size_mb > 500:
                raise Exception(f"File size ({file_size_mb:.2f} MB) exceeds 500 MB limit. Please split data.")
            if num_requests > 50000:
                raise Exception(f"Number of requests ({num_requests}) exceeds 50,000 limit. Please split data.")
            
            # Step 2: Upload file
            print("\n" + "="*70)
            print("STEP 2: Uploading file to Qwen")
            print("="*70)
            input_file_id = self.upload_file(str(jsonl_path))
            
            # Step 3: Create batch job
            print("\n" + "="*70)
            print("STEP 3: Creating batch job")
            print("="*70)
            batch_id = self.create_batch_job(input_file_id)
            
            # Save batch_id for recovery
            batch_info_path = temp_dir / "batch_info.json"
            with open(batch_info_path, 'w') as f:
                json.dump({
                    'batch_id': batch_id,
                    'input_file_id': input_file_id,
                    'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
                }, f, indent=2)
            print(f"Batch info saved to: {batch_info_path}")
            print(f"💡 If interrupted, you can resume with batch_id: {batch_id}")
            
            # Step 4: Poll status
            print("\n" + "="*70)
            print("STEP 4: Waiting for batch completion")
            print("="*70)
            final_status = self.poll_batch_status(batch_id)
            
            # Step 5: Download results
            print("\n" + "="*70)
            print("STEP 5: Downloading results")
            print("="*70)
            results = self.download_results(final_status['output_file_id'], save_path)
            
            # Download error file if exists
            if final_status.get('error_file_id'):
                error_path = temp_dir / "batch_errors.jsonl"
                print(f"\n⚠️  Downloading error file to: {error_path}")
                error_content = self.client.files.content(final_status['error_file_id'])
                with open(error_path, 'w', encoding='utf-8') as f:
                    f.write(error_content.text)
            
            print("\n" + "="*70)
            print("✅ BATCH PROCESSING COMPLETED")
            print("="*70)
            print(f"Total requests: {num_requests}")
            print(f"Successful: {len(results)}")
            print(f"Cost savings: 50% compared to realtime API")
            
            return results
            
        except Exception as e:
            print(f"\n❌ Batch processing failed: {e}")
            raise
    
    def resume_batch(
        self,
        batch_id: str,
        save_path: Optional[str] = None
    ) -> Dict[str, Dict]:
        """
        Resume a previous batch job using batch_id
        
        Useful if the process was interrupted during polling
        
        Args:
            batch_id: Batch job ID to resume
            save_path: Optional path to save results. If None, results are not saved to disk.
        """
        print(f"Resuming batch job: {batch_id}")
        
        try:
            # Poll status
            final_status = self.poll_batch_status(batch_id)
            
            # Download results
            results = self.download_results(final_status['output_file_id'], save_path)
            
            return results
            
        except Exception as e:
            print(f"Failed to resume batch: {e}")
            raise

