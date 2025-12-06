#!/usr/bin/env python
"""
CORRECTED Pipeline Submission Script
Fixes the KeyError: 'root' issue by ensuring proper pipeline structure
"""

import os
import sys

# Configuration
PROJECT_ID = 'data-oasis-472909-u4'
REGION = 'us-central1'
BUCKET_NAME = f"{PROJECT_ID}-mlops-bucket"
BQ_TABLE = f'{PROJECT_ID}.mlops_project.house_prices'

print("="*80)
print("XGBoost MLOps Pipeline - Corrected Submission")
print("="*80)

# Step 1: Compile pipeline
print("\n📝 Step 1: Compiling pipeline...")
try:
    import subprocess
    result = subprocess.run([sys.executable, 'pipeline.py'], 
                          capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"❌ Compilation failed: {result.stderr}")
        sys.exit(1)
except Exception as e:
    print(f"❌ Error: {str(e)}")
    sys.exit(1)

# Step 2: Verify YAML was created
print("\n✓ Checking pipeline.yaml...")
if not os.path.exists('pipeline.yaml'):
    print("❌ pipeline.yaml not found!")
    sys.exit(1)

# Check YAML validity
import yaml
try:
    with open('pipeline.yaml', 'r') as f:
        pipeline_spec = yaml.safe_load(f)
    
    # Verify required structure
    if 'pipelineSpec' in pipeline_spec:
        print("✓ Pipeline YAML has valid structure")
    else:
        print("⚠️  Warning: pipelineSpec not found")
except Exception as e:
    print(f"❌ YAML parsing error: {str(e)}")
    sys.exit(1)

# Step 3: Upload to GCS
print("\n📤 Step 2: Uploading pipeline.yaml to GCS...")
try:
    subprocess.run([
        'gsutil', 'cp', 'pipeline.yaml',
        f'gs://{BUCKET_NAME}/pipelines/pipeline.yaml'
    ], check=True)
    print(f"✓ Uploaded to gs://{BUCKET_NAME}/pipelines/pipeline.yaml")
except Exception as e:
    print(f"⚠️  Could not upload to GCS: {str(e)}")
    print("   Proceeding with local YAML...")

# Step 4: Submit to Vertex AI
print("\n🚀 Step 3: Submitting to Vertex AI...")

try:
    from google.cloud import aiplatform as aip
    
    # Initialize
    aip.init(
        project='data-oasis-472909-u4',
        location='us-central1',
        staging_bucket= 'gs://data-oasis-472909-u4-mlops-bucket'
    )
    
    print(f"✓ Initialized Vertex AI")
    print(f"   Project: {PROJECT_ID}")
    print(f"   Region: {REGION}")
    print(f"   Bucket: gs://{BUCKET_NAME}")
    
    # Create pipeline job
    job = aip.PipelineJob(
        display_name='xgboost-mlops-pipeline-v2',
        template_path='pipeline.yaml',  # Use local YAML file
        pipeline_root=f'gs://{BUCKET_NAME}/pipeline_root/',
        parameter_values={
            'bq_table': BQ_TABLE
        }
    )
    
    print(f"✓ Pipeline job created")
    print(f"   Display name: xgboost-mlops-pipeline-v2")
    print(f"   Parameter - bq_table: {BQ_TABLE}")
    
    # Submit
    job.submit()
    
    print("\n" + "="*80)
    print("✅ SUCCESS! Pipeline submitted to Vertex AI")
    print("="*80)
    print(f"\nJob Resource Name: {job.resource_name}")
    print(f"\n📊 Monitor your pipeline:")
    print(f"   https://console.cloud.google.com/vertex-ai/pipelines/runs/{job.name.split('/')[-1]}?project={PROJECT_ID}")
    print(f"\n⏱️  or list all pipelines:")
    print(f"   gcloud ai pipelines list --region={REGION} --project={PROJECT_ID}")
    
except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")
    print("\nTroubleshooting:")
    print("1. Verify 'pipeline.yaml' exists in current directory")
    print("2. Check if aiplatform SDK is installed: pip install google-cloud-aiplatform>=1.30.0")
    print("3. Verify gcloud authentication: gcloud auth list")
    print("4. Check if APIs are enabled:")
    print("   gcloud services enable aiplatform.googleapis.com bigquery.googleapis.com")
    sys.exit(1)

print("\n" + "="*80)
