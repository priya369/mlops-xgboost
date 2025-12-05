#!/usr/bin/env python
"""
Works with google-cloud-aiplatform >= 1.30.0
"""

import os
import sys
import time

# Configuration
PROJECT_ID = os.getenv('GCP_PROJECT_ID') or os.getenv('GOOGLE_CLOUD_PROJECT')
if not PROJECT_ID:
    PROJECT_ID = input("Enter your GCP Project ID: ")

REGION = 'us-central1'
BUCKET_NAME = f"{PROJECT_ID}-mlops-bucket"
BQ_TABLE = f'{PROJECT_ID}.mlops_project.house_prices'

print("="*80)
print("XGBoost MLOps Pipeline - Corrected Submission")
print("="*80)

# Step 1: Verify pipeline.yaml exists
print("\n✓ Step 1: Checking pipeline.yaml...")
if not os.path.exists('pipeline.yaml'):
    print("❌ pipeline.yaml not found!")
    print("   Run: python pipeline.py")
    sys.exit(1)

print(f"✓ Found pipeline.yaml ({os.path.getsize('pipeline.yaml')} bytes)")

# Step 2: Verify YAML structure
print("\n✓ Step 2: Verifying YAML structure...")
try:
    import yaml
    with open('pipeline.yaml', 'r') as f:
        pipeline_spec = yaml.safe_load(f)
    
    # Verify required structure
    checks = [
        ('pipelineSpec', 'pipelineSpec' in pipeline_spec),
        ('root', 'root' in pipeline_spec.get('pipelineSpec', {})),
        ('components', 'components' in pipeline_spec.get('pipelineSpec', {})),
        ('deploymentSpec', 'deploymentSpec' in pipeline_spec),
    ]
    
    all_ok = True
    for check_name, result in checks:
        status = "✓" if result else "✗"
        print(f"  {status} {check_name}")
        if not result:
            all_ok = False
    
    if not all_ok:
        print("\n❌ YAML structure is invalid!")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ YAML parsing error: {str(e)}")
    sys.exit(1)

# Step 3: Submit to Vertex AI
print("\n🚀 Step 3: Submitting to Vertex AI...")

try:
    from google.cloud import aiplatform as aip
    
    # Initialize
    aip.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=f'gs://{BUCKET_NAME}'
    )
    
    print(f"✓ Initialized Vertex AI")
    print(f"  Project: {PROJECT_ID}")
    print(f"  Region: {REGION}")
    print(f"  Bucket: gs://{BUCKET_NAME}")
    
    # Create pipeline job
    print(f"\n✓ Creating pipeline job...")
    job = aip.PipelineJob(
        display_name='xgboost-mlops-pipeline-v2',
        template_path='pipeline.yaml',
        pipeline_root=f'gs://{BUCKET_NAME}/pipeline_root/',
        parameter_values={
            'bq_table': BQ_TABLE
        }
    )
    
    print(f"✓ Pipeline job created")
    print(f"  Display name: xgboost-mlops-pipeline-v2")
    print(f"  Parameter - bq_table: {BQ_TABLE}")
    
    # Submit (without 'wait' parameter - it was removed in newer versions)
    print(f"\n✓ Submitting to Vertex AI...")
    job.submit()  # Use submit() instead of run(wait=False)
    
    print("\n" + "="*80)
    print("✅ SUCCESS! Pipeline submitted to Vertex AI")
    print("="*80)
    print(f"\nJob Resource Name: {job.resource_name}")
    print(f"Job ID: {job.name.split('/')[-1]}")
    
    print(f"\n📊 Monitor your pipeline:")
    print(f"   Web Console:")
    print(f"   https://console.cloud.google.com/vertex-ai/pipelines?project={PROJECT_ID}")
    
    print(f"\n📋 Command line:")
    print(f"   gcloud ai pipelines list --region={REGION} --project={PROJECT_ID}")
    print(f"   gcloud ai pipelines describe {job.name.split('/')[-1]} --region={REGION}")
    
    print(f"\n⏱️  Pipeline Status (check after ~2 minutes):")
    print(f"   gcloud ai pipelines describe {job.name.split('/')[-1]} \\")
    print(f"     --region={REGION} \\")
    print(f"     --format='value(state)'")
    
    print(f"\n📈 View Logs:")
    print(f"   gcloud ai pipelines describe {job.name.split('/')[-1]} \\")
    print(f"     --region={REGION} \\")
    print(f"     --format='value(error)'")
    
except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")
    print("\nTroubleshooting:")
    print("1. Verify 'pipeline.yaml' exists in current directory: ls -lh pipeline.yaml")
    print("2. Check aiplatform SDK version:")
    print("   python -c \"import google.cloud.aiplatform; print(google.cloud.aiplatform.__version__)\"")
    print("3. Update SDK if needed:")
    print("   pip install --upgrade google-cloud-aiplatform")
    print("4. Verify gcloud authentication:")
    print("   gcloud auth list")
    print("5. Check if APIs are enabled:")
    print("   gcloud services enable aiplatform.googleapis.com bigquery.googleapis.com")
    print("6. Verify project:")
    print(f"   gcloud config get-value project")
    sys.exit(1)

print("\n" + "="*80)
