#!/usr/bin/env python3
"""
Simple script to create a Hugging Face repository before uploading the model.
Use this if the main upload script fails with repository creation issues.
"""

from huggingface_hub import create_repo, HfApi
import sys

def create_repository(repo_name, private=False):
    """Create a Hugging Face repository"""
    
    print(f"Creating repository: {repo_name}")
    print(f"Private: {private}")
    
    try:
        # Create the repository
        repo_url = create_repo(
            repo_id=repo_name,
            repo_type="model",
            private=private,
            exist_ok=True  # Don't fail if already exists
        )
        
        print(f"Repository created successfully!")
        print(f"URL: {repo_url}")
        print(f"You can now run the upload script: python upload_to_huggingface.py")
        
        return True
        
    except Exception as e:
        print(f"Failed to create repository: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're logged in: huggingface-cli login")
        print("2. Check that the repository name is valid")
        print("3. Make sure you have permission to create repositories")
        return False

if __name__ == "__main__":
    # Default repository name - using correct username
    DEFAULT_REPO = "mrehank209/bk-classification-bart-two-stage"
    
    if len(sys.argv) > 1:
        repo_name = sys.argv[1]
    else:
        repo_name = DEFAULT_REPO
        print(f"Using default repository name: {repo_name}")
        print("To use a different name, run: python create_hf_repo.py YOUR_USERNAME/your-model-name")
    
    # Ask about privacy
    private_choice = input("Make repository private? (y/N): ").lower().strip()
    private = private_choice in ['y', 'yes']
    
    # Create the repository
    success = create_repository(repo_name, private=private)
    
    if success:
        print(f"\nRepository '{repo_name}' is ready!")
        print("Next step: Run 'python upload_to_huggingface.py' to upload your model")
    else:
        print(f"\nFailed to create repository. Please check the errors above.")
        sys.exit(1)
