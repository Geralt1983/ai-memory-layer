#!/usr/bin/env python3
"""Serena AI Code Review Driver"""

import os, sys, json, yaml, argparse
from pathlib import Path
from typing import Dict, List, Any
import anthropic

def get_changed_files(base_branch: str, head_branch: str) -> List[str]:
    """Get list of changed files between branches."""
    import subprocess
    try:
        cmd = ["git", "diff", "--name-only", f"origin/{base_branch}...origin/{head_branch}"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return [f.strip() for f in result.stdout.split('\n') if f.strip()]
    except subprocess.CalledProcessError:
        return []

def read_file_diff(file_path: str, base_branch: str, head_branch: str) -> str:
    """Get diff for a specific file."""
    import subprocess
    try:
        cmd = ["git", "diff", f"origin/{base_branch}...origin/{head_branch}", "--", file_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError:
        return ""

def load_rubric(rubric_path: str) -> Dict[str, Any]:
    """Load review rubric from YAML."""
    with open(rubric_path, 'r') as f:
        return yaml.safe_load(f)

def build_review_prompt(rubric: Dict[str, Any], files: List[str], diffs: Dict[str, str]) -> str:
    """Build comprehensive review prompt."""
    
    criteria_text = ""
    for name, details in rubric['criteria'].items():
        criteria_text += f"\n**{name.title()} ({details['weight']}% weight):**\n"
        criteria_text += f"{details['description']}\n"
        for check in details['checks']:
            criteria_text += f"- {check}\n"
    
    files_text = ""
    for file_path in files[:10]:  # Limit to first 10 files
        if file_path in diffs and diffs[file_path]:
            files_text += f"\n### {file_path}\n```diff\n{diffs[file_path][:2000]}...\n```\n"
    
    return f"""# AI Memory Layer Code Review

You are Serena, an expert AI code reviewer for the AI Memory Layer project. Review the following code changes according to the specified criteria.

## Review Criteria
{criteria_text}

## Scoring Scale
- Excellent (90-100): Exceptional implementation
- Good (80-89): Solid implementation with minor issues  
- Needs Improvement (60-79): Functional but has notable issues
- Poor (0-59): Significant problems that need addressing

## Code Changes
{files_text}

## Instructions
1. Review each file against the criteria
2. Provide specific, actionable feedback
3. Give an overall score (0-100) and category
4. Highlight both strengths and areas for improvement
5. Focus on architecture, performance, reliability, and testing

## Response Format
**Overall Score:** [0-100]
**Category:** [excellent/good/needs_improvement/poor]

**Detailed Review:**
[Your detailed analysis here]

**Recommendations:**
- [Specific actionable items]

**Strengths:**
- [What's working well]
"""

def run_review(rubric: Dict[str, Any], files: List[str], diffs: Dict[str, str]) -> Dict[str, Any]:
    """Run AI review using Anthropic API."""
    
    client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    
    prompt = build_review_prompt(rubric, files, diffs)
    
    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            messages=[{
                "role": "user", 
                "content": prompt
            }]
        )
        
        review_text = message.content[0].text
        
        # Parse score from review
        score = 75  # Default score
        category = "good"
        
        for line in review_text.split('\n'):
            if line.startswith('**Overall Score:**'):
                try:
                    score = int(line.split(':')[1].strip())
                except:
                    pass
            elif line.startswith('**Category:**'):
                category = line.split(':')[1].strip()
        
        return {
            "score": score,
            "category": category,
            "review": review_text,
            "passed": score >= rubric.get('thresholds', {}).get('pass', 75)
        }
        
    except Exception as e:
        return {
            "score": 0,
            "category": "error",
            "review": f"Review failed: {e}",
            "passed": False
        }

def post_review_comment(pr_number: int, review_result: Dict[str, Any]) -> None:
    """Post review as PR comment using GitHub API."""
    if not pr_number:
        return
        
    import requests
    
    token = os.getenv('GITHUB_TOKEN')
    if not token:
        print("No GITHUB_TOKEN available, skipping PR comment")
        return
    
    repo = os.getenv('GITHUB_REPOSITORY', 'ai-memory-layer/ai-memory-layer')
    
    status_emoji = "✅" if review_result['passed'] else "❌"
    comment = f"""## {status_emoji} Serena AI Code Review

**Score:** {review_result['score']}/100 ({review_result['category']})

{review_result['review']}

---
*Automated review by Serena AI*
"""
    
    url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    try:
        response = requests.post(url, headers=headers, json={"body": comment})
        response.raise_for_status()
        print(f"Posted review comment to PR #{pr_number}")
    except Exception as e:
        print(f"Failed to post PR comment: {e}")

def main():
    parser = argparse.ArgumentParser(description='Run Serena AI Code Review')
    parser.add_argument('--pr-number', type=int, help='Pull request number')
    parser.add_argument('--base-branch', default='main', help='Base branch')
    parser.add_argument('--head-branch', help='Head branch')
    parser.add_argument('--rubric-path', default='.ai-review/rubric.yml', help='Rubric file path')
    
    args = parser.parse_args()
    
    # Load rubric
    rubric = load_rubric(args.rubric_path)
    
    # Get changed files
    files = get_changed_files(args.base_branch, args.head_branch)
    if not files:
        print("No changed files found")
        return
    
    print(f"Reviewing {len(files)} changed files...")
    
    # Get diffs for changed files
    diffs = {}
    for file_path in files:
        diffs[file_path] = read_file_diff(file_path, args.base_branch, args.head_branch)
    
    # Run review
    result = run_review(rubric, files, diffs)
    
    # Output results
    print(f"\n🤖 Serena AI Review Complete!")
    print(f"Score: {result['score']}/100 ({result['category']})")
    print(f"Status: {'✅ PASSED' if result['passed'] else '❌ NEEDS WORK'}")
    print(f"\n{result['review']}")
    
    # Post to PR if available
    if args.pr_number:
        post_review_comment(args.pr_number, result)
    
    # Exit with appropriate code
    sys.exit(0 if result['passed'] else 1)

if __name__ == '__main__':
    main()