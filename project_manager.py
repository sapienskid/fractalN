import os
import sys
import datetime
import subprocess
from typing import List

class ResearchProjectManager:
    def __init__(self, project_root: str):
        """
        Initialize the Research Project Manager
        
        :param project_root: Root directory of the research project
        """
        self.project_root = project_root
        self.weekly_reports_dir = os.path.join(project_root, 'weekly_reports')
    
    def create_weekly_report(self, goals: List[str] = None, accomplishments: List[str] = None):
        """
        Create a new weekly progress report
        
        :param goals: List of goals for the week
        :param accomplishments: List of accomplishments
        """
        today = datetime.date.today()
        report_filename = f"{today.strftime('%Y-%m-%d')}_weekly_report.md"
        report_path = os.path.join(self.weekly_reports_dir, report_filename)
        
        with open(report_path, 'w') as report_file:
            report_file.write(f"# Weekly Progress Report\n\n")
            report_file.write(f"## Date: {today}\n\n")
            
            report_file.write("### This Week's Objectives\n")
            if goals:
                for goal in goals:
                    report_file.write(f"- [ ] {goal}\n")
            else:
                report_file.write("- [ ] \n")
            
            report_file.write("\n### Accomplishments\n")
            if accomplishments:
                for accomplishment in accomplishments:
                    report_file.write(f"- {accomplishment}\n")
            else:
                report_file.write("- \n")
            
            report_file.write("\n### Challenges Encountered\n- \n")
            report_file.write("\n### Next Week's Goals\n- [ ] \n")
            report_file.write("\n### Notes and Insights\n- \n")
            report_file.write("\n### Supervisor Feedback\n- \n")
        
        # Automatically stage and commit the new report
        try:
            subprocess.run(["git", "add", report_path], check=True)
            subprocess.run(["git", "commit", "-m", f"Add weekly report for {today}"], check=True)
            print(f"Created and committed weekly report: {report_filename}")
        except subprocess.CalledProcessError as e:
            print(f"Error committing report: {e}")
    
    def manage_attachments(self, attachment_path: str, category: str = 'misc'):
        """
        Manage and track project attachments
        
        :param attachment_path: Path to the attachment file
        :param category: Category of the attachment (e.g., 'paper', 'data', 'misc')
        """
        if not os.path.exists(attachment_path):
            print(f"Error: File {attachment_path} does not exist.")
            return
        
        # Determine destination directory
        attachments_root = os.path.join(self.project_root, 'attachments')
        category_dir = os.path.join(attachments_root, category)
        os.makedirs(category_dir, exist_ok=True)
        
        # Generate a timestamped filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.basename(attachment_path)
        new_filename = f"{timestamp}_{filename}"
        destination = os.path.join(category_dir, new_filename)
        
        # Copy the file and track it with git
        try:
            import shutil
            shutil.copy2(attachment_path, destination)
            
            subprocess.run(["git", "add", destination], check=True)
            subprocess.run(["git", "commit", "-m", f"Add {category} attachment: {new_filename}"], check=True)
            
            print(f"Attachment added: {destination}")
        except Exception as e:
            print(f"Error managing attachment: {e}")
    
    def track_research_progress(self):
        """
        Generate a comprehensive progress tracking report
        """
        # Placeholder for more advanced progress tracking
        # Could include:
        # - Git commit analysis
        # - Code complexity metrics
        # - Research output tracking
        pass

def main():
    # Get project root (current directory or specified)
    project_root = os.getcwd()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'report':
            # Create a weekly report
            manager = ResearchProjectManager(project_root)
            manager.create_weekly_report()
        
        elif sys.argv[1] == 'attach':
            # Manage an attachment
            if len(sys.argv) < 4:
                print("Usage: python project_manager.py attach <file_path> <category>")
                sys.exit(1)
            
            manager = ResearchProjectManager(project_root)
            manager.manage_attachments(sys.argv[2], sys.argv[3])
    
    else:
        print("Research Project Management Tool")
        print("Commands:")
        print("  report           - Create a new weekly progress report")
        print("  attach <file> <category>  - Add and track a project attachment")

if __name__ == '__main__':
    main()