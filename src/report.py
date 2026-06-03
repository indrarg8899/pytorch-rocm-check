"""HTML/JSON report generator."""
import json
import os
from datetime import datetime
from typing import List, Optional, Any


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ROCm Compatibility Report</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: #0d1117; color: #c9d1d9; padding: 2rem; }
.container { max-width: 1000px; margin: 0 auto; }
h1 { color: #58a6ff; margin-bottom: 1rem; }
.score { font-size: 3rem; font-weight: bold; text-align: center;
         padding: 2rem; border-radius: 12px; margin: 1rem 0; }
.score-good { background: #0d2818; color: #3fb950; border: 2px solid #238636; }
.score-warn { background: #2d1b00; color: #d29922; border: 2px solid #9e6a03; }
.score-bad { background: #2d0000; color: #f85149; border: 2px solid #da3633; }
.stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 1rem 0; }
.stat { background: #161b22; padding: 1rem; border-radius: 8px; text-align: center; }
.stat-val { font-size: 1.5rem; font-weight: bold; color: #58a6ff; }
.stat-label { font-size: 0.85rem; color: #8b949e; }
.issues { margin-top: 1.5rem; }
.issue { background: #161b22; padding: 0.8rem; margin: 0.5rem 0;
         border-radius: 6px; border-left: 4px solid; }
.issue.error { border-color: #f85149; }
.issue.warning { border-color: #d29922; }
.issue.info { border-color: #58a6ff; }
.meta { color: #8b949e; font-size: 0.85rem; margin-top: 2rem; }
</style>
</head>
<body>
<div class="container">
<h1>🔧 ROCm Compatibility Report</h1>
<p class="meta">Generated: {{ timestamp }}</p>
{{ content }}
<p class="meta">pytorch-rocm-check v1.0.0</p>
</div>
</body>
</html>"""


class ReportGenerator:
    def __init__(self):
        self.results: List[Any] = []

    def add_result(self, result):
        self.results.append(result)

    def generate_html(self, output_path: str):
        sections = []
        for result in self.results:
            result.calculate_score()
            score_class = "score-good" if result.score >= 80 else "score-warn" if result.score >= 50 else "score-bad"

            section = f"""
<div class="score {score_class}">{result.score:.1f}/100</div>
<h2>{result.target}</h2>
<div class="stats">
  <div class="stat"><div class="stat-val">{result.compatible_ops}</div><div class="stat-label">Compatible</div></div>
  <div class="stat"><div class="stat-val">{result.partial_ops}</div><div class="stat-label">Partial</div></div>
  <div class="stat"><div class="stat-val">{result.incompatible_ops}</div><div class="stat-label">Incompatible</div></div>
  <div class="stat"><div class="stat-val">{result.error_count}</div><div class="stat-label">Errors</div></div>
</div>
<div class="issues">
"""
            for issue in result.issues:
                section += f"""<div class="issue {issue.severity}">
<strong>[{issue.category}]</strong> {issue.message}
"""
                if issue.suggestion:
                    section += f"<br><em>→ {issue.suggestion}</em>"
                section += "</div>\n"
            section += "</div>"
            sections.append(section)

        html = HTML_TEMPLATE.replace(
            "{{ content }}", "\n".join(sections)
        ).replace(
            "{{ timestamp }}", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(html)

    def generate_json(self, output_path: str):
        data = {
            "generated_at": datetime.now().isoformat(),
            "tool_version": "1.0.0",
            "results": [],
        }
        for result in self.results:
            result.calculate_score()
            data["results"].append({
                "target": result.target,
                "score": result.score,
                "compatible_ops": result.compatible_ops,
                "partial_ops": result.partial_ops,
                "incompatible_ops": result.incompatible_ops,
                "errors": result.error_count,
                "warnings": result.warning_count,
                "issues": [
                    {
                        "severity": i.severity,
                        "category": i.category,
                        "message": i.message,
                        "suggestion": i.suggestion,
                        "file": i.file,
                        "line": i.line,
                    }
                    for i in result.issues
                ],
            })

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
