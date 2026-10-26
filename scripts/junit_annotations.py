"""Turn failed tests in JUnit XML reports into GitHub Actions error annotations.

    python scripts/junit_annotations.py reports/*.xml

Annotations appear on the public run page, so anyone can see which test failed and why
without opening the job log (which needs a GitHub login).
"""

import sys
import xml.etree.ElementTree as ET


def escape(text: str) -> str:
    # Workflow-command escaping: the message must stay on one line.
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def main(paths: list[str]) -> int:
    failed = 0
    for path in paths:
        for case in ET.parse(path).getroot().iter("testcase"):
            for problem in (*case.findall("failure"), *case.findall("error")):
                failed += 1
                name = f"{case.get('classname')}::{case.get('name')}"
                detail = (problem.get("message") or "") + "\n" + (problem.text or "")
                print(f"::error title={escape(name)}::{escape(detail.strip()[:1500])}")
    print(f"{failed} failed test(s) annotated")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
